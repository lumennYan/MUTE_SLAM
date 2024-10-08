import torch
import os
import time

from colorama import Fore, Style
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (matrix_to_cam_pose, cam_pose_to_matrix, get_samples)
from src.utils.datasets import get_dataset
from src.utils.Frame_Visualizer import Frame_Visualizer
from .encoding import SubMap


class Tracker(object):
    """
    Tracking main class.
    Args:
        cfg (dict): config dict
        args (argparse.Namespace): arguments
        slam (MUTE_SLAM): MUTE_SLAM object
    """
    def __init__(self, cfg, args, slam):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']

        self.idx = slam.idx
        self.output = slam.output
        self.verbose = slam.verbose
        self.renderer = slam.renderer
        self.gt_c2w_list = slam.gt_c2w_list
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.truncation = slam.truncation

        self.submap_dict_list = slam.submap_dict_list
        self.submap_list = []
        self.submap_bound_list = slam.submap_bound_list

        self.encoding_type = slam.encoding_type
        self.encoding_levels = slam.encoding_levels
        self.base_resolution = slam.base_resolution
        self.per_level_feature_dim = slam.per_level_feature_dim
        self.use_tcnn = slam.use_tcnn

        self.cam_lr_T = cfg['tracking']['lr_T']
        self.cam_lr_R = cfg['tracking']['lr_R']
        self.device = cfg['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.w_sdf_fs = cfg['tracking']['w_sdf_fs']
        self.w_sdf_center = cfg['tracking']['w_sdf_center']
        self.w_sdf_tail = cfg['tracking']['w_sdf_tail']
        self.w_depth = cfg['tracking']['w_depth']
        self.w_color = cfg['tracking']['w_color']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['tracking']['no_vis_on_first_frame']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, shuffle=False,
                                       num_workers=1, pin_memory=True, prefetch_factor=2)

        self.visualizer = Frame_Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.output, 'tracking_vis'), renderer=self.renderer,
                                           truncation=self.truncation, verbose=self.verbose, device=self.device)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
        self.decoders = self.shared_decoders


    def get_map_from_mapper(self):
        """
        Get the sub_maps and corresponding parameters from mapper
        Args:
            None
        Returns:
            None
        """
        for i in range(len(self.submap_list)):
            self.submap_list[i].load_state_dict(self.submap_dict_list[i])

        while len(self.submap_list) < len(self.submap_dict_list):
            self.submap_list.append(SubMap(device=self.device, boundary=self.submap_bound_list[len(self.submap_list)],
                                           use_tcnn=self.use_tcnn,
                                           encoding_type=self.encoding_type,
                                           input_dim=2,
                                           num_levels=self.encoding_levels,
                                           level_dim=self.per_level_feature_dim,
                                           base_resolution=self.base_resolution))
            self.submap_list[-1].load_state_dict(self.submap_dict_list[len(self.submap_list)-1])

    def sdf_losses(self, sdf, z_vals, gt_depth):
        """
        Computes the losses for a signed distance function (SDF) given its values, depth values and ground truth depth.

        Args:
        - self: instance of the class containing this method
        - sdf: a tensor of shape (R, N) representing the SDF values
        - z_vals: a tensor of shape (R, N) representing the depth values
        - gt_depth: a tensor of shape (R,) containing the ground truth depth values

        Returns:
        - sdf_losses: a scalar tensor representing the weighted sum of the free space, center, and tail losses of SDF
        """

        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation),
                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation),
                                torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) *
                                  (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)),
                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        center_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        tail_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))

        sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss

        return sdf_losses

    def optimize_tracking(self, cam_pose, gt_color, gt_depth, batch_size, optimizer):
        """
        Do one iteration of camera tracking. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            cam_pose (tensor): camera pose.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """

        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        c2w = cam_pose_to_matrix(cam_pose)
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(self.ignore_edge_H, H-self.ignore_edge_H,
                                                                                 self.ignore_edge_W, W-self.ignore_edge_W,
                                                                                 batch_size, H, W, fx, fy, cx, cy, c2w,
                                                                                 gt_depth, gt_color, device)
        depth_mask = (batch_gt_depth > 0)
        batch_gt_depth = batch_gt_depth[depth_mask]
        batch_gt_color = batch_gt_color[depth_mask]
        batch_rays_o = batch_rays_o[depth_mask]
        batch_rays_d = batch_rays_d[depth_mask]
        depth, color, sdf, z_vals, inmap_mask = self.renderer.render_batch_ray(self.submap_list, self.decoders, batch_rays_d, batch_rays_o,
                                                                   self.device, self.truncation, gt_depth=batch_gt_depth)

        ## Filtering the rays for which the rendered depth error is greater than 10 times of the median depth error
        batch_gt_depth = batch_gt_depth[inmap_mask]
        batch_gt_color = batch_gt_color[inmap_mask]

        depth_error = (batch_gt_depth - depth.detach()).abs()
        depth_error_median = depth_error.median()
        depth_mask = (depth_error < 20 * depth_error_median)

        mask = depth_mask

        ## SDF losses
        loss = self.sdf_losses(sdf[mask], z_vals[mask], batch_gt_depth[mask])

        ## Color Loss
        loss = loss + self.w_color * torch.square(batch_gt_color[mask] - color[mask]).mean()

        ### Depth loss
        loss = loss + self.w_depth * torch.square(batch_gt_depth[mask] - depth[mask]).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


    def run(self):
        """
            Runs the tracking thread for the input RGB-D frames.

            Args:
                None

            Returns:
                None
        """
        device = self.device

        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader, smoothing=0.05)

        for idx, gt_color, gt_depth, gt_c2w in pbar:
            gt_color = gt_color.to(device, non_blocking=True)
            gt_depth = gt_depth.to(device, non_blocking=True)
            gt_c2w = gt_c2w.to(device, non_blocking=True)

            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")
            idx = idx[0]

            # initiate mapping every self.every_frame frames
            if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                while self.mapping_idx[0] != idx - 1:
                    time.sleep(0.001)
                pre_c2w = self.estimate_c2w_list[idx - 1].unsqueeze(0).to(device)

            self.get_map_from_mapper()

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)

            if idx == 0 or self.gt_camera:
                c2w = gt_c2w
                if not self.no_vis_on_first_frame:
                    self.visualizer.save_imgs(idx, 0, gt_depth, gt_color, c2w.squeeze(), self.submap_list, self.decoders)

            else:
                if self.const_speed_assumption and idx - 2 >= 0:
                    ## Linear prediction for initialization
                    pre_poses = torch.stack([self.estimate_c2w_list[idx - 2], pre_c2w.squeeze(0)], dim=0)
                    pre_poses = matrix_to_cam_pose(pre_poses)
                    cam_pose = 2 * pre_poses[1:] - pre_poses[0:1]
                else:
                    ## Initialize with the last known pose
                    cam_pose = matrix_to_cam_pose(pre_c2w)
                T = torch.nn.Parameter(cam_pose[:, -3:].clone())
                R = torch.nn.Parameter(cam_pose[:,:4].clone())
                cam_para_list_T = [T]
                cam_para_list_R = [R]
                optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr_T, 'betas':(0.5, 0.999)},
                                                     {'params': cam_para_list_R, 'lr': self.cam_lr_R, 'betas':(0.5, 0.999)}])

                current_min_loss = torch.tensor(float('inf')).float().to(device)
                for cam_iter in range(self.num_cam_iters):
                    cam_pose = torch.cat([R, T], -1)

                    self.visualizer.save_imgs(idx, cam_iter, gt_depth, gt_color, cam_pose, self.submap_list, self.decoders)

                    loss = self.optimize_tracking(cam_pose, gt_color, gt_depth, self.tracking_pixels, optimizer_camera)
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_pose = cam_pose.clone().detach()

                c2w = cam_pose_to_matrix(candidate_cam_pose)

            self.estimate_c2w_list[idx] = c2w.squeeze(0).clone()
            self.gt_c2w_list[idx] = gt_c2w.squeeze(0).clone()
            pre_c2w = c2w.clone()
            self.idx[0] = idx
