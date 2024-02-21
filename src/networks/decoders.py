import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate, normalize_3d_coordinate_to_unit

class Decoders(nn.Module):
    """
    Decoders for SDF and RGB.
    Args:
        c_dim: feature dimensions
        hidden_size: hidden size of MLP
        truncation: truncation of SDF
        n_blocks: number of MLP blocks
        learnable_beta: whether to learn beta

    """
    def __init__(self, device, in_dim=32, hidden_size=32, truncation=0.08, n_blocks=2, learnable_beta=True, use_tcnn=False):
        super().__init__()
        self.device = device
        self.in_dim = in_dim
        self.truncation = truncation
        self.n_blocks = n_blocks
        self.bound = torch.empty(3, 2)
        self.use_tcnn = use_tcnn

        ## layers for SDF decoder
        self.linears = nn.ModuleList(
            [nn.Linear(in_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])

        ## layers for RGB decoder
        self.c_linears = nn.ModuleList(
            [nn.Linear(in_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])

        self.output_linear = nn.Linear(hidden_size, 1)
        self.c_output_linear = nn.Linear(hidden_size, 3)

        if learnable_beta:
            self.beta = nn.Parameter(10 * torch.ones(1))
        else:
            self.beta = 10

    def sample_plane_feature(self, p_nor, planes_xy, planes_xz, planes_yz):
        """
        Sample feature from planes
        Args:
            p_nor (tensor): normalized 3D coordinates
            planes_xy (list): xy planes
            planes_xz (list): xz planes
            planes_yz (list): yz planes
        Returns:
            feat (tensor): sampled features
        """

        xy = planes_xy(p_nor[..., [0, 1]])
        xz = planes_xz(p_nor[..., [0, 2]])
        yz = planes_yz(p_nor[..., [1, 2]])
        feat = xy + xz + yz  # [N, 32]

        return feat

    def get_feature_from_points(self, pts, submap_list):
        """
        Get features from inputting points
        Args:
            pts (tensor, (N,3)): normalized 3D coordinates
            submap_list (List): The list of sub_map objects
        Returns:
            features (tensor)
        """
        index = torch.linspace(0, pts.shape[0]-1, pts.shape[0], dtype=torch.long, device=self.device)
        feat_list = []
        c_feat_list = []
        indices_list = []
        pre_mask = torch.zeros(pts.shape[0], dtype=torch.bool, device=self.device)
        for submap in submap_list:
            pts_mask = torch.logiacl_and((pts[..., :] > submap.boundary[0]).all(dim=-1),
                                         (pts[..., :] < submap.boundary[1]).all(dim=-1))
            pts_mask = torch.logiacl_and(pts_mask, torch.logiacl_xor(pre_mask, pts_mask))
            pre_mask = torch.logical_or(pre_mask, pts_mask)
            indices_list.append(index[pts_mask])
            if self.use_tcnn:
                p_nor = normalize_3d_coordinate_to_unit(pts[pts_mask], submap.boundary)
            else:
                p_nor = normalize_3d_coordinate(pts[pts_mask], submap.boundary)
            c_feat_list.append(self.sample_plane_feature(p_nor, submap.c_planes_xy, submap.c_planes_xz, submap.c_planes_yz))
            feat_list.append(self.sample_plane_feature(p_nor, submap.planes_xy, submap.planes_xz, submap.planes_yz))
        feat_all = torch.zeros((pts.shape[0], feat_list[0].shape[1]), device=self.device)
        c_feat_all = torch.zeros((pts.shape[0], c_feat_list[0].shape[1]), device=self.device)

        for feat, c_feat, indices in zip(feat_list, c_feat_list, indices_list):
            feat_all.index_put_((indices,), feat)
            c_feat_all.index_put_((indices,), c_feat)

        return feat_all, c_feat_all

    def get_feature_from_points_for_mesher(self, pts, submap_list):
        """
        Get features from inputting points
        Args:
            pts (tensor, (N,3)): normalized 3D coordinates
            submap_list (List): The list of sub_map objects
        Returns:
            features (tensor)
        """
        with torch.no_grad():
            index = torch.linspace(0, pts.shape[0]-1, pts.shape[0], dtype=torch.long, device=self.device)
            feat_list = []
            c_feat_list = []
            indices_list = []
            pre_mask = torch.zeros(pts.shape[0], dtype=torch.bool, device=self.device)
            for submap in submap_list:
                pts_mask = torch.logical_and((pts[..., :] > submap.boundary[0]).all(dim=-1),
                                            (pts[..., :] < submap.boundary[1]).all(dim=-1))
                pts_mask = torch.logical_and(pts_mask, torch.logical_xor(pre_mask, pts_mask))
                pre_mask = torch.logical_or(pre_mask, pts_mask)
                indices_list.append(index[pts_mask])
                if self.use_tcnn:
                    p_nor = normalize_3d_coordinate_to_unit(pts[pts_mask], submap.boundary)
                else:
                    p_nor = normalize_3d_coordinate(pts[pts_mask], submap.boundary)
                feat_list.append(self.sample_plane_feature(p_nor, submap.planes_xy, submap.planes_xz, submap.planes_yz))
                c_feat_list.append(self.sample_plane_feature(p_nor, submap.c_planes_xy, submap.c_planes_xz, submap.c_planes_yz))
            feat_all = torch.full((pts.shape[0], feat_list[0].shape[1]), float('nan'), device=self.device)
            c_feat_all = torch.zeros((pts.shape[0], c_feat_list[0].shape[1]), device=self.device)
            for feat, c_feat, indices in zip(feat_list, c_feat_list, indices_list):
                feat_all.index_put_((indices,), feat)
                c_feat_all.index_put_((indices,), c_feat)

            out_bound_indices = index[torch.isnan(feat_all).any(dim=-1)]
        return feat_all, c_feat_all, out_bound_indices

    def get_raw_sdf(self, feat):
        """
        Get raw SDF
        Args:
            feat (tensor, (N, feature dimension))
        Returns:
            sdf (tensor): raw SDF
        """
        h = feat
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h, inplace=True)
        sdf = torch.tanh(self.output_linear(h)).squeeze()

        return sdf

    def get_raw_rgb(self, feat):
        """
        Get raw RGB
        Args:
            feat (tensor, (N, feature dimension))
        Returns:
            rgb (tensor): raw RGB
        """
        h = feat
        for i, l in enumerate(self.c_linears):
            h = self.c_linears[i](h)
            h = F.relu(h, inplace=True)
        rgb = torch.sigmoid(self.c_output_linear(h))

        return rgb

    def get_raw_for_mesher(self, p, submap_list):
        """
        Forward pass
        Args:
            p (tensor): 3D coordinates
            submap_list (List): The list of sub_map objects
        Returns:
            raw (tensor): raw SDF and RGB
        """
        p_shape = p.shape
        p = p.reshape(-1, 3)
        with torch.no_grad():
            features, c_features, out_bound_indices = self.get_feature_from_points_for_mesher(p, submap_list)

            sdf = self.get_raw_sdf(features).detach()
            sdf.index_put_((out_bound_indices,), torch.tensor([-1.], device=self.device))
            rgb = self.get_raw_rgb(c_features).detach()

            raw = torch.cat([rgb, sdf.unsqueeze(-1)], dim=-1)
            raw = raw.reshape(*p_shape[:-1], -1)

        return raw

    def forward(self, p, submap_list):
        """
        Forward pass
        Args:
            p (tensor): 3D coordinates
            submap_list (List): The list of sub_map objects
        Returns:
            raw (tensor): raw SDF and RGB
        """
        p_shape = p.shape
        p = p.reshape(-1, 3)
        features, c_features = self.get_feature_from_points(p, submap_list)

        sdf = self.get_raw_sdf(features)
        rgb = self.get_raw_rgb(c_features)

        raw = torch.cat([rgb, sdf.unsqueeze(-1)], dim=-1)
        raw = raw.reshape(*p_shape[:-1], -1)

        return raw
