
import numpy as np
import open3d as o3d
import skimage
import torch
import trimesh
from packaging import version
from src.utils.datasets import get_dataset


class Mesher(object):
    """
    Mesher class.
    Args:
        cfg (dict): configuration dictionary.
        args (argparse.Namespace): arguments.
        slam (MUTE_SLAM): MUTE_SLAM object.
        points_batch_size (int): number of points to be processed in each batch.
        ray_batch_size (int): number of rays to be processed in each batch.

    """
    def __init__(self, cfg, args, slam, points_batch_size=500000, ray_batch_size=100000):
        self.points_batch_size = points_batch_size
        self.ray_batch_size = ray_batch_size
        self.renderer = slam.renderer
        self.scale = cfg['scale']

        self.resolution = cfg['meshing']['resolution']
        self.level_set = cfg['meshing']['level_set']
        self.mesh_bound_scale = cfg['meshing']['mesh_bound_scale']
        self.verbose = slam.verbose
        self.frame_reader = get_dataset(cfg, args, self.scale, device='cpu')
        self.n_img = len(self.frame_reader)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy


    def get_bound_from_submaps(self, submap_list):
        """
        Created marching cubes bound from all sub_maps

        Args:
            submap_list (List): The list of sub_map objects

        Returns:
            boundary (torch.Tensor) : The boundary that encompasses all the sub-maps (2, 3)

        """
        boundaries = submap_list[0].boundary
        for idx, submap in enumerate(submap_list):
            if idx == 0:
                pass
            else:
                boundaries = torch.cat([boundaries, submap.boundary], dim=0)
        boundary, _ = torch.min(boundaries, dim=0)
        boundary = torch.stack([boundary, torch.max(boundaries, dim=0)[0]], dim=0)

        return boundary

    def get_bound_from_frames(self, keyframe_dict, scale=1):
        """
        Get the scene bound (convex hull),
        using sparse estimated camera poses and corresponding depth images.

        Args:
            keyframe_dict (list): list of keyframe info dictionary.
            scale (float): scene scale.

        Returns:
            return_mesh (trimesh.Trimesh): the convex hull.
        """

        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            # for new version as provided in environment.yaml
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        else:
            # for lower version
            volume = o3d.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.integration.TSDFVolumeColorType.RGB8)
        cam_points = []
        for keyframe in keyframe_dict:
            c2w = keyframe['est_c2w'].cpu().numpy()
            # convert to open3d camera pose
            c2w[:3, 1] *= -1.0
            c2w[:3, 2] *= -1.0
            w2c = np.linalg.inv(c2w)
            cam_points.append(c2w[:3, 3])
            depth = keyframe['depth'].cpu().numpy()
            color = keyframe['color'].cpu().numpy()

            depth = o3d.geometry.Image(depth.astype(np.float32))
            color = o3d.geometry.Image(np.array(
                (color * 255).astype(np.uint8)))

            intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=1,
                depth_trunc=1000,
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic, w2c)

        cam_points = np.stack(cam_points, axis=0)
        mesh = volume.extract_triangle_mesh()
        mesh_points = np.array(mesh.vertices)
        points = np.concatenate([cam_points, mesh_points], axis=0)
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        mesh, _ = o3d_pc.compute_convex_hull()
        mesh.compute_vertex_normals()
        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            mesh = mesh.scale(self.mesh_bound_scale, mesh.get_center())
        else:
            mesh = mesh.scale(self.mesh_bound_scale, center=True)
        points = np.array(mesh.vertices)
        faces = np.array(mesh.triangles)
        return_mesh = trimesh.Trimesh(vertices=points, faces=faces)
        return return_mesh

    def eval_points(self, p, submap_list, decoders):
        """
        Evaluates the TSDF and/or color value for the points.

        Args:
            p (torch.Tensor): points to be evaluated, shape (N, 3).
            submap_list (List): The list of sub_map objects
            decoders (torch.nn.Module): decoders for TSDF and color.

        Returns:
            ret (torch.Tensor): the evaluation result, shape (N, 4).
        """

        p_split = torch.split(p, self.points_batch_size)
        bound = self.get_bound_from_submaps(submap_list)
        rets = []
        for pi in p_split:
            # mask for points out of bound
            mask_x = (pi[:, 0] < bound[1][0]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[0][1])
            mask_z = (pi[:, 2] < bound[1][2]) & (pi[:, 2] > bound[0][2])
            mask = mask_x & mask_y & mask_z

            ret = decoders.get_raw_for_mesher(pi, submap_list)

            ret[~mask, -1] = -1
            rets.append(ret)

        ret = torch.cat(rets, dim=0)
        return ret

    def get_grid_uniform(self, resolution, submap_list):
        """
        Get query point coordinates for marching cubes.

        Args:
            resolution (int): marching cubes resolution.

        Returns:
            (dict): points coordinates and sampled coordinates for each axis.
        """
        bound = self.get_bound_from_submaps(submap_list) * self.scale
        bound = bound.cpu()
        padding = 0.05

        nsteps_x = ((bound[1][0] - bound[0][0] + 2 * padding) / resolution).int().item()
        x = np.linspace(bound[0][0] - padding, bound[1][0] + padding, nsteps_x)
        
        nsteps_y = ((bound[1][1] - bound[0][1] + 2 * padding) / resolution).int().item()
        y = np.linspace(bound[0][1] - padding, bound[1][1] + padding, nsteps_y)
        
        nsteps_z = ((bound[1][2] - bound[0][2] + 2 * padding) / resolution).int().item()
        z = np.linspace(bound[0][2] - padding, bound[1][2] + padding, nsteps_z)

        x_t, y_t, z_t = torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(z).float()
        grid_x, grid_y, grid_z = torch.meshgrid(x_t, y_t, z_t, indexing='xy')
        print(grid_x.shape)
        grid_points_t = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=1)

        return {"grid_points": grid_points_t, "xyz": [x, y, z]}

    def get_mesh(self, mesh_out_file, submap_list, decoders, keyframe_dict, device='cuda:0', color=True):
        """
        Get mesh from keyframes and feature planes and save to file.

        Args:
            mesh_out_file (str): output mesh file.
            submap_list (List): The list of sub_map objects
            decoders (torch.nn.Module): decoders for TSDF and color.
            keyframe_dict (dict): keyframe dictionary.
            device (str): device to run the model.
            color (bool): whether to use color.

        Returns:
            None

        """

        with torch.no_grad():
            print('reach getmesh')
            grid = self.get_grid_uniform(self.resolution, submap_list)
            points = grid['grid_points']
            mesh_bound = self.get_bound_from_frames(keyframe_dict, self.scale)
            z = []
            mask = []
            for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                mask.append(mesh_bound.contains(pnts.cpu().numpy()))
            mask = np.concatenate(mask, axis=0)

            for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                z.append(self.eval_points(pnts.to(device), submap_list, decoders).cpu().numpy()[:, -1])
            z = np.concatenate(z, axis=0)

            z[~mask] = -1
            print('getting mesh')
            try:
                if version.parse(
                        skimage.__version__) > version.parse('0.15.0'):
                    # for new version as provided in environment.yaml
                    verts, faces, normals, values = skimage.measure.marching_cubes(
                        volume=z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
                    print('got mesh')
                else:
                    # for lower version
                    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
                        volume=z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
            except:
                print('marching_cubes error. Possibly no surface extracted from the level set.')
                return

            # convert back to world coordinates
            vertices = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

            if color:
                # color is extracted by passing the coordinates of mesh vertices through the network
                points = torch.from_numpy(vertices)
                z = []
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    z_color = self.eval_points(pnts.to(device).float(), submap_list, decoders).cpu()[..., :3]
                    z.append(z_color)
                z = torch.cat(z, dim=0)
                vertex_colors = z.numpy()
            else:
                vertex_colors = None

            vertices /= self.scale
            mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
            mesh.export(mesh_out_file)
            if self.verbose:
                print('Saved mesh at', mesh_out_file)