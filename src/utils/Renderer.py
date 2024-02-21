import torch
from src.common import get_rays, sample_pdf

class Renderer(object):
    """
    Renderer class for rendering depth and color.
    Args:
        cfg (dict): configuration.
        slam (MUTE_SLAM): MUTE_SLAM object.
        ray_batch_size (int): batch size for sampling rays.
    """
    def __init__(self, cfg, slam, ray_batch_size=10000):
        self.ray_batch_size = ray_batch_size

        self.perturb = cfg['rendering']['perturb']
        self.n_stratified = cfg['rendering']['n_stratified']
        self.n_importance = cfg['rendering']['n_importance']

        self.scale = cfg['scale']

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def perturbation(self, z_vals):
        """
        Add perturbation to sampled depth values on the rays.
        Args:
            z_vals (tensor): sampled depth values on the rays.
        Returns:
            z_vals (tensor): perturbed depth values on the rays.
        """
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)

        return lower + (upper - lower) * t_rand

    def render_batch_ray(self, submap_list, decoders, rays_d, rays_o, device, truncation, gt_depth=None):
        """
        Render depth and color for a batch of rays.
        Args:
            submap_list (List): The list of sub_map objects
            decoders (torch.nn.Module): decoders for TSDF and color.
            rays_d (tensor): ray directions.
            rays_o (tensor): ray origins.
            device (torch.device): device to run on.
            truncation (float): truncation threshold.
            gt_depth (tensor): ground truth depth.
        Returns:
            depth_map (tensor): depth map.
            color_map (tensor): color map.
            volume_densities (tensor): volume densities for sampled points.
            z_vals (tensor): sampled depth values on the rays.

        """
        n_stratified = self.n_stratified
        n_importance = self.n_importance
        near = 0.
        t_vals_uni = torch.linspace(0., 1., steps=n_stratified, device=device)
        t_vals_surface = torch.linspace(0., 1., steps=n_importance, device=device)

        ### pixels with gt depth:
        gt_depth = gt_depth.reshape(-1, 1)
        gt_mask = (gt_depth > 0).squeeze()
        gt_nonezero = gt_depth[gt_mask]
        rays_o = rays_o[gt_mask]
        rays_d = rays_d[gt_mask]

        ## Sampling points around the gt depth (surface)
        gt_depth_surface = gt_nonezero.expand(-1, n_importance)
        ## in the range of gt_depth +-1.5 truncation, a uniform sampling
        z_vals_surface = gt_depth_surface - (1.5 * truncation) + (3 * truncation * t_vals_surface)

        gt_depth_free = gt_nonezero.expand(-1, n_stratified)
        ## in the range of 1.2*gt_depth, a uniform sampling
        z_vals_free = near + 1.2 * gt_depth_free * t_vals_uni

        z_vals_end = z_vals_free[..., -1]
        pts_end = rays_o + rays_d * z_vals_end[:, None]
        inmap_mask = torch.ones(pts_end.shape[0], dtype=torch.bool, device=device)

        ## filter out rays that end outside all current sub_maps
        for submap in submap_list:
            cur_mask = torch.bitwise_and((pts_end > submap.boundary[0]).all(-1),
                                           (pts_end < submap.boundary[1]).all(dim=-1))
            inmap_mask = torch.bitwise_or(inmap_mask, cur_mask)

        z_vals, _ = torch.sort(torch.cat([z_vals_free, z_vals_surface], dim=-1), dim=-1)

        if self.perturb:
            z_vals = self.perturbation(z_vals)

        z_vals = z_vals[inmap_mask]
        rays_o = rays_o[inmap_mask]
        rays_d = rays_d[inmap_mask]

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # [n_rays, n_stratified+n_importance, 3]

        raw = decoders(pts, submap_list)  # [n_rays, n_stratified+n_importance, 4]
        alpha = self.sdf2alpha(raw[..., -1], decoders.beta)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device)
                                                , (1. - alpha + 1e-10)], -1), -1)[:, :-1]  # [n_rays, n_stratified+n_importance]

        rendered_rgb = torch.sum(weights[..., None] * raw[..., :3], -2)
        rendered_depth = torch.sum(weights * z_vals, -1)

        return rendered_depth, rendered_rgb, raw[..., -1], z_vals, inmap_mask

    def sdf2alpha(self, sdf, beta=10):
        """
        Convert sdf values to volume densitise
        """
        return 1. - torch.exp(-beta * torch.sigmoid(-sdf * beta))

    def render_img(self, submap_list, decoders, c2w, truncation, device, gt_depth=None):
        """
        Renders out depth and color images.
        Args:
            submap_list (List): The list of sub_map objects
            decoders (torch.nn.Module): decoders for TSDF and color.
            c2w (tensor, 4*4): camera pose.
            truncation (float): truncation distance.
            device (torch.device): device to run on.
            gt_depth (tensor, H*W): ground truth depth image.
        Returns:
            rendered_depth (tensor, H*W): rendered depth image.
            rendered_rgb (tensor, H*W*3): rendered color image.

        """
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(H, W, self.fx, self.fy, self.cx, self.cy,  c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            color_list = []

            ray_batch_size = self.ray_batch_size
            gt_depth = gt_depth.reshape(-1)

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i+ray_batch_size]
                rays_o_batch = rays_o[i:i+ray_batch_size]
                if gt_depth is None:
                    ret = self.render_batch_ray(submap_list, decoders, rays_d_batch, rays_o_batch,
                                                device, truncation, gt_depth=None)
                else:
                    gt_depth_batch = gt_depth[i:i+ray_batch_size]
                    ret = self.render_batch_ray(submap_list, decoders, rays_d_batch, rays_o_batch,
                                                device, truncation, gt_depth=gt_depth_batch)

                depth, color, _, _, _ = ret
                depth_list.append(depth.double())
                color_list.append(color)

            depth = torch.cat(depth_list, dim=0)
            color = torch.cat(color_list, dim=0)

            depth = depth.reshape(H, W)
            color = color.reshape(H, W, 3)

            return depth, color
