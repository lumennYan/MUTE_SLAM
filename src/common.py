import numpy as np
import torch
import torch.nn.functional as F


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    # pyre-ignore [16]: `torch.Tensor` has no attribute `new_tensor`.
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(*batch_dim, 4)


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]

    return K


def sample_pdf(bins, weights, N_samples, det=False, device='cuda:0'):
    """
    Hierarchical sampling in NeRF paper.
    """
    # Get pdf
    # weights = weights + 1e-5  # prevent nans
    #pdf = weights / torch.sum(weights, -1, keepdim=True)
    pdf = weights

    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def random_select(l, k):
    """
    Random select k values from 0..l.

    """
    return list(np.random.permutation(np.array(range(l)))[:min(l, k)])


def get_rays_from_uv(i, j, c2ws, H, W, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.

    """
    dirs = torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i, device=device)], -1)
    dirs = dirs.unsqueeze(-2)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2ws[:, None, :3, :3], -1)
    rays_o = c2ws[:, None, :3, -1].expand(rays_d.shape)

    return rays_o, rays_d


def select_uv(i, j, n, b, depths, colors, device='cuda:0'):
    """
    Select n uv from dense uv.

    """
    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n * b,), device=device)
    indices = indices.clamp(0, i.shape[0])
    i = i[indices]  # (n * b)
    j = j[indices]  # (n * b)

    indices = indices.reshape(b, -1)
    i = i.reshape(b, -1)
    j = j.reshape(b, -1)

    depths = depths.reshape(b, -1)
    colors = colors.reshape(b, -1, 3)

    depths = torch.gather(depths, 1, indices)  # (b, n)
    colors = torch.gather(colors, 1, indices.unsqueeze(-1).expand(-1, -1, 3))  # (b, n, 3)

    return i, j, depths, colors


def get_sample_uv(H0, H1, W0, W1, n, b, depths, colors, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    depths = depths[:, H0:H1, W0:W1]
    colors = colors[:, H0:H1, W0:W1]

    i, j = torch.meshgrid(torch.linspace(W0, W1 - 1, W1 - W0, device=device), torch.linspace(H0, H1 - 1, H1 - H0, device=device), indexing='ij')

    i = i.t()  # transpose
    j = j.t()
    i, j, depth, color = select_uv(i, j, n, b, depths, colors, device=device)

    return i, j, depth, color


def get_samples(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, c2ws, depths, colors, device):
    """
    Get n rays from the image region H0..H1, W0..W1.
    c2w is its camera pose and depth/color is the corresponding image tensor.

    """
    b = c2ws.shape[0]
    i, j, sample_depth, sample_color = get_sample_uv(
        H0, H1, W0, W1, n, b, depths, colors, device=device)

    rays_o, rays_d = get_rays_from_uv(i, j, c2ws, H, W, fx, fy, cx, cy, device)

    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), sample_depth.reshape(-1), sample_color.reshape(-1, 3)


def get_rays_cam_cord(H, W, fx, fy, cx, cy):
    """
    Get rays for a whole image.

    """
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='xy')
    dirs = torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1)
    rays_d = dirs.reshape(H, W, 3)
    return rays_d


def matrix_to_cam_pose(batch_matrices, RT=True):
    """
    Convert transformation matrix to quaternion and translation.
    Args:
        batch_matrices: (B, 4, 4)
        RT: if True, return (B, 7) with [R, T], else return (B, 7) with [T, R]
    Returns:
        (B, 7) with [R, T] or [T, R]
    """
    if RT:
        return torch.cat([matrix_to_quaternion(batch_matrices[:,:3,:3]), batch_matrices[:,:3,3]], dim=-1)
    else:
        return torch.cat([batch_matrices[:, :3, 3], matrix_to_quaternion(batch_matrices[:, :3, :3])], dim=-1)


def cam_pose_to_matrix(batch_poses):
    """
    Convert quaternion and translation to transformation matrix.
    Args:
        batch_poses: (B, 7) with [R, T] or [T, R]
    Returns:
        (B, 4, 4) transformation matrix
    """
    c2w = torch.eye(4, device=batch_poses.device).unsqueeze(0).repeat(batch_poses.shape[0], 1, 1)
    c2w[:,:3,:3] = quaternion_to_matrix(batch_poses[:,:4])
    c2w[:,:3,3] = batch_poses[:,4:]

    return c2w


def get_rays(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij')
    i = i.t()  # transpose
    j = j.t()
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H, W, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_points(H, W, fx, fy, cx, cy, c2w, gt_depth, device):
    """
    Get points in world coordinate for a whole image.
    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    depth_mask = (gt_depth > 0)
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij')
    i = i.t()  # transpose
    j = j.t()
    i = i[depth_mask].reshape(-1)
    j = j[depth_mask].reshape(-1)
    gt_depth = gt_depth[depth_mask].reshape(-1)
    dirs = torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs[..., None]
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.matmul(c2w[:3, :3], dirs).squeeze()
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    pts = rays_o[..., :] + rays_d[..., :] * gt_depth[..., None]
    pts.reshape(-1, 3)
    return pts


def get_sample_points(H, W, fx, fy, cx, cy, c2w, n, gt_depth, device):
    """
    Get sampled points in world coordinate for an image.
    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    depth_mask = (gt_depth > 0)
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij')
    i = i.t().to(device) # transpose
    j = j.t().to(device)
    i = i[depth_mask].reshape(-1)
    j = j[depth_mask].reshape(-1)
    gt_depth = gt_depth[depth_mask].reshape(-1)
    indices = torch.randint(i.shape[0], (n,), device=device)
    i = i[indices]
    j = j[indices]
    gt_depth = gt_depth[indices]
    dirs = torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs[..., None]
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.matmul(c2w[:3, :3], dirs).squeeze()
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    pts = rays_o[..., :] + rays_d[..., :] * gt_depth[..., None]
    pts.reshape(-1, 3)
    return pts


def normalize_3d_coordinate(p, bound):
    """
    Normalize 3d coordinate to [-1, 1] range.
    Args:
        p: (N, 3) 3d coordinate
        bound: (3, 2) min and max of each dimension
    Returns:
        (N, 3) normalized 3d coordinate

    """
    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0]-bound[0, 0])/(bound[1, 0]-bound[0, 0]))*2-1.0
    p[:, 1] = ((p[:, 1]-bound[0, 1])/(bound[1, 1]-bound[0, 1]))*2-1.0
    p[:, 2] = ((p[:, 2]-bound[0, 2])/(bound[1, 2]-bound[0, 2]))*2-1.0
    return p


def normalize_3d_coordinate_to_unit(p, bound):
    """
    Normalize 3d coordinate to [0, 1] range.
    Args:
        p: (N, 3) 3d coordinate
        bound: (3, 2) min and max of each dimension
    Returns:
        (N, 3) normalized 3d coordinate

    """
    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0]-bound[0, 0])/(bound[1, 0]-bound[0, 0]))
    p[:, 1] = ((p[:, 1]-bound[0, 1])/(bound[1, 1]-bound[0, 1]))
    p[:, 2] = ((p[:, 2]-bound[0, 2])/(bound[1, 2]-bound[0, 2]))
    return p
