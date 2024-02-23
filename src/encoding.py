import torch
import torch.nn as nn
import numpy as np

class SubMap(nn.Module):
    def __init__(self, device, boundary, use_tcnn=False, encoding_type='hashgrid',
                 input_dim=2, num_levels=16, level_dim=2, base_resolution=16, align_corners=True):
        super().__init__()
        self.device = device
        self.boundary = boundary.to(self.device)

        with torch.no_grad():
            edge_length = self.boundary[1] - self.boundary[0]
            desired_resolution = (torch.pow(edge_length[0] * edge_length[1] * edge_length[2], 1/3) * 50).int().item()
            log2_hashmap_size = int(np.log2(desired_resolution ** 2))
            per_level_scale = np.exp2(
                np.log2(desired_resolution / base_resolution) / (num_levels - 1))
        if use_tcnn:
            import tinycudann as tcnn
            encoding_dict = dict(n_levels=num_levels, otype=encoding_type,
                                         n_features_per_level=level_dim,
                                         log2_hashmap_size=log2_hashmap_size, base_resolution=base_resolution,
                                         per_level_scale=per_level_scale)
            self.planes_xy = tcnn.Encoding(input_dim, encoding_config=encoding_dict, dtype=torch.float32)
            self.planes_xz = tcnn.Encoding(input_dim, encoding_config=encoding_dict, dtype=torch.float32)
            self.planes_yz = tcnn.Encoding(input_dim, encoding_config=encoding_dict, dtype=torch.float32)
            self.c_planes_xy = tcnn.Encoding(input_dim, encoding_config=encoding_dict, dtype=torch.float32)
            self.c_planes_xz = tcnn.Encoding(input_dim, encoding_config=encoding_dict, dtype=torch.float32)
            self.c_planes_yz = tcnn.Encoding(input_dim, encoding_config=encoding_dict, dtype=torch.float32)

        else:
            self.planes_xy, _ = get_encoder(encoding_type, input_dim,
                                            num_levels, level_dim, base_resolution, log2_hashmap_size,
                                            desired_resolution, align_corners)

            self.planes_xz, _ = get_encoder(encoding_type, input_dim,
                                            num_levels, level_dim, base_resolution, log2_hashmap_size,
                                            desired_resolution, align_corners)
            self.planes_yz, _ = get_encoder(encoding_type, input_dim,
                                            num_levels, level_dim, base_resolution, log2_hashmap_size,
                                            desired_resolution, align_corners)

            self.c_planes_xy, _ = get_encoder(encoding_type, input_dim,
                                            num_levels, level_dim, base_resolution, log2_hashmap_size,
                                            desired_resolution, align_corners)
            self.c_planes_xz, _ = get_encoder(encoding_type, input_dim,
                                            num_levels, level_dim, base_resolution, log2_hashmap_size,
                                            desired_resolution, align_corners)
            self.c_planes_yz, _ = get_encoder(encoding_type, input_dim,
                                            num_levels, level_dim, base_resolution, log2_hashmap_size,
                                            desired_resolution, align_corners)

        all_planes = nn.ModuleList([self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz])
        for planes in all_planes:
            planes = planes.to(self.device)


def get_encoder(encoding='hashgrid', input_dim=2,
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=18, desired_resolution=2048,
                align_corners=True):
    if encoding == 'None':
        return lambda x, **kwargs: x, input_dim

    elif encoding == 'HashGrid':
        from .gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim,
                              base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
                              desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners)

    else:
        raise NotImplementedError('Unknown encoding mode')

    return encoder, encoder.output_dim
