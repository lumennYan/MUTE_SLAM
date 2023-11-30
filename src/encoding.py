import torch
import numpy as np


class Encoder:
    def __init__(self, device, boundary, use_tcnn=False, encoding_type='hashgrid',
                 input_dim=2, num_levels=16, level_dim=2, base_resolution=16, align_corners=True):
        self.boundary = boundary.to(device)
        self.edge_length = self.boundary[1] - self.boundary[0]
        self.device = device
        self.desired_resolution_xy = (torch.sqrt(self.edge_length[0] * self.edge_length[1]) * 100).int().item()
        self.desired_resolution_xz = (torch.sqrt(self.edge_length[0] * self.edge_length[2]) * 100).int().item()
        self.desired_resolution_yz = (torch.sqrt(self.edge_length[1] * self.edge_length[2]) * 100).int().item()
        self.log2_hashmap_size_xy = int(np.log2(self.desired_resolution_xy ** 2))
        self.log2_hashmap_size_xz = int(np.log2(self.desired_resolution_xz ** 2))
        self.log2_hashmap_size_yz = int(np.log2(self.desired_resolution_yz ** 2))

        if use_tcnn:
            import tinycudann as tcnn

            per_level_scale_xy = np.exp2(
                np.log2(self.desired_resolution_xy / base_resolution) / (num_levels - 1))
            per_level_scale_xz = np.exp2(
                np.log2(self.desired_resolution_xz / base_resolution) / (num_levels - 1))
            per_level_scale_yz = np.exp2(
                np.log2(self.desired_resolution_yz / base_resolution) / (num_levels - 1))
            self.encoding_dict_xy = dict(n_levels=num_levels, otype=encoding_type,
                                         n_features_per_level=level_dim,
                                         log2_hashmap_size=self.log2_hashmap_size_xy, base_resolution=base_resolution,
                                         per_level_scale=per_level_scale_xy)
            self.encoding_dict_xz = dict(n_levels=num_levels, otype=encoding_type,
                                         n_features_per_level=level_dim,
                                         log2_hashmap_size=self.log2_hashmap_size_xz, base_resolution=base_resolution,
                                         per_level_scale=per_level_scale_xz)
            self.encoding_dict_yz = dict(n_levels=num_levels, otype=encoding_type,
                                         n_features_per_level=level_dim,
                                         log2_hashmap_size=self.log2_hashmap_size_yz, base_resolution=base_resolution,
                                         per_level_scale=per_level_scale_yz)
            self.planes_xy = tcnn.Encoding(input_dim, encoding_config=self.encoding_dict_xy, dtype=torch.float32)
            self.planes_xz = tcnn.Encoding(input_dim, encoding_config=self.encoding_dict_xz, dtype=torch.float32)
            self.planes_yz = tcnn.Encoding(input_dim, encoding_config=self.encoding_dict_yz, dtype=torch.float32)

        else:
            self.planes_xy, _ = get_encoder(encoding_type, input_dim,
                                            num_levels, level_dim, base_resolution, self.log2_hashmap_size_xy,
                                            self.desired_resolution_xy, align_corners)
            self.planes_xz, _ = get_encoder(encoding_type, input_dim,
                                            num_levels, level_dim, base_resolution, self.log2_hashmap_size_xz,
                                            self.desired_resolution_xz, align_corners)
            self.planes_yz, _ = get_encoder(encoding_type, input_dim,
                                            num_levels, level_dim, base_resolution, self.log2_hashmap_size_yz,
                                            self.desired_resolution_yz, align_corners)

        for planes in [self.planes_xy, self.planes_xz, self.planes_yz]:
            planes = planes.to(self.device)
            planes.share_memory()

    def to_device(self, device):
        self.boundary = self.boundary.to(device)
        for planes in [self.planes_xy, self.planes_xz, self.planes_yz]:
            planes = planes.to(device)
            planes.share_memory()


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
