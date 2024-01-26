import torch
import torch.nn as nn
import torch.nn.functional as F


def get_encoder(encoding='hashgrid', input_dim=2,
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=18, desired_resolution=2048, align_corners=True,):

    if encoding == 'None':
        return lambda x, **kwargs: x, input_dim

    elif encoding == 'HashGrid':
        from .gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners)

    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]')

    return encoder, encoder.output_dim
