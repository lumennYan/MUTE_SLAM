# This file is a part of ESLAM.
#
# ESLAM is a NeRF-based SLAM system. It utilizes Neural Radiance Fields (NeRF)
# to perform Simultaneous Localization and Mapping (SLAM) in real-time.
# This software is the implementation of the paper "ESLAM: Efficient Dense SLAM
# System Based on Hybrid Representation of Signed Distance Fields" by
# Mohammad Mahdi Johari, Camilla Carta, and Francois Fleuret.
#
# Copyright 2023 ams-OSRAM AG
#
# Author: Mohammad Mahdi Johari <mohammad.johari@idiap.ch>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is a modified version of https://github.com/cvg/nice-slam/blob/master/src/NICE_SLAM.py
# which is covered by the following copyright and permission notice:
    #
    # Copyright 2022 Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, Marc Pollefeys
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

import os
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp
#import tinycudann as tcnn


from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.utils.datasets import get_dataset
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer
from .encoding import get_encoder

torch.multiprocessing.set_sharing_strategy('file_system')


class ESLAM():
    """
    ESLAM main class.
    Mainly allocate shared resources, and dispatch mapping and tracking processes.
    Args:
        cfg (dict): config dict
        args (argparse.Namespace): arguments
    """

    def __init__(self, cfg, args):

        self.cfg = cfg
        self.args = args

        self.verbose = cfg['verbose']
        self.device = cfg['device']
        self.dataset = cfg['dataset']
        self.truncation = cfg['model']['truncation']

        if args.output is None:
            self.output = cfg['data']['output']
        else:
            self.output = args.output
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.update_cam()

        model = config.get_model(cfg)
        self.shared_decoders = model

        self.scale = cfg['scale']

        self.load_bound(cfg)
        self.init_planes(cfg)

        # need to use spawn
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.frame_reader = get_dataset(cfg, args, self.scale)
        self.n_img = len(self.frame_reader)
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4), device=self.device)
        self.estimate_c2w_list.share_memory_()

        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list.share_memory_()
        self.idx = torch.zeros((1)).int()
        self.idx.share_memory_()
        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_first_frame.share_memory_()

        # the id of the newest frame Mapper is processing
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        self.mapping_cnt = torch.zeros((1)).int()  # counter for mapping
        self.mapping_cnt.share_memory_()

        ## Moving feature planes and decoders to the processing device
        for shared_planes in [self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz]:
            shared_planes = shared_planes.to(self.device)
            shared_planes.share_memory()

        '''
        for shared_c_planes in [self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz]:
            shared_c_planes = shared_c_planes.to(self.device)
            shared_c_planes.share_memory()
        '''

        self.shared_decoders = self.shared_decoders.to(self.device)
        self.shared_decoders.share_memory()

        self.renderer = Renderer(cfg, self)
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(self)
        self.mapper = Mapper(cfg, args, self)
        self.tracker = Tracker(cfg, args, self)
        self.print_output_desc()

    def get_gpu_memory_usage(self):
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        return allocated, reserved

    def print_output_desc(self):
        print(f"INFO: The output folder is {self.output}")
        print(
            f"INFO: The GT, generated and residual depth/color images can be found under " +
            f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpt/")

    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def load_bound(self, cfg):
        """
        Pass the scene bound parameters to different decoders and self.

        Args:
            cfg (dict): parsed config dict.
        """

        # scale the bound if there is a global scaling factor
        self.bound = torch.from_numpy(np.array(cfg['mapping']['bound'])*self.scale).float()

        #bound_dividable = cfg['planes_res']['bound_dividable']
        bound_dividable = 0.02
        # enlarge the bound a bit to allow it dividable by bound_dividable
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) /
                            bound_dividable).int()+1)*bound_dividable+self.bound[:, 0]

        self.shared_decoders.bound = self.bound

    def init_planes(self, cfg):
        """
        Initialize the feature planes.

        Args:
            cfg (dict): parsed config dict.
        """
        '''
        self.coarse_planes_res = cfg['planes_res']['coarse']
        self.fine_planes_res = cfg['planes_res']['fine']

        self.coarse_c_planes_res = cfg['c_planes_res']['coarse']
        self.fine_c_planes_res = cfg['c_planes_res']['fine']

        c_dim = cfg['model']['c_dim']
        '''

        self.encoding_type = cfg['encoding']['type']
        self.encoding_levels = cfg['encoding']['n_levels']
        self.desired_resolution = cfg['encoding']['desired_resolution']
        self.base_resolution = cfg['encoding']['base_resolution']
        self.log2_hashmap_size = cfg['encoding']['log2_hashmap_size']
        self.per_level_feature_dim = cfg['encoding']['feature_dim']
        self.per_level_scale = np.exp2(np.log2(self.desired_resolution / self.base_resolution) / (self.encoding_levels-1))
        self.encoding_dict = {"n_levels": self.encoding_levels, "otype": self.encoding_type, "n_features_per_level": self.per_level_feature_dim,
                              "log2_hashmap_size": self.log2_hashmap_size, "base_resolution": self.base_resolution, "per_level_scale": self.per_level_scale}
        ####### Initializing Planes ############
        #planes_xy = tcnn.Encoding(n_input_dims=2, encoding_config=self.encoding_dict, dtype=torch.float32)
        #planes_xz = tcnn.Encoding(n_input_dims=2, encoding_config=self.encoding_dict, dtype=torch.float32)
        #planes_yz = tcnn.Encoding(n_input_dims=2, encoding_config=self.encoding_dict, dtype=torch.float32)


        desired_res = 512
        planes_xy, self.planes_dim = get_encoder('hashgrid', input_dim=2, desired_resolution=desired_res)
        planes_xz, _ = get_encoder('hashgrid', input_dim=2, desired_resolution=desired_res)
        planes_yz, _ = get_encoder('hashgrid', input_dim=2, desired_resolution=desired_res)


        self.shared_planes_xy = planes_xy
        self.shared_planes_xz = planes_xz
        self.shared_planes_yz = planes_yz

    def tracking(self, rank):
        """
        Tracking Thread.

        Args:
            rank (int): Thread ID.
        """

        # should wait until the mapping of first frame is finished
        while True:
            if self.mapping_first_frame[0] == 1:
                break
            time.sleep(1)

        self.tracker.run()

    def mapping(self, rank):
        """
        Mapping Thread.

        Args:
            rank (int): Thread ID.
        """

        self.mapper.run()

    def run(self):
        """
        Dispatch Threads.
        """

        processes = []
        for rank in range(0, 2):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank, ))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        allocated, reserved = self.get_gpu_memory_usage()
        print('memory_allocated', allocated)
        print('memory_reserved', reserved)
# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
