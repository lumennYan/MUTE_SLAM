import os
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp
from .encoding import Maps

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.utils.datasets import get_dataset
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer


mp.set_sharing_strategy('file_system')


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

        self.use_tcnn = cfg['encoding']['tcnn']

        #self.load_bound(cfg)
        #self.submap_size = cfg['mapping']['submap_size']

        self.submap_dict_list = mp.Manager().list()
        self.submap_bound_list = mp.Manager().list()

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

    def init_planes(self, cfg):
        """
        Initialize the feature planes.

        Args:
            cfg (dict): parsed config dict.
        """

        ####### Initializing Planes ############

        self.encoding_type = cfg['encoding']['type']
        self.encoding_levels = cfg['encoding']['n_levels']
        self.base_resolution = cfg['encoding']['base_resolution']
        self.per_level_feature_dim = cfg['encoding']['feature_dim']

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
