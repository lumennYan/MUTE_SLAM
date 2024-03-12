import os
import torch

class Logger(object):
    """
    Save checkpoints to file.

    """

    def __init__(self, eslam):
        self.verbose = eslam.verbose
        self.ckptsdir = eslam.ckptsdir
        self.gt_c2w_list = eslam.gt_c2w_list
        self.shared_decoders = eslam.shared_decoders
        self.estimate_c2w_list = eslam.estimate_c2w_list

    def log(self, idx, keyframe_list):
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(idx))
        torch.save({
            'decoder_state_dict': self.shared_decoders.state_dict(),
            'gt_c2w_list': self.gt_c2w_list,
            'estimate_c2w_list': self.estimate_c2w_list,
            'keyframe_list': keyframe_list,
            'idx': idx,
        }, path, _use_new_zipfile_serialization=False)


        if self.verbose:
            print('Saved checkpoints at', path)
