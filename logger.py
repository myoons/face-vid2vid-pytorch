import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
from skimage.draw import circle

import matplotlib.pyplot as plt
import collections


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, zfill_num=8, log_file_name='log.txt'):

        self.models = None
        self.names = None
        self.loss_list = []
        self.cpk_dir = log_dir

        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)

        self.epoch = 0
        self.zfill_num = zfill_num
        self.best_loss = float('inf')
        self.checkpoint_freq = checkpoint_freq
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num))
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path,
                 appearance_feature_extractor=None,
                 canonical_keypoint_detector=None,
                 head_expression_estimator=None,
                 occlusion_aware_generator=None,
                 multi_scale_discriminator=None,
                 optimizer_appearance_feature_extractor=None,
                 optimizer_canonical_keypoint_detector=None,
                 optimizer_head_expression_estimator=None,
                 optimizer_occlusion_aware_generator=None,
                 optimizer_multi_scale_discriminator=None):

        checkpoint = torch.load(checkpoint_path)

        if appearance_feature_extractor is not None:
            appearance_feature_extractor.load_state_dict(checkpoint['appearance_feature_extractor'])

        if canonical_keypoint_detector is not None:
            canonical_keypoint_detector.load_state_dict(checkpoint['canonical_keypoint_detector'])

        if head_expression_estimator is not None:
            head_expression_estimator.load_state_dict(checkpoint['head_expression_estimator'])

        if occlusion_aware_generator is not None:
            occlusion_aware_generator.load_state_dict(checkpoint['occlusion_aware_generator'])

        if multi_scale_discriminator is not None:
            try:
                multi_scale_discriminator.load_state_dict(checkpoint['discriminator'])
            except Exception as e:
                print('No discriminator in the state-dict. Discriminator will be randomly initialized')

        if optimizer_appearance_feature_extractor is not None:
            optimizer_appearance_feature_extractor.load_state_dict(checkpoint['optimizer_appearance_feature_extractor'])

        if optimizer_canonical_keypoint_detector is not None:
            optimizer_canonical_keypoint_detector.load_state_dict(checkpoint['optimizer_canonical_keypoint_detector'])

        if optimizer_head_expression_estimator is not None:
            optimizer_head_expression_estimator.load_state_dict(checkpoint['optimizer_head_expression_estimator'])

        if optimizer_occlusion_aware_generator is not None:
            optimizer_occlusion_aware_generator.load_state_dict(checkpoint['optimizer_occlusion_aware_generator'])

        if optimizer_multi_scale_discriminator is not None:
            try:
                optimizer_multi_scale_discriminator.load_state_dict(checkpoint['optimizer_multi_scale_discriminator'])
            except RuntimeError as e:
                print('No discriminator optimizer in the state-dict. Optimizer will be not initialized')

        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
