import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import os
import imageio
import numpy as np
import collections
import matplotlib.pyplot as plt
from skimage.draw import circle


class Logger:
    def __init__(self, log_dir, checkpoint_freq=1, visualizer_params=None, zfill_num=6, log_file_name='log.txt'):

        self.names = None
        self.models = None
        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        self.tensorboard_writer = SummaryWriter(logdir=log_dir)

        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)

        self.epoch = 0
        self.zfill_num = zfill_num
        self.best_loss = float('inf')
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join([f"{name} - {value:.4f}" for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, f"{str(self.epoch).zfill(self.zfill_num)}-rec.png"), image)

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, f'{str(self.epoch).zfill(self.zfill_num)}-checkpoint.pth.tar')
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

        self.log_file.flush()
        self.log_file.close()
        self.tensorboard_writer.close()

    def log_iter(self, losses, learning_rates, step):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

        self.log_tensorboard(losses, learning_rates)

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()

        self.log_scores(self.names)
        self.visualize_rec(inp, out)

    def log_tensorboard(self, losses, learning_rates, step):
        for key, value in losses.items():
            self.tensorboard_writer.add_scalar(key, value, step)

        for key, value in learning_rates.items():
            self.tensorboard_writer.add_scalar(key, value, step)

        self.tensorboard_writer.add_scalars('all_losses', losses, step)
        self.tensorboard_writer.add_scalars('all_lrs', learning_rates, step)


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2

        num_kp = kp_array.shape[0]
        for kp_idx, kp in enumerate(kp_array):
            rr, cc = circle(kp[0], kp[1], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_idx / num_kp))[:3]

        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    # TODO() : deformed image, sparse_deformed images (Needs 3D -> 2D)
    def visualize(self, source, driving, out):
        images = []

        # Source image with keypoints
        source = source.data.cpu()
        source = np.transpose(source, [0, 2, 3, 1])
        kp_source = out['kp_source']['keypoints'].data[..., 1:].cpu().numpy()
        images.append((source, kp_source))

        # Equivariance visualization
        if 'transformed_frame' in out:
            transformed = out['transformed_frame'].data.cpu().numpy()
            transformed = np.transpose(transformed, [0, 2, 3, 1])
            transformed_kp = out['transformed_kp']['keypoints'].data[..., 1:].cpu().numpy()
            images.append((transformed, transformed_kp))

        # Driving image with keypoints
        kp_driving = out['kp_driving']['keypoints'].data[..., 1:].cpu().numpy()
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))

        # Result with and without keypoints
        prediction = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        if 'kp_norm' in out:
            kp_norm = out['kp_norm']['keypoints'].data.cpu().numpy()
            images.append((prediction, kp_norm))
        images.append(prediction)

        # Occlusion map
        if 'occlusion_map' in out:
            occlusion_map = out['occlusion_map'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
