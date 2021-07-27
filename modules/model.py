import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torchvision import models
from torch.autograd import grad
from itertools import combinations
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid, get_grid, get_rotation_matrix


class Vgg19(nn.Module):
    """
    Vgg19 network for perceptual loss.
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()

        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramid(nn.Module):
    """
    Create multi-scale images for multi-scale perceptual loss.
    """

    def __init__(self, scales, num_channels):
        super(ImagePyramid, self).__init__()

        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints.
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0, std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


class GeneratorFullModel(nn.Module):
    """
    Merge all generator related upadtes into single model for better multi-gpu usage
    """

    def __init__(self, appearance_feature_extractor, canonical_keypoint_detector, head_expression_estimator,
                 pretrained_hopenet, generator, discriminator, train_params):
        super(GeneratorFullModel, self).__init__()

        self.appearance_feature_extractor = appearance_feature_extractor
        self.canonical_keypoint_detector = canonical_keypoint_detector
        self.head_expression_estimator = head_expression_estimator
        self.pretrained_hopenet = pretrained_hopenet
        self.generator = generator
        self.discriminator = discriminator

        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramid(self.scales, generator.output_channels)

        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']
        self.l1_loss = nn.L1Loss()

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

    def get_keypoint(self, img_source, img_driving, canonical_keypoint):
        kp_source, kp_driving = dict(), dict()
        (yaw_s, pitch_s, roll_s), translation_s, deformation_s = self.head_expression_estimator(img_source)
        (yaw_d, pitch_d, roll_d), translation_d, deformation_d = self.head_expression_estimator(img_driving)
        kp_source['deformation'] = deformation_s
        kp_driving['deformation'] = deformation_d

        idx_tensor = torch.arange(self.train_params['num_bins'], dtype=torch.float32).cuda()
        rotation_s, (yaw_s, pitch_s, roll_s) = get_rotation_matrix(yaw_s, pitch_s, roll_s, idx_tensor)
        rotation_d, (yaw_d, pitch_d, roll_d) = get_rotation_matrix(yaw_d, pitch_d, roll_d, idx_tensor)
        kp_source['yaw'], kp_source['pitch'], kp_source['roll'] = yaw_s, pitch_s, roll_s
        kp_driving['yaw'], kp_driving['pitch'], kp_driving['roll'] = yaw_d, pitch_d, roll_d

        kp_source['keypoints'] = canonical_keypoint['keypoints'] * rotation_s + translation_s.unsqueeze(-2) + deformation_s.view(-1, self.train_params['num_kp'], 3)
        kp_source['jacobian'] = canonical_keypoint['jacobian'] * rotation_s
        kp_driving['keypoints'] = canonical_keypoint['keypoints'] * rotation_d + translation_d.unsqueeze(-2) + deformation_d.view(-1, self.train_params['num_kp'], 3)
        kp_driving['jacobian'] = canonical_keypoint['jacobian'] * rotation_d

        return kp_source, kp_driving

    def calculate_loss_values(self, pyramid_real, pyramid_generated, generated, kp_source, kp_driving):

        loss_values = dict()

        """ Perceptual loss """
        perceptual = 0.
        for scale in self.scales:
            vgg_real = self.vgg(pyramid_real[f'prediction_{scale}'])
            vgg_generated = self.vgg(pyramid_generated[f'prediction_{scale}'])

            for i, weight in enumerate(self.loss_weights['perceptual']):
                value = torch.abs(vgg_real[i] - vgg_generated[i].detach()).mean()
                perceptual += weight * value
        loss_values['perceptual'] = perceptual

        """ GAN loss """
        generator_gan = 0.
        out_dict_real = self.discriminator(pyramid_real)
        out_dict_generated = self.discriminator(pyramid_generated)
        for scale in self.scales:
            generator_gan -= out_dict_generated[f'prediction_map_{scale}'].mean() * self.loss_weights['generator_gan']
        loss_values['generator_gan'] = generator_gan

        feature_matching = 0.
        for scale in self.disc_scales:
            key = f'feature_maps_%{scale}'
            for i, (y, x) in enumerate(zip(out_dict_real[key], out_dict_generated[key])):
                if self.loss_weights['feature_matching'][i] == 0:
                    continue
                value = torch.abs(y - x).mean()
                feature_matching += self.loss_weights['feature_matching'][i] * value
        loss_values['feature_matching'] = feature_matching

        """ Equivariance loss """
        transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
        transformed_frame = transform.transform_frame(x['driving'])
        transformed_kp = self.ckp_extractor(transformed_frame)

        generated['transformed_frame'] = transformed_frame
        generated['transformed_kp'] = transformed_kp

        if self.loss_weights['equivariance_keypoints'] != 0:
            equivariance_keypoints = torch.abs(kp_driving['keypoints'][:, :, :2] - transform.warp_coordinates(
                transformed_kp['keypoints'][:, :, :2])).mean()
            loss_values['equivariance_keypoints'] = self.loss_weights['equivariance_keypoints'] * equivariance_keypoints

        if self.loss_weights['equivariance_jacobian'] != 0:
            jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['keypoints'][:, :, :2]),
                                                transformed_kp['jacobian'][:, :, :2, :2])

            normed_driving = torch.inverse(kp_driving['jacobian'][:, :, :2, :2])
            normed_transformed = jacobian_transformed
            equivariance_jacobian = torch.matmul(normed_driving, normed_transformed)

            eye = torch.eye(2).view(1, 1, 2, 2).type(equivariance_jacobian.type())
            equivariance_jacobian = torch.abs(eye - equivariance_jacobian).mean()
            loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * equivariance_jacobian

        """ Keypoint prior loss """
        keypoint_prior = 0.
        for keypoint in kp_source['keypoints']:
            points = combinations(keypoint, r=2)
            for (p1, p2) in points:
                keypoint_prior += F.relu(self.train_params['keypoint_distance_threshold'] - self.torch.dist(p1, p2))

        for keypoint in kp_driving['keypoints']:
            points = combinations(keypoint, r=2)
            for (p1, p2) in points:
                keypoint_prior += F.relu(self.train_params['keypoint_distance_threshold'] - self.torch.dist(p1, p2))

        keypoint_prior += self.l1_loss(torch.mean(self.train_params[:, :, 0]),
                                       self.train_params['keypoint_depth_target'])
        loss_values['keypoint_prior'] = self.loss_weights['keypoint_prior'] * keypoint_prior

        """ Head pose loss """
        head_pose = 0.
        yaw_target_source, pitch_target_source, roll_target_source = self.pretrained_hopenet(x['source'])
        yaw_target_driving, pitch_target_driving, roll_target_driving = self.pretrained_hopenet(x['source'])

        idx_tensor = torch.arange(self.train_params['num_bins'], dtype=torch.float32).cuda()
        _, (yaw_target_source, pitch_target_source, roll_target_source) = get_rotation_matrix(yaw_target_source,
                                                                                              pitch_target_source,
                                                                                              roll_target_source,
                                                                                              idx_tensor)
        _, (yaw_target_driving, pitch_target_driving, roll_target_driving) = get_rotation_matrix(yaw_target_driving,
                                                                                                 pitch_target_driving,
                                                                                                 roll_target_driving,
                                                                                                 idx_tensor)

        head_pose += self.l1_loss(torch.cat([kp_source['yaw'], kp_source['pitch'], kp_source['roll']], dim=-1),
                                  torch.cat([yaw_target_source, pitch_target_source, roll_target_source], dim=-1))
        head_pose += self.l1_loss(torch.cat([kp_driving['yaw'], kp_driving['pitch'], kp_driving['roll']], dim=-1),
                                  torch.cat([yaw_target_driving, pitch_target_driving, roll_target_driving], dim=-1))
        loss_values['head_pose'] = self.loss_weights['head_pose'] * head_pose

        """ Deformation prior loss """
        deformation_prior = 0.
        deformation_prior += self.l1_loss(kp_source['deformation'], torch.zeros_like(kp_source['deformation']))
        deformation_prior += self.l1_loss(kp_driving['deformation'], torch.zeros_like(kp_driving['deformation']))
        loss_values['deformation_prior'] = self.loss_weights['deformation_prior'] * deformation_prior
        return loss_values

    def forward(self, x):

        appearance_feature = self.appearance_feature_extractor(x['source'])
        canonical_keypoint = self.canonical_keypoint_detector(x['source'])
        kp_source, kp_driving = self.get_keypoint(x['source'], x['driving'], canonical_keypoint)

        generated = self.generator(source_feature=appearance_feature, kp_source=kp_source, kp_driving=kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        pyramid_real = self.pyramid(x['driving'])
        pyramid_generated = self.pyramid(generated['prediction'])

        loss_values = self.calculate_loss_values(pyramid_real, pyramid_generated, generated, kp_source, kp_driving)
        return loss_values


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramid(self.scales, generator.output_channels)

        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramid_real = self.pyramid(x['driving'])
        pyramid_generated = self.pyramid(generated['prediction'].detach())

        out_dict_real = self.discriminator(pyramid_real)
        out_dict_generated = self.discriminator(pyramid_generated)

        loss_values = {}
        discriminator_gan = 0.
        for scale in self.scales:
            patch_grid_real = get_grid(out_dict_real[f'prediction_map_{scale}'], is_real=True).cuda()
            discriminator_gan += F.relu(patch_grid_real - out_dict_real[f'prediction_map_{scale}']).mean() * self.loss_weights['discriminator_gan']
            discriminator_gan += F.relu(patch_grid_real + out_dict_generated[f'prediction_map_{scale}']).mean() * self.loss_weights['discriminator_gan']

        loss_values['discriminator_gan'] = discriminator_gan
        return loss_values, generated
