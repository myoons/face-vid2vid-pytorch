import torch
import torch.nn as nn
import torch.nn.functional as F

from math import pi
from modules.pretrained_models import Vgg19, Hopenet, FaceVgg
from modules.util import Transform, ImagePyramid


class GeneratorFullModel(nn.Module):
    """
    Merge all generator related upadtes into single model for better multi-gpu usage
    """

    def __init__(self, appearance_feature_extractor, canonical_keypoint_detector, head_expression_estimator,
                 occlusion_aware_generator, multi_scale_discriminator, train_params, train=False):
        super(GeneratorFullModel, self).__init__()

        self.appearance_feature_extractor = appearance_feature_extractor
        self.canonical_keypoint_detector = canonical_keypoint_detector
        self.head_expression_estimator = head_expression_estimator
        self.occlusion_aware_generator = occlusion_aware_generator
        self.multi_scale_discriminator = multi_scale_discriminator

        self.train = train
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.multi_scale_discriminator.scales
        self.pyramid = ImagePyramid(self.scales, occlusion_aware_generator.output_channels)

        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        self.idx_tensor = torch.arange(self.train_params['num_bins'], dtype=torch.float32)
        self.num_kp = appearance_feature_extractor.num_kp

        if 'head_pose' in self.loss_weights:
            self.hopenet = Hopenet(requires_grad=False, weight_dir='pretrained/hopenet.pkl')

        if 'perceptual' in self.loss_weights != 0:
            self.vgg = Vgg19(requires_grad=False)
            self.face_vgg = FaceVgg(requires_grad=False, weight_dir='pretrained/face_vgg.pth')

        if torch.cuda.is_available():
            self.hopenet = self.hopenet.cuda()
            self.vgg = self.vgg.cuda()
            self.face_vgg = self.face_vgg.cuda()

    def get_rotation_matrix(self, yaw, pitch, roll, idx_tensor):
        idx_tensor = idx_tensor.to(yaw.device)

        yaw = torch.sum(torch.softmax(yaw, dim=1) * idx_tensor, 1) * 3 - 99
        pitch = torch.sum(torch.softmax(pitch, dim=1) * idx_tensor, 1) * 3 - 99
        roll = torch.sum(torch.softmax(roll, dim=1) * idx_tensor, 1) * 3 - 99

        yaw = -yaw * pi / 180
        pitch = pitch * pi / 180
        roll = roll * pi / 180

        cy, cp, cr = torch.cos(yaw), torch.cos(pitch), torch.cos(roll)
        sy, sp, sr = torch.sin(yaw), torch.sin(pitch), torch.sin(roll)

        rotation_matrix = torch.stack([
            torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], dim=1),
            torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], dim=1),
            torch.stack([-sp, cp * sr, cp * cr], dim=1)
        ], dim=1).unsqueeze(1).repeat(1, self.num_kp, 1, 1)

        return rotation_matrix, torch.stack([yaw, pitch, roll], dim=1)

    def get_keypoint(self, img_source, img_driving, canonical_keypoint):

        """ Just using pretrained hopenet for getting head pose angles """
        # (yaw_s, pitch_s, roll_s), translation_s, deformation_s = self.head_expression_estimator(img_source)
        # (yaw_d, pitch_d, roll_d), translation_d, deformation_d = self.head_expression_estimator(img_driving)
        
        translation_s, deformation_s = self.head_expression_estimator(img_source)
        translation_d, deformation_d = self.head_expression_estimator(img_driving)

        yaw_s, pitch_s, roll_s = self.hopenet(img_source)
        yaw_d, pitch_d, roll_d = self.hopenet(img_driving)

        rotation_s, euler_angle_s = self.get_rotation_matrix(yaw_s, pitch_s, roll_s, self.idx_tensor)
        rotation_d, euler_angle_d = self.get_rotation_matrix(yaw_d, pitch_d, roll_d, self.idx_tensor)

        kp_source = dict(deformation=deformation_s, euler_angle=euler_angle_s)
        kp_driving = dict(deformation=deformation_d, euler_angle=euler_angle_d)

        rotated_keypoints_s = torch.matmul(rotation_s, canonical_keypoint['keypoints'].unsqueeze(-1)).squeeze(-1)
        rotated_keypoints_d = torch.matmul(rotation_d, canonical_keypoint['keypoints'].unsqueeze(-1)).squeeze(-1)

        kp_source['keypoints'] = rotated_keypoints_s + translation_s + deformation_s
        kp_driving['keypoints'] = rotated_keypoints_d + translation_d + deformation_d

        if 'jacobian' in canonical_keypoint:
            kp_source['jacobian'] = torch.matmul(rotation_s, canonical_keypoint['jacobian'])
            kp_driving['jacobian'] = torch.matmul(rotation_d, canonical_keypoint['jacobian'])

        return kp_source, kp_driving

    def calculate_loss_values(self, x, pyramid_real, pyramid_generated, generated, kp_source, kp_driving):
        loss_values = dict()

        """ Perceptual loss """
        perceptual = 0.
        for scale in self.scales:
            vgg_real = self.vgg(pyramid_real[f'prediction_{scale}'])
            vgg_generated = self.vgg(pyramid_generated[f'prediction_{scale}'])

            for i, layer_weight in enumerate(self.loss_weights['perceptual_layers']):
                perceptual += layer_weight * torch.abs(vgg_real[i] - vgg_generated[i].detach()).mean()
        
        face_vgg_real = self.face_vgg(pyramid_real['prediction_1'])
        face_vgg_generated = self.face_vgg(pyramid_generated['prediction_1'])
        perceptual += torch.abs(face_vgg_real - face_vgg_generated.detach()).mean()

        loss_values['perceptual'] = self.loss_weights['perceptual'] * perceptual

        """ GAN loss """
        generator_gan = 0.
        feature_matching = 0.
        out_dict_real = self.multi_scale_discriminator(pyramid_real)
        out_dict_generated = self.multi_scale_discriminator(pyramid_generated)

        for scale in self.disc_scales:
            generator_gan -= out_dict_generated[f'prediction_map_{scale}'].mean()
        loss_values['generator_gan'] = self.loss_weights['generator_gan'] * generator_gan

        for scale in self.disc_scales:
            key = f'feature_maps_{scale}'
            for i, (feature_real, feature_generated) in enumerate(zip(out_dict_real[key], out_dict_generated[key])):
                feature_matching += torch.abs(feature_real - feature_generated).mean()
        loss_values['feature_matching'] = self.loss_weights['feature_matching'] * feature_matching

        """ Equivariance loss """
        transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
        transformed_frame = transform.transform_frame(x['driving'])
        transformed_kp = self.canonical_keypoint_detector(transformed_frame)

        generated['transformed_frame'] = transformed_frame
        generated['transformed_kp'] = transformed_kp

        if 'equivariance_keypoints' in self.loss_weights:
            equivariance_keypoints = torch.abs(kp_driving['keypoints'][..., 1:] - transform.warp_coordinates(transformed_kp['keypoints'][..., 1:])).mean()
            loss_values['equivariance_keypoints'] = self.loss_weights['equivariance_keypoints'] * equivariance_keypoints

        if 'jacobian' in transformed_kp and 'equivariance_jacobian' in self.loss_weights:
            jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['keypoints'][..., 1:]),
                                                transformed_kp['jacobian'][..., 1:, 1:])

            normed_driving = torch.inverse(kp_driving['jacobian'][..., 1:, 1:])
            normed_transformed = jacobian_transformed
            equivariance_jacobian = torch.matmul(normed_driving, normed_transformed)

            eye = torch.eye(2).view(1, 1, 2, 2).type(equivariance_jacobian.type())
            equivariance_jacobian = torch.abs(eye - equivariance_jacobian).mean()
            loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * equivariance_jacobian

        """ Keypoint prior loss """
        keypoint_prior = 0.

        keypoints_repeat = torch.cat([kp_source['keypoints'], kp_driving['keypoints']], dim=0).unsqueeze(2).repeat(1, 1, self.num_kp, 1)
        keypoints_diff = keypoints_repeat - keypoints_repeat.transpose(1, 2)
        keypoints_diff = F.relu(0.1 - torch.sum(keypoints_diff * keypoints_diff, dim=-1))

        mask = torch.tril(torch.ones_like(keypoints_diff[0]), diagonal=-1)
        keypoints_diff = keypoints_diff * mask
        keypoint_prior += torch.sum(keypoints_diff, dim=[1, 2]).mean()

        mean_depth = torch.mean(torch.cat([kp_source['keypoints'], kp_driving['keypoints']], dim=0)[..., 0], dim=-1)
        keypoint_prior += torch.abs(mean_depth - self.train_params['keypoint_depth_target']).mean()
        loss_values['keypoint_prior'] = self.loss_weights['keypoint_prior'] * keypoint_prior

        """ Head pose loss """
        # head_pose = 0.
        # yaw_target_source, pitch_target_source, roll_target_source = self.hopenet(x['source'])
        # yaw_target_driving, pitch_target_driving, roll_target_driving = self.hopenet(x['driving'])

        # _, target_euler_angle_source = self.get_rotation_matrix(yaw_target_source,
        #                                                         pitch_target_source,
        #                                                         roll_target_source,
        #                                                         self.idx_tensor)
        # _, target_euler_angle_driving = self.get_rotation_matrix(yaw_target_driving,
        #                                                          pitch_target_driving,
        #                                                          roll_target_driving,
        #                                                          self.idx_tensor)
        
        # head_pose += torch.sum(torch.abs(kp_source['euler_angle'] - target_euler_angle_source), dim=-1).mean()
        # head_pose += torch.sum(torch.abs(kp_driving['euler_angle'] - target_euler_angle_driving), dim=-1).mean()
        # loss_values['head_pose'] = self.loss_weights['head_pose'] * head_pose

        """ Deformation prior loss """
        deformation_prior = 0.
        deformation_prior += torch.sum(torch.abs(kp_source['deformation']), dim=-1).mean()
        deformation_prior += torch.sum(torch.abs(kp_driving['deformation']), dim=-1).mean()
        loss_values['deformation_prior'] = self.loss_weights['deformation_prior'] * deformation_prior
        return loss_values

    def forward(self, x):
        appearance_feature = self.appearance_feature_extractor(x['source'])
        canonical_keypoint = self.canonical_keypoint_detector(x['source'])
        kp_source, kp_driving = self.get_keypoint(x['source'], x['driving'], canonical_keypoint)

        generated = self.occlusion_aware_generator(source_feature=appearance_feature, kp_source=kp_source,
                                                   kp_driving=kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        pyramid_real = self.pyramid(x['driving'])
        pyramid_generated = self.pyramid(generated['prediction'])

        loss_values = None
        if self.train:
            loss_values = self.calculate_loss_values(x, pyramid_real, pyramid_generated, generated, kp_source, kp_driving)
        return loss_values, generated


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, multi_scale_discriminator, generator_output_channels, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.multi_scale_discriminator = multi_scale_discriminator
        self.train_params = train_params
        self.scales = self.multi_scale_discriminator.scales
        self.pyramid = ImagePyramid(self.scales, generator_output_channels)

        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    @staticmethod
    def get_grid(inputs, is_real=True):
        if is_real:
            grid = torch.FloatTensor(inputs.shape).fill_(1.0)
        else:
            grid = torch.FloatTensor(inputs.shape).fill_(0.0)

        grid = grid.to(inputs.device)
        return grid

    def calculate_loss_values(self, out_dict_real, out_dict_generated):
        loss_values = {}

        discriminator_gan = 0.
        for scale in self.scales:
            patch_grid_real = self.get_grid(out_dict_real[f'prediction_map_{scale}'], is_real=True)
            discriminator_gan += self.loss_weights['discriminator_gan'] * F.relu(patch_grid_real - out_dict_real[f'prediction_map_{scale}']).mean()
            discriminator_gan += self.loss_weights['discriminator_gan'] * F.relu(patch_grid_real + out_dict_generated[f'prediction_map_{scale}']).mean()
                                 
        loss_values['discriminator_gan'] = discriminator_gan
        return loss_values

    def forward(self, x, generated):
        pyramid_real = self.pyramid(x['driving'])
        pyramid_generated = self.pyramid(generated['prediction'].detach())

        out_dict_real = self.multi_scale_discriminator(pyramid_real)
        out_dict_generated = self.multi_scale_discriminator(pyramid_generated)

        loss_values = self.calculate_loss_values(out_dict_real, out_dict_generated)
        return loss_values
