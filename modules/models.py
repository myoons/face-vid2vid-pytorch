import face_alignment

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.pretrained_models import Vgg19, Hopenet, FaceVgg
from modules.util import Transform, ImagePyramid
from torchvision import transforms


class GeneratorFullModel(nn.Module):
    """
    Merge all generator related upadtes into single model for better multi-gpu usage
    """

    def __init__(self, af_extractor, kp_detector, he_estimator, generator, discriminator, train_params, args,
                 train=False):
        super(GeneratorFullModel, self).__init__()

        self.af_extractor = af_extractor
        self.kp_detector = kp_detector
        self.he_estimator = he_estimator
        self.generator = generator
        self.discriminator = discriminator

        self.args = args
        self.train = train
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramid(self.scales, generator.output_channels)

        self.loss_weights = train_params['loss_weights']
        self.idx_tensor = torch.arange(66, dtype=torch.float32).to(args.local_rank)
        self.num_kp = af_extractor.num_kp

        self.transform_hopenet = transforms.Compose([transforms.Resize(size=(224, 224)),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225])])

        self.hopenet = Hopenet(requires_grad=False, weight_dir='pretrained/hopenet.pkl')
        self.hopenet.eval()

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
                                               device=f'cuda:{args.local_rank}')

        if 'perceptual' in self.loss_weights != 0:
            self.vgg = Vgg19(requires_grad=False)
            self.face_vgg = FaceVgg(requires_grad=False, weight_dir='pretrained/face_vgg.pth')
            self.vgg.eval()
            self.face_vgg.eval()

    @staticmethod
    def pred_to_degree(pred, idx_tensor):
        degree = torch.sum(torch.softmax(pred, dim=-1) * idx_tensor, 1) * 3 - 99
        degree = degree / 180 * 3.14
        return degree

    @staticmethod
    def get_rotation_matrix(yaw, pitch, roll):
        roll = roll.unsqueeze(1)
        pitch = pitch.unsqueeze(1)
        yaw = yaw.unsqueeze(1)

        roll_mat = torch.cat([torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll),
                              torch.zeros_like(roll), torch.cos(roll), -torch.sin(roll),
                              torch.zeros_like(roll), torch.sin(roll), torch.cos(roll)], dim=1)
        roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

        pitch_mat = torch.cat([torch.cos(pitch), torch.zeros_like(pitch), torch.sin(pitch),
                               torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch),
                               -torch.sin(pitch), torch.zeros_like(pitch), torch.cos(pitch)], dim=1)
        pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

        yaw_mat = torch.cat([torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw),
                             torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw),
                             torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=1)
        yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

        rotation_matrix = torch.einsum('bij,bjk,bkm->bim', roll_mat, pitch_mat, yaw_mat)
        return rotation_matrix

    def get_keypoint(self, he_out, kp_canonical):
        value = {}

        kp = kp_canonical['keypoints']
        yaw, pitch, roll = he_out['yaw'], he_out['pitch'], he_out['roll']
        translation = he_out['translation']
        deformation = he_out['deformation']

        yaw = self.pred_to_degree(yaw, self.idx_tensor)
        pitch = self.pred_to_degree(pitch, self.idx_tensor)
        roll = self.pred_to_degree(roll, self.idx_tensor)

        rotation_matrix = self.get_rotation_matrix(yaw, pitch, roll)

        kp_rotated = torch.einsum('bmp,bkp->bkm', rotation_matrix, kp)
        kp_translated = kp_rotated + translation
        kp_deformed = kp_translated + deformation
        value['keypoints'] = kp_deformed

        if 'jacobian' in kp_canonical:
            jacobian = kp_canonical['jacobian']  # (bs, k ,3, 3)
            jacobian_rotated = torch.einsum('bmp,bkps->bkms', rotation_matrix, jacobian)
            value['jacobian'] = jacobian_rotated

        return value

    def get_head_pose(self, images, he, device):

        b, c, h, w = images.shape
        preds = self.fa.face_detector.detect_from_batch(images * 255)

        aligned_images = []
        for idx, pred in enumerate(preds):
            if len(pred) == 1:
                lt_x, lt_y, rb_x, rb_y, _ = pred[0]
                bw = rb_x - lt_x
                bh = rb_y - lt_y

                x_min = int(max(lt_x - 2 * bw / 4, 0))
                x_max = int(min(rb_x + 2 * bw / 4, w-1))
                y_min = int(max(lt_y - 3 * bh / 4, 0))
                y_max = int(min(rb_y + bh / 4, h-1))

                if y_min == y_max or x_min == x_max:
                    img = images[idx]
                    img = self.transform_hopenet(img)
                    aligned_images.append(img)
                else:
                    img = images[idx, :, y_min:y_max, x_min:x_max]
                    img = self.transform_hopenet(img)
                    aligned_images.append(img)
            else:
                img = images[idx]
                img = self.transform_hopenet(img)
                aligned_images.append(img)

        aligned_tensor = torch.stack(aligned_images)
        yaw, pitch, roll = self.hopenet(aligned_tensor)
        
        he['yaw'] = yaw
        he['pitch'] = pitch
        he['roll'] = roll

    def calculate_loss_values(self, x, pyramid_real, pyramid_generated, generated, kp_canonical, kp_driving,
                              he_driving):
        loss_values = dict()

        """ Perceptual loss """
        if 'perceptual' in self.loss_weights:
            perceptual = 0.
            for scale in self.scales:
                vgg_real = self.vgg(pyramid_real[f'prediction_{scale}'])
                vgg_generated = self.vgg(pyramid_generated[f'prediction_{scale}'])

                for i, layer_weight in enumerate(self.loss_weights['perceptual']):
                    perceptual += layer_weight * torch.abs(vgg_real[i] - vgg_generated[i].detach()).mean()

            if 'face_perceptual' in self.loss_weights:
                face_vgg_real = self.face_vgg(pyramid_real['prediction_1'])
                face_vgg_generated = self.face_vgg(pyramid_generated['prediction_1'])
                perceptual += self.loss_weights['face_perceptual'] * torch.abs(face_vgg_real - face_vgg_generated.detach()).mean()

            loss_values['perceptual'] = perceptual

        """ GAN loss """
        if 'generator_gan' in self.loss_weights:
            generator_gan = 0.
            out_dict_real = self.discriminator(pyramid_real)
            out_dict_generated = self.discriminator(pyramid_generated)

            for scale in self.disc_scales:
                generator_gan -= out_dict_generated[f'prediction_map_{scale}'].mean()

            loss_values['generator_gan'] = self.loss_weights['generator_gan'] * generator_gan

            if 'feature_matching' in self.loss_weights:
                feature_matching = 0.
                for scale in self.disc_scales:
                    key = f'feature_maps_{scale}'
                    for i, (feature_real, feature_generated) in enumerate(zip(out_dict_real[key], out_dict_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue

                        feature_matching += self.loss_weights['feature_matching'][i] * torch.abs(feature_real - feature_generated).mean()

                loss_values['feature_matching'] = feature_matching

        """ Equivariance loss """
        transform = Transform(x['driving'].shape[0], self.args.local_rank, **self.train_params['transform_params'])
        transformed_frame = transform.transform_frame(x['driving'])
        transformed_he_driving = self.he_estimator(transformed_frame)

        self.get_head_pose(transformed_frame, transformed_he_driving, self.args.local_rank)
        transformed_kp = self.get_keypoint(transformed_he_driving, kp_canonical)

        generated['transformed_frame'] = transformed_frame
        generated['transformed_kp'] = transformed_kp

        if 'equivariance_keypoints' in self.loss_weights:
            kp_driving_2d = kp_driving['keypoints'][..., :2]
            transformed_kp_2d = transformed_kp['keypoints'][..., :2]
            equivariance_keypoints = torch.abs(kp_driving_2d - transform.warp_coordinates(transformed_kp_2d)).mean()

            loss_values['equivariance_keypoints'] = self.loss_weights['equivariance_keypoints'] * equivariance_keypoints

        if 'jacobian' in transformed_kp and 'equivariance_jacobian' in self.loss_weights:
            transformed_kp_2d = transformed_kp['keypoints'][..., :2]
            transformed_jacobian_2d = transformed_kp['jacobian'][..., :2, :2]
            jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp_2d), transformed_jacobian_2d)

            jacobian_2d = kp_driving['jacobian'][..., :2, :2]
            normed_driving = torch.inverse(jacobian_2d)
            normed_transformed = jacobian_transformed
            equivariance_jacobian = torch.matmul(normed_driving, normed_transformed)

            eye = torch.eye(2).view(1, 1, 2, 2).type(equivariance_jacobian.type())
            equivariance_jacobian = torch.abs(eye - equivariance_jacobian).mean()

            loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * equivariance_jacobian

        """ Keypoint prior loss """
        if 'keypoint_prior' in self.loss_weights:
            keypoint_prior = 0.

            keypoints_repeat = kp_driving['keypoints'].unsqueeze(2).repeat(1, 1, self.num_kp, 1)
            keypoints_diff = keypoints_repeat - keypoints_repeat.transpose(1, 2)
            keypoints_diff = F.relu(0.1 - torch.sum(keypoints_diff * keypoints_diff, dim=-1))

            # mask = torch.tril(torch.ones_like(keypoints_diff[0]), diagonal=-1)
            # keypoints_diff = keypoints_diff * mask
            keypoint_prior += torch.sum(keypoints_diff, dim=[1, 2]).mean()

            mean_depth = kp_driving['keypoints'][:, :, -1].mean(-1)
            keypoint_prior += torch.abs(mean_depth - self.train_params['keypoint_depth_target']).mean()

            loss_values['keypoint_prior'] = self.loss_weights['keypoint_prior'] * keypoint_prior

        """ Head pose loss """
        if 'head_pose' in self.loss_weights:
            transform_hopenet = transforms.Compose([transforms.Resize(size=(224, 224)),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])

            driving_for_hopenet = transform_hopenet(x['driving'])

            yaw_target, pitch_target, roll_target = self.hopenet(driving_for_hopenet)
            yaw_target = self.pred_to_degree(yaw_target, self.idx_tensor)
            pitch_target = self.pred_to_degree(pitch_target, self.idx_tensor)
            roll_target = self.pred_to_degree(roll_target, self.idx_tensor)

            yaw, pitch, roll = he_driving['yaw'], he_driving['pitch'], he_driving['roll']
            yaw = self.pred_to_degree(yaw)
            pitch = self.pred_to_degree(pitch)
            roll = self.pred_to_degree(roll)

            head_pose = torch.abs(yaw - yaw_target).mean() + torch.abs(pitch - pitch_target).mean() + torch.abs(
                roll - roll_target).mean()

            loss_values['head_pose'] = self.loss_weights['head_pose'] * head_pose

        """ Deformation prior loss """
        if 'deformation_prior' in self.loss_weights:
            deformation_prior = torch.norm(he_driving['deformation'], p=1, dim=-1).mean()

            loss_values['deformation_prior'] = self.loss_weights['deformation_prior'] * deformation_prior

        return loss_values

    def forward(self, x):
        kp_canonical = self.kp_detector(x['source'])
        he_source = self.he_estimator(x['source'])
        he_driving = self.he_estimator(x['driving'])

        self.get_head_pose(x['source'], he_source, self.args.local_rank)
        self.get_head_pose(x['driving'], he_driving, self.args.local_rank)

        kp_source = self.get_keypoint(he_source, kp_canonical)
        kp_driving = self.get_keypoint(he_driving, kp_canonical)

        appearance_feature = self.af_extractor(x['source'])
        generated = self.generator(appearance_feature, kp_source, kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        pyramid_real = self.pyramid(x['driving'])
        pyramid_generated = self.pyramid(generated['prediction'])

        loss_values = None
        if self.train:
            loss_values = self.calculate_loss_values(x, pyramid_real, pyramid_generated, generated, kp_source,
                                                     kp_driving, he_driving)
        return loss_values, generated


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, discriminator, generator_output_channels, train_params, args):
        super(DiscriminatorFullModel, self).__init__()
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramid(self.scales, generator_output_channels)

        self.loss_weights = train_params['loss_weights']
        self.zero_tensor = None

    def get_zero_tensor(self, inp):
        if self.zero_tensor is None:
            self.zero_tensor = torch.zeros_like(inp)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(inp)

    def calculate_loss_values(self, out_dict_real, out_dict_generated):
        loss_values = {}

        discriminator_gan = 0.
        for scale in self.scales:
            key = f'prediction_map_{scale}'
            discriminator_gan -= torch.mean(torch.min(out_dict_real[key] - 1, self.get_zero_tensor(out_dict_real[key])))
            discriminator_gan -= torch.mean(torch.min(-out_dict_generated[key] - 1, self.get_zero_tensor(out_dict_generated[key])))

        loss_values['discriminator_gan'] = self.loss_weights['discriminator_gan'] * discriminator_gan
        return loss_values

    def forward(self, x, generated):
        pyramid_real = self.pyramid(x['driving'])
        pyramid_generated = self.pyramid(generated['prediction'].detach())

        out_dict_real = self.discriminator(pyramid_real)
        out_dict_generated = self.discriminator(pyramid_generated)

        loss_values = self.calculate_loss_values(out_dict_real, out_dict_generated)
        return loss_values
