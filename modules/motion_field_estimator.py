import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.util import kp2gaussian, make_coordinate_grid
from modules.blocks import Hourglass, AntiAliasInterpolation2d


class MotionFieldEstimator(nn.Module):
    """
    Estimating motion field & occlusion mask. Return K motion field masks and occlusion mask.
    """

    def __init__(self, depth, num_kp, source_channels, estimate_occlusion_map, block_expansion,
                 num_blocks, max_features, compress_channels, scale_factor, kp_variance):
        super(MotionFieldEstimator, self).__init__()

        self.compressor = nn.Conv3d(source_channels, compress_channels, kernel_size=(1, 1, 1))

        self.hourglass = Hourglass(block_expansion=block_expansion,
                                   in_features=(num_kp + 1) * (compress_channels + 1),
                                   depth=depth,
                                   num_blocks=num_blocks,
                                   max_features=max_features,
                                   is_keypoint=False)

        self.mask = nn.Conv3d(in_channels=(self.hourglass.decoder.out_filters + (num_kp + 1) * (compress_channels + 1)),
                              out_channels=num_kp + 1,
                              kernel_size=(3, 3, 3),  # (7, 7, 7) is replaced to (3, 3, 3)
                              padding=(1, 1, 1))

        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(
                in_channels=(self.hourglass.decoder.out_filters + (num_kp + 1) * (compress_channels + 1)) * depth,
                out_channels=1,
                kernel_size=(7, 7),
                padding=(3, 3))
        else:
            self.occlusion = None

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(compress_channels, self.scale_factor)

    def create_heatmap_representations(self, source_feature, kp_source, kp_driving):
        spatial_size = source_feature.shape[2:]
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        # adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2], dtype=heatmap.dtype)
        zeros = zeros.to(heatmap.device)

        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap

    def create_sparse_motions(self, source_feature, kp_source, kp_driving):
        batch_size, _, d, h, w = source_feature.shape
        identity_grid = make_coordinate_grid((d, h, w), dtype=kp_source['keypoints'].type()).to(source_feature.device)
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)
        coordinate_grid = identity_grid - kp_driving['keypoints'].view(batch_size, self.num_kp, 1, 1, 1, 3)

        if 'jacobian' in kp_driving:
            jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
            jacobian = jacobian.unsqueeze(2).unsqueeze(2).unsqueeze(2)
            jacobian = jacobian.repeat(1, 1, d, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = coordinate_grid + kp_source['keypoints'].view(batch_size, self.num_kp, 1, 1, 1, 3)

        # adding background feature
        identity_grid = identity_grid.repeat(batch_size, 1, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        return sparse_motions

    def create_deformed_source_image(self, source_feature, sparse_motions):
        batch_size, _, d, h, w = source_feature.shape
        source_repeat = source_feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(batch_size * (self.num_kp + 1), -1, d, h, w)
        sparse_motions = sparse_motions.view(batch_size * (self.num_kp + 1), d, h, w, -1)

        sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((batch_size, self.num_kp + 1, -1, d, h, w))
        return sparse_deformed

    def forward(self, source_feature, kp_source, kp_driving):
        source_feature = self.compressor(source_feature)

        batch_size, _, d, h, w = source_feature.shape

        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_feature, kp_source, kp_driving)
        sparse_motion = self.create_sparse_motions(source_feature, kp_source, kp_driving)
        deformed_source = self.create_deformed_source_image(source_feature, sparse_motion)
        out_dict['sparse_deformed'] = deformed_source

        inputs = torch.cat([heatmap_representation, deformed_source], dim=2)
        inputs = inputs.view(batch_size, -1, d, h, w)

        prediction = self.hourglass(inputs)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask

        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 4, 1)
        out_dict['deformation'] = deformation

        if self.occlusion:
            occlusion_map = self.occlusion(prediction.view(batch_size, -1, h, w))
            occlusion_map = torch.sigmoid(occlusion_map)
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
