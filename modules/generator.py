import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.blocks import ResBlock2d, UpBlock2d
from modules.dense_motion import DenseMotionNetwork


class OcclusionAwareGenerator(nn.Module):
    """
    Generator that given source feature and dense motion field. Returns output image.
    """

    def __init__(self, depth, num_kp, num_channels, output_channels, block_expansion, source_channels,
                 num_res_blocks, num_up_blocks, estimate_occlusion_map=False, dense_motion=None):
        super(OcclusionAwareGenerator, self).__init__()

        self.num_channels = num_channels

        if dense_motion:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp,
                                                           depth=depth,
                                                           source_channels=source_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion)
        else:
            self.dense_motion_network = None

        self.first = nn.Sequential(
            nn.Conv2d(depth * source_channels, block_expansion * 4, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(block_expansion * 4),
            nn.LeakyReLU(),
            nn.Conv2d(block_expansion * 4, block_expansion * 4, kernel_size=(1, 1))
        )

        res_blocks = [ResBlock2d(in_features=block_expansion * 4,
                                 kernel_size=(3, 3),
                                 padding=(1, 1)) for _ in range(num_res_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)

        up_blocks = [UpBlock2d(in_features=block_expansion * (2 ** (i + 1)),
                               out_features=block_expansion * (2 ** i),
                               kernel_size=(3, 3),
                               padding=(1, 1)) for i in range(num_up_blocks)[::-1]]
        self.up_blocks = nn.Sequential(*up_blocks)

        self.final = nn.Conv2d(block_expansion, out_channels=output_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.output_channels = output_channels

    @staticmethod
    def deform_feature(source_feature, deformation):
        _, d_deform, h_deform, w_deform, _ = deformation.shape
        _, _, d_feature, h_feature, w_feature = source_feature.shape

        if (d_deform, h_deform, w_deform) != (d_feature, h_feature, w_feature):
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d_feature, h_feature, w_feature), mode='trilinear')
            deformation = deformation.permute(0, 2, 3, 4, 1)

        return F.grid_sample(source_feature, deformation)

    def forward(self, source_feature, kp_source, kp_driving):

        output_dict = {}
        dense_motion_output = self.dense_motion_network(source_feature, kp_source, kp_driving)
        output_dict['mask'] = dense_motion_output['mask']

        if 'occlusion_map' in dense_motion_output:
            occlusion_map = dense_motion_output['occlusion_map']
            output_dict['occlusion_map'] = occlusion_map
        else:
            occlusion_map = None

        deformation = dense_motion_output['deformation']
        deformed_feature = self.deform_feature(source_feature, deformation)

        bs, c, d, h, w = deformed_feature.shape
        deformed_feature = deformed_feature.view(bs, c * d, h, w)
        out = self.first(deformed_feature)

        if occlusion_map is not None:
            if out.shape[2:] != occlusion_map.shape[2:]:
                occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
            out *= occlusion_map

        out = self.res_blocks(out)
        out = self.up_blocks(out)
        out = self.final(out)
        out = torch.sigmoid(out)

        output_dict['prediction'] = out
        return output_dict
