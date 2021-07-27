import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid


class CanonicalKeypointDetector(nn.Module):
    """
    Detecting canonical keypoints. Return keypoints position and jacobian near each keypoints.
    """

    def __init__(self, block_expansion=64, num_kp=20, num_channels=3, max_features=1024, num_blocks=5,
                 depth=16, temperature=0.1, estimate_jacobian=False, scale_factor=1, pad=(3, 3, 3), **kwargs):
        super(CanonicalKeypointDetector, self).__init__()

        self.predictor = Hourglass(block_expansion=block_expansion,
                                   in_features=num_channels,
                                   depth=depth,
                                   num_blocks=num_blocks,
                                   max_features=max_features,
                                   encoder_3d=False)

        self.kp = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7, 7), padding=pad)

        if estimate_jacobian:
            self.jacobian = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=9 * num_kp, kernel_size=(7, 7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1] * num_kp, dtype=torch.float))
        else:
            self.jacobian = None

        self.num_kp = num_kp
        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    @staticmethod
    def gaussian2kp(heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze(0).unsqueeze(0)
        keypoints = (heatmap * grid).sum(dim=(2, 3, 4))
        kp = {'keypoints': keypoints}

        return kp

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_kp, 9, final_shape[2], final_shape[3], final_shape[4])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 9, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.reshape(jacobian.shape[0], jacobian.shape[1], 3, 3)
            out['jacobian'] = jacobian

        return out
