import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.util import make_coordinate_grid
from modules.blocks import Hourglass, AntiAliasInterpolation2d


class CanonicalKeypointDetector(nn.Module):
    """
    Detecting canonical keypoints. Return keypoints position and jacobian near each keypoints.
    """

    def __init__(self, depth, num_kp, num_channels, block_expansion, max_features, num_blocks,
                 temperature, scale_factor, estimate_jacobian=False, is_keypoint=True):
        super(CanonicalKeypointDetector, self).__init__()

        self.first = Hourglass(block_expansion=block_expansion,
                               in_features=num_channels,
                               depth=depth,
                               num_blocks=num_blocks,
                               max_features=max_features,
                               is_keypoint=is_keypoint)

        self.kp = nn.Conv3d(in_channels=self.first.out_filters,
                            out_channels=num_kp,
                            kernel_size=(3, 3, 3),  # (7, 7, 7) is replaced to (3, 3, 3)
                            padding=(1, 1, 1))

        if estimate_jacobian:
            self.jacobian = nn.Conv3d(in_channels=self.first.out_filters,
                                      out_channels=9 * num_kp,
                                      kernel_size=(3, 3, 3),  # (7, 7, 7) is replaced to (3, 3, 3)
                                      padding=(1, 1, 1))
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.Tensor([1, 0, 0, 0, 1, 0, 0, 0, 1] * num_kp).float())
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
        grid = make_coordinate_grid(shape[2:], dtype=heatmap.type()).unsqueeze(0).unsqueeze(0).to(heatmap.device)
        keypoints = (heatmap * grid).sum(dim=(2, 3, 4))
        kp = {'keypoints': keypoints}

        return kp

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        out = self.first(x)
        keypoints = self.kp(out)

        final_shape = keypoints.shape
        heatmap = keypoints.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=-1)
        heatmap = heatmap.view(*final_shape)

        result = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(out)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_kp, 9, final_shape[2], final_shape[3], final_shape[4])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 9, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.reshape(jacobian.shape[0], jacobian.shape[1], 3, 3)
            result['jacobian'] = jacobian

        return result
