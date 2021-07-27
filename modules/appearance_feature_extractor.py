import torch.nn as nn
import torch.nn.functional as F
from modules.util import DownBlock2d, ResBlock3d
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


class AppearanceFeatureExtractor(nn.Module):
    """
    Extracting appearance feature. Return feature map that encodes source image.
    """

    def __init__(self, block_expansion=64, num_channels=3, res_block_features=32,
                 num_down_blocks=2, num_res_blocks=6, depth=16, **kwargs):
        super(AppearanceFeatureExtractor, self).__init__()

        self.depth = depth

        self.first = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=block_expansion, kernel_size=(7, 7), padding=(3, 3)),
            BatchNorm2d(block_expansion)
        )

        self.down_blocks = nn.ModuleList([DownBlock2d(in_features=block_expansion * (2 ** i),
                                                      out_features=block_expansion * (2 ** (i+1)),
                                                      kernel_size=(3, 3),
                                                      padding=(1, 1)) for i in range(num_down_blocks)])

        self.conv = nn.Conv2d(block_expansion*4, block_expansion*8, kernel_size=(1, 1))

        self.res_blocks = nn.ModuleList([ResBlock3d(in_features=res_block_features,
                                                    kernel_size=(3, 3, 3),
                                                    padding=(1, 1, 1)) for _ in range(num_res_blocks)])

    def forward(self, x):
        out = self.first(x)
        out = F.relu(out)

        for down_block in self.down_blocks:
            out = down_block(out)

        out = self.conv(out)
        b, _, h, w = out.shape

        out = out.view(b, -1, self.depth, h, w)
        for res_block in self.res_blocks:
            out = res_block(out)

        return out
