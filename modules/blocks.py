import torch
from torch import nn
import torch.nn.functional as F
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from sync_batchnorm.batchnorm import SynchronizedBatchNorm3d as BatchNorm3d


class ResBottleneck(nn.Module):

    def __init__(self, in_features, down_sample=False):
        super(ResBottleneck, self).__init__()

        if down_sample:
            down = 2
        else:
            down = 1

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=in_features // 4, kernel_size=(1, 1)),
            BatchNorm2d(in_features // 4, affine=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_features // 4, out_channels=in_features // 4, kernel_size=(3, 3),
                      stride=(down, down), padding=(1, 1)),
            BatchNorm2d(in_features // 4, affine=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=in_features // 4, out_channels=in_features, kernel_size=(1, 1)),
            BatchNorm2d(in_features, affine=True)
        )

        if down_sample:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=(1, 1),
                          stride=(down, down)),
                BatchNorm2d(in_features, affine=True))

        self.down_sample = down_sample

    def forward(self, x):
        residual = x

        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = self.layer3(out)

        if self.down_sample:
            residual = self.residual(x)

        out += residual
        out = F.relu(out)
        return out

class ResBlock2d(nn.Module):

    def __init__(self, in_features, kernel_size=(3, 3), padding=(1, 1)):
        super(ResBlock2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):

    def __init__(self, in_features, out_features, kernel_size=(3, 3), padding=(1, 1)):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):

    def __init__(self, in_features, out_features, kernel_size=(3, 3), padding=(1, 1)):
        super(DownBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm2d(num_features=out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class ResBlock3d(nn.Module):

    def __init__(self, in_features, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
        super(ResBlock3d, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm3d(in_features, affine=True)
        self.norm2 = BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock3d(nn.Module):

    def __init__(self, in_features, out_features, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
        super(UpBlock3d, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        _, _, d, h, w = x.shape
        out = F.interpolate(x, size=(d, h * 2, w * 2))
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock3d(nn.Module):

    def __init__(self, in_features, out_features, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
        super(DownBlock3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, stride=(1, 2, 2))
        self.norm = BatchNorm3d(num_features=out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class CanonicalKeypointEncoder(nn.Module):
    """
    Hourglass Canonical Keypoint Detector Encoder, consists of DownBlock2d & 1x1 2d to 3d mapping
    """

    def __init__(self, block_expansion, in_features, num_blocks, depth, max_features):
        super(CanonicalKeypointEncoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(in_features=in_features if i == 0 else min(max_features, block_expansion * (2 ** (i - 1))),
                            out_features=min(max_features, block_expansion * (2 ** i)),
                            kernel_size=(3, 3),
                            padding=(1, 1)),
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        convs = []
        for i in range(num_blocks)[::-1]:
            convs.append(nn.Conv2d(
                in_channels=min(max_features, block_expansion * (2 ** i)),
                out_channels=depth * min(max_features, block_expansion * (2 ** i)),
                kernel_size=(1, 1))
            )
        self.convs = nn.ModuleList(convs)
        self.depth = depth

    def forward(self, x):
        temp = [x]
        for down_block in self.down_blocks:
            temp.append(down_block(temp[-1]))

        outs = []
        for conv in self.convs:
            target = temp.pop()
            b, c, h, w = target.shape
            outs.append(conv(target).view(b, c, self.depth, h, w))

        return outs[::-1]


class MotionFieldEncoder(nn.Module):
    """
    Hourglass 3D Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks, max_features):
        super(MotionFieldEncoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock3d(in_features=in_features if i == 0 else min(max_features, block_expansion * (2 ** (i - 1))),
                            out_features=min(max_features, block_expansion * (2 ** i)),
                            kernel_size=(3, 3, 3),
                            padding=(1, 1, 1), )
            )

        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass 3D Decoder
    """

    def __init__(self, block_expansion, num_blocks, depth, max_features):
        super(Decoder, self).__init__()

        self.depth = depth
        self.out_filters = block_expansion // 2

        up_blocks = []
        for i in range(num_blocks)[::-1]:
            up_blocks.append(
                UpBlock3d(in_features=(1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** i)),
                          out_features=min(max_features, int(block_expansion * (2 ** (i - 1)))),
                          kernel_size=(3, 3, 3),
                          padding=(1, 1, 1))
            )
        self.up_blocks = nn.ModuleList(up_blocks)

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            if x:
                out = torch.cat([out, x.pop()], dim=1)

        return out


class Hourglass(nn.Module):
    """
    Hourglass (U-Net) architecture for CanonicalKeypointDetector
    """

    def __init__(self, block_expansion, in_features, depth, num_blocks, max_features, is_keypoint=False):
        super(Hourglass, self).__init__()

        if is_keypoint:
            self.encoder = CanonicalKeypointEncoder(block_expansion, in_features, num_blocks, depth, max_features)
        else:
            self.encoder = MotionFieldEncoder(block_expansion, in_features, num_blocks, max_features)

        self.decoder = Decoder(block_expansion, num_blocks, depth, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()

        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1

        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]

        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, x):
        if self.scale == 1.0:
            return x

        out = F.pad(x, [self.ka, self.kb, self.ka, self.kb])
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out
