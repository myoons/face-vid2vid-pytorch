from math import pi, cos, sin
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d


def get_rotation_matrix(yaw, pitch, roll, idx_tensor):
    yaw = torch.sum(torch.softmax(yaw, dim=1) * idx_tensor, 1) * 3 - 99
    pitch = torch.sum(torch.softmax(pitch, dim=1) * idx_tensor, 1) * 3 - 99
    roll = torch.sum(torch.softmax(roll, dim=1) * idx_tensor, 1) * 3 - 99

    y = -yaw * pi / 180
    p = pitch * pi / 180
    r = roll * pi / 180

    cy, cp, cr = cos(y), cos(p), cos(r)
    sy, sp, sr = sin(y), sin(p), sin(r)

    rotation_matrix = torch.tensor([
        [cp*cy, -cp*sy, sp],
        [cr*sy + sr*sp*cy, cr*cy - sr*sp*sy, -sr*cp],
        [sr*sy - cr*sp*cy, sr*cy + cr*sp*sy, cr*cp]
    ]).float().cuda()

    return rotation_matrix, (yaw, pitch, roll)


def get_grid(inputs, is_real=True):
    if is_real:
        grid = torch.FloatTensor(inputs.shape).fill_(1.0)
    else:
        grid = torch.FloatTensor(inputs.shape).fill_(0.0)

    return grid


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform keypoints into gaussian like representation
    """
    mean = kp['keypoints']
    number_of_leading_dimensions = len(mean.shape) - 1

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean[:number_of_leading_dimensions] + (1, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    return out


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    d, h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    z = torch.arange(d).type(type)

    z = (2 * (z / (d - 1)) - 1)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    zz = z.view(-1, 1, 1).repeat(1, w, h)
    xx = x.view(1, -1, 1).repeat(d, 1, h)
    yy = y.view(1, 1, -1).repeat(d, w, 1)

    meshed = torch.cat([zz.unsqueeze_(3), xx.unsqueeze_(3), yy.unsqueeze_(3)], dim=3)
    return meshed


class ResBottleneck(nn.Module):

    def __init__(self, in_features, out_features, down_sample=False):
        super(ResBottleneck, self).__init__()

        if down_sample:
            down = 2
        else:
            down = 1

        self.mid_features = out_features // 4

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=self.mid_features, kernel_size=(1, 1)),
            BatchNorm2d(self.mid_features)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=self.mid_features, out_channels=self.mid_features, kernel_size=(3, 3),
                      stride=(down, down), padding=(1, 1)),
            BatchNorm2d(self.mid_features),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=self.mid_features, out_channels=out_features, kernel_size=(1, 1)),
            BatchNorm2d(out_features)
        )

        if down_sample:
            self.residual = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=(1, 1),
                                      stride=(down, down), bias=False)
        else:
            self.residual = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=(1, 1))

        self.down_sample = down_sample

    def forward(self, x):
        residual = x

        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = self.layer3(out)

        if residual.size() is not out.size():
            residual = self.residual(x)

        out += residual
        return F.relu(out)


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
                              padding=padding)
        self.norm = BatchNorm3d(num_features=out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks, max_features):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(in_features=in_features if i == 0 else min(max_features, block_expansion * (2 ** (i - 1))),
                            out_features=min(max_features, block_expansion * (2 ** i)),
                            kernel_size=(3, 3),
                            padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, depth, num_blocks, max_features):
        super(Decoder, self).__init__()

        self.depth = depth
        self.out_filters = block_expansion // 2

        up_blocks = []
        for i in range(num_blocks)[::-1]:
            up_blocks.append(UpBlock3d(
                in_features=(1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** i)),
                out_features=min(max_features, int(block_expansion * (2 ** (i - 1)))),
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        convs = []
        for i in range(num_blocks)[::-1]:
            convs.append(nn.Conv2d(
                in_channels=min(max_features, block_expansion * (2 ** i)),
                out_channels=depth * min(max_features, block_expansion * (2 ** i)),
                kernel_size=(1, 1)))
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        out = None
        for (up_block, conv) in zip(self.up_blocks, self.convs):
            if out is None:
                out = x.pop()
                b, c, h, w = out.shape
                out = conv(out).view(b, -1, self.depth, h, w)
            else:
                skip = x.pop()
                b, c, h, w = skip.shape
                skip = conv(skip).view(b, -1, self.depth, h, w)
                out = torch.cat([out, skip], dim=1)

            out = up_block(out)

        return out


class Encoder3d(nn.Module):
    """
    Hourglass 3D Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks, max_features):
        super(Encoder3d, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock3d(in_features=in_features if i == 0 else min(max_features, block_expansion * (2 ** (i - 1))),
                            out_features=min(max_features, block_expansion * (2 ** i)),
                            kernel_size=(3, 3, 3),
                            padding=(1, 1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder3d(nn.Module):
    """
    Hourglass 3D Decoder which follows 3D Encoder
    """

    def __init__(self, block_expansion, num_blocks, max_features):
        super(Decoder3d, self).__init__()

        self.out_filters = block_expansion // 2

        up_blocks = []
        for i in range(num_blocks)[::-1]:
            up_blocks.append(UpBlock3d(
                in_features=(1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** i)),
                out_features=min(max_features, int(block_expansion * (2 ** (i - 1)))),
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

    def forward(self, x):
        out = x.pop()
        for idx, up_block in enumerate(self.up_blocks):
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)

        return out


class Hourglass(nn.Module):
    """
    Hourglass (U-Net) architecture for CanonicalKeypointDetector
    """

    def __init__(self, block_expansion, in_features, depth, num_blocks, max_features, encoder_3d=False):
        super(Hourglass, self).__init__()

        if encoder_3d:
            self.encoder = Encoder3d(block_expansion, in_features, num_blocks, max_features)
            self.decoder = Decoder3d(block_expansion, num_blocks, max_features)
        else:
            self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
            self.decoder = Decoder(block_expansion, depth, num_blocks, max_features)

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

        out = F.pad(x, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out
