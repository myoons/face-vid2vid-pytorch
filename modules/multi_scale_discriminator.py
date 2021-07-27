import torch.nn as nn
import torch.nn.functional as F


class DownBlock2d(nn.Module):

    def __init__(self, in_features, out_features, stride=(2, 2), kernel_size=(4, 4), padding=(1, 1),
                 negative_slope=0.2, act=False, norm=False, **kwargs):
        super(DownBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              stride=stride, padding=padding)

        if act:
            self.act = nn.LeakyReLU(negative_slope)
        else:
            self.act = None

        if norm:
            self.norm = nn.InstanceNorm2d(out_features)
        else:
            self.norm = None

    def forward(self, x):
        out = self.conv(x)
        if self.norm:
            out = self.norm(out)

        if self.act:
            out = self.act(out)

        return out


class PatchDiscriminator(nn.Module):
    """
    Discriminator similar with PatchGAN.
    """

    def __init__(self, in_channels=3, max_features=512, block_expansion=64, negative=0.2, num_blocks=5, **kwargs):
        super(PatchDiscriminator, self).__init__()

        down_blocks = [DownBlock2d(in_features=in_channels if i == 0 else min(max_features, block_expansion * (2 ** (i - 1))),
                       out_features=1 if i == (num_blocks - 1) else min(max_features, block_expansion * (2 ** i)),
                       stride=(1, 1) if i > 2 else (2, 2),
                       negative_slope=negative,
                       act=False if i == (num_blocks - 1) else True,
                       norm=False if i == 0 or i == (num_blocks - 1) else True)
                       for i in range(num_blocks)]
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))

        return outs[1:]


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator with PatchGAN.
    """

    def __init__(self, scales=(), **kwargs):
        super(MultiScaleDiscriminator, self).__init__()

        self.scales = scales

        discriminators = {}
        for scale in scales:
            discriminators[str(scale).replace('.', '-')] = PatchDiscriminator(**kwargs)
        self.discs = nn.ModuleDict(discriminators)

    def forward(self, x):
        out_dict = {}
        for scale, discriminator in self.discs.items():
            scale = str(scale).replace('-', '.')
            key = f'prediction_{scale}'
            outs = discriminator(x[key])
            out_dict[f'feature_maps_{scale}'] = outs[:-1]
            out_dict[f'prediction_map_{scale}'] = outs[-1]

        return out_dict
