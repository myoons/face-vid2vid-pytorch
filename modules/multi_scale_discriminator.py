import torch.nn as nn


class DownBlock2d(nn.Module):

    def __init__(self, in_features, out_features, stride=(2, 2), kernel_size=(4, 4),
                 padding=(1, 1), negative_slope=0.2, act=False, norm=False):
        super(DownBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)

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

    def __init__(self, in_channels, block_expansion, num_blocks, max_features, negative):
        super(PatchDiscriminator, self).__init__()

        down_blocks = [
            DownBlock2d(in_features=in_channels if i == 0 else min(max_features, block_expansion * (2 ** (i - 1))),
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

    def __init__(self, depth, num_kp, num_channels, scales, block_expansion, num_blocks, max_features, negative):
        super(MultiScaleDiscriminator, self).__init__()

        self.depth = depth
        self.num_kp = num_kp
        self.scales = scales

        discriminators = {}
        for scale in scales:
            discriminators[str(scale).replace('.', '-')] = PatchDiscriminator(in_channels=num_channels,
                                                                              block_expansion=block_expansion,
                                                                              num_blocks=num_blocks,
                                                                              max_features=max_features,
                                                                              negative=negative)
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


if __name__ == '__main__':
    import yaml, cv2, torch

    with open('../config/vox.yaml') as f:
        config = yaml.load(f)

    image = cv2.imread('../data/vox/id00017/iJOjkDq6rW8/0001.jpg')
    image = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2).float()

    d = MultiScaleDiscriminator(**config['model_params']['common_params'],
                                **config['model_params']['multi_scale_discriminator'])

    from modules.util import ImagePyramid

    ip = ImagePyramid(config['model_params']['multi_scale_discriminator']['scales'], 3)
    inputs = ip(image)
    out = d(inputs)
    print('A')