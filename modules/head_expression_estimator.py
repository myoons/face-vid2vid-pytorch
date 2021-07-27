""" Head pose & Expression Estimator module which estimates the rotation angles, translation vector, expression deformations from images """

import torch.nn as nn
from modules.util import ResBottleneck
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


class HeadExpressionEstimator(nn.Module):

    def __init__(self, block_expansion=256, num_bins=66, num_layers=(3, 3, 5, 2), num_kp=20, **kwargs):
        super(HeadExpressionEstimator, self).__init__()

        self.first = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self.make_layer(64, block_expansion, num_layers[0], down_sample=True)
        self.layer2 = self.make_layer(block_expansion * 2, block_expansion * 2, num_layers[1], down_sample=True)
        self.layer3 = self.make_layer(block_expansion * 4, block_expansion * 4, num_layers[2], down_sample=True)
        self.layer4 = self.make_layer(block_expansion * 8, block_expansion * 8, num_layers[3])

        self.avg_pool = nn.AvgPool2d(7)

        self.fc_yaw = nn.Linear(block_expansion * 8, num_bins)
        self.fc_pitch = nn.Linear(block_expansion * 8, num_bins)
        self.fc_roll = nn.Linear(block_expansion * 8, num_bins)

        self.fc_translation = nn.Linear(block_expansion * 8, 3)
        self.fc_deformation = nn.Linear(block_expansion * 8, num_kp * 3)
        self.num_kp = num_kp

    @staticmethod
    def make_layer(in_feature, out_feature, num_blocks, down_sample=False):
        layer = [ResBottleneck(in_features=(in_feature if i == 0 else out_feature), out_features=out_feature)
                 for i in range(num_blocks)]
        if down_sample:
            layer.append(ResBottleneck(in_features=out_feature, out_features=out_feature * 2, down_sample=True))
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.first(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(x.size(0), -1)

        yaw, pitch, roll = self.fc_yaw(out), self.fc_pitch(out), self.fc_roll(out)
        translation = self.fc_translation(out)
        deformation = self.fc_deformation(out)

        translation = translation.unsqueeze(-2).repeat(1, self.num_kp, 1)
        deformation = deformation.view(-1, self.num_kp, 3)

        return (yaw, pitch, roll), translation, deformation
