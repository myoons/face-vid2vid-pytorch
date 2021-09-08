""" Head pose & Expression Estimator module which estimates the rotation angles, translation vector, expression deformations from images """

import torch.nn as nn
import torch.nn.functional as F
from modules.blocks import ResBottleneck
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


class HeadExpressionEstimator(nn.Module):

    def __init__(self, depth, num_kp, num_channels, block_expansion, num_layers):
        super(HeadExpressionEstimator, self).__init__()

        self.depth = depth
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=block_expansion, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.norm1 = BatchNorm2d(block_expansion)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=block_expansion, out_channels=block_expansion * 4, kernel_size=1)
        self.norm2 = BatchNorm2d(256, affine=True)

        self.block1 = nn.Sequential()
        for i in range(3):
            self.block1.add_module(f'block1_{i+1}', ResBottleneck(in_features=block_expansion * 4, down_sample=False))

        self.conv3 = nn.Conv2d(in_channels=block_expansion * 4, out_channels=block_expansion * 8, kernel_size=(1, 1))
        self.norm3 = BatchNorm2d(block_expansion * 8, affine=True)
        self.block2 = ResBottleneck(in_features=block_expansion * 8, down_sample=True)

        self.block3 = nn.Sequential()
        for i in range(3):
            self.block3.add_module(f'block3_{i + 1}', ResBottleneck(in_features=block_expansion * 8, down_sample=False))

        self.conv4 = nn.Conv2d(in_channels=block_expansion * 8, out_channels=block_expansion * 16, kernel_size=1)
        self.norm4 = BatchNorm2d(block_expansion * 16, affine=True)
        self.block4 = ResBottleneck(in_features=block_expansion * 16, down_sample=True)

        self.block5 = nn.Sequential()
        for i in range(5):
            self.block5.add_module(f'block5_{i + 1}', ResBottleneck(in_features=block_expansion * 16, down_sample=False))

        self.conv5 = nn.Conv2d(in_channels=block_expansion * 16, out_channels=block_expansion * 32, kernel_size=1)
        self.norm5 = BatchNorm2d(block_expansion * 32, affine=True)
        self.block6 = ResBottleneck(in_features=block_expansion * 32, down_sample=True)

        self.block7 = nn.Sequential()
        for i in range(2):
            self.block7.add_module(f'block7_{i + 1}', ResBottleneck(in_features=block_expansion * 32, down_sample=False))

        # Just using pre-trained hopenet
        # self.fc_yaw = nn.Linear(block_expansion * 32, 66)
        # self.fc_pitch = nn.Linear(block_expansion * 32, 66)
        # self.fc_roll = nn.Linear(block_expansion * 32, 66)

        self.fc_translation = nn.Linear(block_expansion * 32, 3)
        self.fc_deformation = nn.Linear(block_expansion * 32, num_kp * 3)
        self.num_kp = num_kp

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.max_pool(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)

        out = self.block1(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = F.relu(out)
        out = self.block2(out)

        out = self.block3(out)

        out = self.conv4(out)
        out = self.norm4(out)
        out = F.relu(out)
        out = self.block4(out)

        out = self.block5(out)

        out = self.conv5(out)
        out = self.norm5(out)
        out = F.relu(out)
        out = self.block6(out)

        out = self.block7(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.shape[0], -1)

        # Just using pre-trained hopenet
        # yaw, pitch, roll = self.fc_yaw(out), self.fc_pitch(out), self.fc_roll(out)
        translation = self.fc_translation(out)
        deformation = self.fc_deformation(out)

        translation = translation.unsqueeze(-2).repeat(1, self.num_kp, 1)
        deformation = deformation.view(-1, self.num_kp, 3)

        # return {'yaw': yaw, 'pitch': pitch, 'roll': roll, 'translation': translation, 'deformation': deformation}
        return {'translation': translation, 'deformation': deformation}
