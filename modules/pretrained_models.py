import torch
import torchvision
import torch.nn as nn
from torchvision import models

import math
import numpy as np


class Hopenet(nn.Module):
    """
    Hope network for head pose loss
    """

    def __init__(self, requires_grad=False, weight_dir='pretrained/hopenet.pkl'):
        super(Hopenet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(torchvision.models.resnet.Bottleneck, 64, 3)
        self.layer2 = self._make_layer(torchvision.models.resnet.Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(torchvision.models.resnet.Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(torchvision.models.resnet.Bottleneck, 512, 3, stride=2)

        self.avgpool = nn.AvgPool2d(7)

        self.fc_yaw = nn.Linear(512 * torchvision.models.resnet.Bottleneck.expansion, 66)
        self.fc_pitch = nn.Linear(512 * torchvision.models.resnet.Bottleneck.expansion, 66)
        self.fc_roll = nn.Linear(512 * torchvision.models.resnet.Bottleneck.expansion, 66)

        # Load pretrained weight
        weight = torch.load(weight_dir) if torch.cuda.is_available() else torch.load(weight_dir, map_location='cpu')
        self.load_state_dict(weight, strict=False)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        idx_tensor = torch.arange(66, dtype=torch.float32)

        return pre_yaw, pre_pitch, pre_roll, idx_tensor


class Vgg19(nn.Module):
    """
    Vgg19 network for perceptual loss
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()

        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
