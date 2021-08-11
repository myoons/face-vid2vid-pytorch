import torch
from torch.autograd import grad

import torch.nn as nn
import torch.nn.functional as F
from modules.blocks import AntiAliasInterpolation2d


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform keypoints into gaussian like representation
    """
    mean = kp['keypoints']
    number_of_leading_dimensions = len(mean.shape) - 1

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    coordinate_grid = coordinate_grid.to(mean.device)

    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 3)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    return out


def make_coordinate_grid(spatial_size, dtype):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    d, h, w = spatial_size
    z = torch.arange(d).type(dtype)
    x = torch.arange(w).type(dtype)
    y = torch.arange(h).type(dtype)

    z = (2 * (z / (d - 1)) - 1)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    zz = z.view(-1, 1, 1).repeat(1, w, h)
    xx = x.view(1, -1, 1).repeat(d, 1, h)
    yy = y.view(1, 1, -1).repeat(d, w, 1)

    meshed = torch.cat([zz.unsqueeze_(3), xx.unsqueeze_(3), yy.unsqueeze_(3)], dim=3)
    return meshed


def make_coordinate_grid_2d(spatial_size, dtype):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(dtype)
    y = torch.arange(h).type(dtype)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    xx = x.view(1, -1).repeat(h, 1)
    yy = y.view(-1, 1).repeat(1, w)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], dim=2)
    return meshed


class Transform:
    """
    Random tps transformation for equivariance constraints.
    """
    def __init__(self, batch_size, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([batch_size, 2, 3]))
        self.theta = torch.eye(2, 3).view(1, 2, 3) + noise
        self.batch_size = batch_size

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid_2d((kwargs['points_tps'], kwargs['points_tps']), dtype=noise.dtype)
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0, std=kwargs['sigma_tps'] * torch.ones([batch_size, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid_2d(frame.shape[2:], dtype=frame.dtype).unsqueeze(0).to(frame.device)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.batch_size, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    # TODO() : 학습 완료후 시각화 필요
    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.batch_size, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


class ImagePyramid(nn.Module):
    """
    Create multi-scale images for multi-scale perceptual loss.
    """

    def __init__(self, scales, num_channels):
        super(ImagePyramid, self).__init__()

        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict
