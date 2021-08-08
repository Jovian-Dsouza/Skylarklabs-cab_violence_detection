import torch
from torch import nn
from torchsummary import summary

class SeperableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(SeperableConv3D, self).__init__()

        self.depthwise = nn.Conv3d(in_channels, in_channels, groups = in_channels, kernel_size = kernel_size, **kwargs)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, bias = False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x

class Conv3DSimple(SeperableConv3D):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)

class Conv2Plus1D(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1D, self).__init__(
            SeperableConv3D(in_planes, midplanes, kernel_size = (1, 3, 3), stride = (1, stride, stride), padding = (0, padding, padding), bias = False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            SeperableConv3D(midplanes, out_planes, kernel_size = (3, 1, 1), stride = (stride, 1, 1), padding = (padding, 0, 0), bias = False))

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv3DNoTemporal(SeperableConv3D):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):

        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        self.conv1 = nn.Sequential(
            SeperableConv3D(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            SeperableConv3D(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out