'''ShuffleNet in PyTorch.

See the paper "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from starship.umtf.common.model import BACKBONES


def conv_bn(inp, oup, kernel=3, stride=1, padding=0, dilation=1, no_relu=False):
    ret = [
        nn.Conv3d(inp, oup, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm3d(oup)]
    if not no_relu:
        ret.append(nn.ReLU(inplace=True))
    return nn.Sequential(*ret
                         )


def channel_shuffle(x, groups):
    '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
    batchsize, num_channels, depth, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, depth, height, width)
    # permute
    x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
    # flatten
    x = x.view(batchsize, num_channels, depth, height, width)
    return x


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, dilation, groups, padding=1):
        super(Bottleneck, self).__init__()
        self.dilation = dilation
        self.groups = groups
        mid_planes = out_planes // 4
        self.padding = padding
        if self.dilation >= 2:
            out_planes = out_planes - in_planes
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv3d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=1, padding=self.padding,
                               dilation=self.dilation, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        if self.dilation >= 2 or self.padding == 0:
            self.shortcut = conv_bn(in_planes, in_planes, padding=self.padding, dilation=self.dilation, no_relu=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = channel_shuffle(out, self.groups)
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))

        if self.dilation >= 2 or self.padding == 0:
            out = self.relu(torch.cat([out, self.shortcut(x)], 1))
        else:
            out = self.relu(out + x)

        return out


@BACKBONES.register_module
class ShuffleNet3D(nn.Module):
    def __init__(self,
                 groups=3,
                 width_mult=1,
                 d_count=1000, c_count=1):
        super(ShuffleNet3D, self).__init__()
        self.groups = groups
        num_blocks = [4, 8, 4]

        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if groups == 1:
            out_planes = [24, 144, 288, 567]
        elif groups == 2:
            out_planes = [24, 200, 400, 800]
        elif groups == 3:
            out_planes = [24, 240, 480, 960]
        elif groups == 4:
            out_planes = [24, 272, 544, 1088]
        elif groups == 8:
            out_planes = [24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(groups))
        out_planes = [int(i * width_mult) for i in out_planes]
        self.in_planes = out_planes[0]
        self.conv1 = conv_bn(3, self.in_planes)
        self.layer1 = self._make_layer(out_planes[1], num_blocks[0], self.groups, 2)
        self.layer2 = self._make_layer(out_planes[2], num_blocks[1], self.groups, 2)
        self.layer3 = self._make_layer(out_planes[3], num_blocks[2], self.groups, 3)

        # building classifier
        self.layerD = nn.Conv3d(in_channels=out_planes[-1], out_channels=d_count, kernel_size=1, stride=1, padding=0)
        self.layer9 = conv_bn(d_count, 256, 1)
        self.layer10 = conv_bn(256, 128, 1)
        self.layerC = nn.Conv3d(in_channels=128, out_channels=c_count, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, out_planes, num_blocks, groups, dilation):
        layers = []
        for i in range(num_blocks):
            dil = dilation if i == 0 else 1
            pad = 0 if i == 0 else 1
            layers.append(Bottleneck(self.in_planes, out_planes, dilation=dil, groups=groups, padding=pad))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, img0, img1, img2):
        x = torch.cat([img0, img1, img2], dim=1)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        feature = out[:, :, 1:2, 1:2, 1:2]
        d_predict = self.layerD(feature)
        feature = self.layer9(d_predict)
        feature = self.layer10(feature)
        c_predict = self.layerC(feature)
        return d_predict.reshape((d_predict.size(0), -1)), c_predict.reshape((c_predict.size(0), -1))
