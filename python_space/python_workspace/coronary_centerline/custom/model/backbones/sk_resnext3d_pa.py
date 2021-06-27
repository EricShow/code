#import torch.nn as nn
#import torch
#import torch.nn.functional as F
#
#from starship.umtf.common.model import BACKBONES
#from .center_sensitive_network import conv_block
#import math
#
#
#class Bottleneck(nn.Module):
#    """
#    RexNeXt bottleneck type C
#    """
#    expansion = 4
#
#    def __init__(self, inplanes, planes, baseWidth, cardinality, padding=0, dilation=1, downsample=None):
#        """ Constructor
#        Args:
#            inplanes: input channel dimensionality
#            planes: output channel dimensionality
#            baseWidth: base width.
#            cardinality: num of convolution groups.
#            stride: conv stride. Replaces pooling layer.
#        """
#        super(Bottleneck, self).__init__()
#
#        D = int(math.floor(planes * (baseWidth / 64)))
#        C = cardinality
#
#        self.conv1 = nn.Conv3d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
#        self.bn1 = nn.BatchNorm3d(D * C)
#        self.conv2 = nn.Conv3d(D * C, D * C, kernel_size=3, padding=padding, dilation=dilation, groups=C, bias=False)
#        self.bn2 = nn.BatchNorm3d(D * C)
#        self.conv3 = nn.Conv3d(D * C, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
#        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
#        self.relu = nn.ReLU(inplace=True)
#
#        self.downsample = downsample
#
#    def forward(self, x):
#        residual = x
#
#        out = self.conv1(x)
#        out = self.bn1(out)
#        out = self.relu(out)
#
#        out = self.conv2(out)
#        out = self.bn2(out)
#        out = self.relu(out)
#
#        out = self.conv3(out)
#        out = self.bn3(out)
#
#        if self.downsample is not None:
#            residual = self.downsample(x)
#
#        out += residual
#        out = self.relu(out)
#
#        return out
#
#
#@BACKBONES.register_module
#class SkResNeXt3D(nn.Module):
#    def __init__(self, in_count=6, layers=[1, 2, 1, 1], baseWidth=4, cardinality=32, d_count=1000, c_count=1):
#        """ Constructor
#        Args:
#            baseWidth: baseWidth for ResNeXt.
#            cardinality: number of convolution groups.
#            layers: config of layers, e.g., [3, 4, 6, 3]
#            num_classes: number of classes
#        """
#        super(SkResNeXt3D, self).__init__()
#        block = Bottleneck
#
#        self.cardinality = cardinality
#        self.baseWidth = baseWidth
#        self.inplanes = 64
#        self.output_size = 64
#
#        self.conv1 = nn.Conv3d(in_count, 64, 3, 1, 0, bias=False)
#        self.bn1 = nn.BatchNorm3d(64)
#        self.relu = nn.ReLU(inplace=True)
#        self.layer1 = self._make_layer(block, 64, layers[0], dilation=1, padding=0)
#        self.layer2 = self._make_layer(block, 128, layers[1], dilation=2, padding=0)
#        self.layer3 = self._make_layer(block, 256, layers[2], dilation=4, padding=0)
#        self.layer4 = conv_block(1024, 2048, 1, stride=1, p_size=0)
#        #self.layer4 = self._make_layer(block, 512, layers[3], dilation=1, padding=0)
#        #self.layer8 = conv_block(256 * block.expansion, 2048, 1, stride=1, p_size=0)
#        self.layerD = nn.Conv3d(in_channels=2048, out_channels=d_count, kernel_size=3, stride=1, padding=0)
#        self.layer9 = conv_block(d_count, 256, 1, stride=1, p_size=0)
#        self.layer10 = conv_block(256, 128, 1, stride=1, p_size=0)
#        self.layerC = nn.Conv3d(in_channels=128, out_channels=c_count, kernel_size=1, stride=1, padding=0)
#
#        for m in self.modules():
#            if isinstance(m, nn.Conv3d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
#                m.weight.data.normal_(0, math.pow(2. / n, 1 / 3.0))
#            elif isinstance(m, nn.BatchNorm3d):
#                m.weight.data.fill_(1)
#                m.bias.data.zero_()
#
#    def _make_layer(self, block, planes, blocks, dilation=1, padding=0):
#        """ Stack n bottleneck modules where n is inferred from the depth of the network.
#        Args:
#            block: block type used to construct ResNext
#            planes: number of output channels (need to multiply by block.expansion)
#            blocks: number of blocks to be built
#            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
#        Returns: a Module consisting of n sequential bottlenecks.
#        """
#        downsample = nn.Sequential(
#            nn.Conv3d(self.inplanes, planes * block.expansion,
#                      kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=False),
#            nn.BatchNorm3d(planes * block.expansion),
#        )
#
#        layers = []
#        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, padding=padding, dilation=dilation,
#                            downsample=downsample))
#        self.inplanes = planes * block.expansion
#        for i in range(1, blocks):
#            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, padding=1))
#
#        return nn.Sequential(*layers)
#
#    def forward(self, img0, img1, img2):
#        with torch.no_grad():
#            device = img0.device
#            d, h, w = img0.size()[2:]
#            grid = torch.meshgrid([torch.arange(d, device=device)/float(d),torch.arange(h, device=device)/float(h),torch.arange(w, device=device)/float(w)])
#            grid = [g[None] for g in grid]
#            grid = torch.cat(grid, dim = 0)
#            grid = grid[None]
#            grid = grid.expand(img0.size(0),-1,-1,-1,-1)
#            x = torch.cat([img0, img1, img2, grid], dim=1)
#        x = self.conv1(x)
#        x = self.bn1(x)
#        x = self.relu(x)
#        x = self.layer1(x)
#        x = self.layer2(x)
#        x = self.layer3(x)
#        x = self.layer4(x)
#        #x = x[:,:,1:2,1:2,1:2]
#        #x = self.layer8(x)
#        d_predict = self.layerD(x)
#        x = self.layer9(d_predict)
#        x = self.layer10(x)
#        c_predict = self.layerC(x)
#        return d_predict.reshape((d_predict.size(0), -1)), c_predict.reshape((c_predict.size(0), -1))
