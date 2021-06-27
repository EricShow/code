import torch
from .sk_resnext3d import SkResNeXt3D
import torch.nn as nn
import torch
import torch.nn.functional as F

from starship.umtf.common.model import BACKBONES
from .center_sensitive_network import conv_block
import math
from .cbam import CBAM

class Bottleneck(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, padding=0, dilation=1, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(Bottleneck, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv3d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(D * C)
        self.conv2 = nn.Conv3d(D * C, D * C, kernel_size=3, padding=padding, dilation=dilation, groups=C, bias=False)
        self.bn2 = nn.BatchNorm3d(D * C)
        self.conv3 = nn.Conv3d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

        #self.cbam = CBAM(planes * 4)
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        #out = self.cbam(out);
        out += residual
        out = self.relu(out)

        return out

@BACKBONES.register_module
class ResNext3D_lstm(torch.nn.Module):
    def __init__(self, requires_grad=False, in_count=3, layers=[3, 4, 6, 3], baseWidth=4, cardinality=32, d_count=1000, c_count=1):
        super(ResNext3D_lstm, self).__init__()
        # vgg_pretrained_features = models.vgg19(pretrained=True).features
        block = Bottleneck

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.inplanes = 64
        self.output_size = 64

        self.conv1 = nn.Conv3d(in_count, 64, 3, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], dilation=2, padding=0)
        self.layer2 = self._make_layer(block, 128, layers[1], dilation=2, padding=0)
        self.layer3 = self._make_layer(block, 256, layers[2], dilation=3, padding=0)
        #self.layer4 = self._make_layer(block, 512, layers[3], dilation=2, padding=0)
        
        self.layerD = nn.Conv3d(in_channels=1024, out_channels=d_count, kernel_size=1, stride=1, padding=0)
        self.layer9 = conv_block(d_count, 256, 1, stride=1, p_size=0)
        self.layer10 = conv_block(256, 128, 1, stride=1, p_size=0)
        
        self.conv_i = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=128, kernel_size = 1, stride = 1, padding = 0),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=128, kernel_size = 1, stride = 1, padding = 0),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=128, kernel_size = 1, stride = 1, padding = 0),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=128, kernel_size = 1, stride = 1, padding = 0),
            nn.Sigmoid()
        )
        self.layerfp = nn.Conv3d(in_channels=128, out_channels=c_count, kernel_size=1, stride=1, padding=0)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.pow(2. / n, 1 / 3.0))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, dilation=1, padding=0):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNext
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = nn.Sequential(
            nn.Conv3d(self.inplanes, planes * block.expansion,
                      kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm3d(planes * block.expansion),
        )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, padding=padding, dilation=dilation,
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, padding=1))

        return nn.Sequential(*layers)

    def forward(self, img0, img1, img2, h, c):
        #print("forward")
        #print(h)
        #print(h.shape)
        with torch.no_grad():
            x = torch.cat([img0, img1, img2], dim=1)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x) #output [32 512 3 3 3]
            #print(x.shape)
            x = self.layer3(x) #output [32 1024 3 3 3]
            #print(x.shape)
            
            #print(h.shape)
            #print(h.shape)
            #print(x.shape)
            #B,C,H,W,D = x.shape;
            
            #print(B)
            x = x[:, :, 1:2, 1:2, 1:2]
            #x = self.layer4(x)
            d_predict = self.layerD(x)
            #print(d_predict.shape)
            x = self.layer9(d_predict)
            x = self.layer10(x)
        #print(x.shape)
        x = torch.cat((x, h), 1)
        #print(x.shape)
        i = self.conv_i(x)
        f = self.conv_f(x)
        g = self.conv_g(x)
        o = self.conv_o(x)
        c = f * c + i * g
        h = o * torch.tanh(c)
        #print(h.shape)
        #print(x.shape)
        fp_predict = self.layerfp(h)
        #print(c_predict.shape)
        #d_predict = d_predict.reshape((d_predict.size(0), -1))
        fp_predict = fp_predict.reshape((fp_predict.size(0), -1))
        return fp_predict, h , c