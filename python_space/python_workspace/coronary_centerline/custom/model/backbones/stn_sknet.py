import torch.nn as nn
import torch
import torch.nn.functional as F

from starship.umtf.common.model import BACKBONES


class conv_block(nn.Module):
    def __init__(self, chann_in, chann_out, k_size, stride, p_size, dilation=1, no_bn=False):
        super(conv_block, self).__init__()
        if no_bn:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels=chann_in, out_channels=chann_out, kernel_size=k_size, stride=stride,
                          padding=p_size,
                          dilation=dilation),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels=chann_in, out_channels=chann_out, kernel_size=k_size, stride=stride,
                          padding=p_size,
                          dilation=dilation),
                nn.BatchNorm3d(chann_out),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class SKBranch(nn.Module):
    def __init__(self, d_out):
        super(SKBranch, self).__init__()

        self.layer1 = conv_block(1, 32, 3, stride=1, p_size=0)
        self.layer2 = conv_block(32, 32, 3, stride=1, p_size=0)
        self.layer3 = conv_block(32, 32, 3, stride=1, p_size=0, dilation=2)
        self.layer4 = conv_block(32, 32, 3, stride=1, p_size=0, dilation=4)
        self.layer5 = conv_block(32, 64, 3, stride=1, p_size=0)
        self.layer6 = conv_block(64, 64, 1, stride=1, p_size=0)
        self.layer7 = conv_block(64, d_out, 1, stride=1, p_size=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)

        return out


class SKNetLoc(nn.Module):
    def __init__(self, d_count):
        super(SKNetLoc, self).__init__()
        branch_d_out = 64
        self.branch0 = SKBranch(branch_d_out)
        self.branch1 = SKBranch(branch_d_out)
        self.branch2 = SKBranch(branch_d_out)
        self.layer8 = conv_block(3 * branch_d_out, 1024, 1, stride=1, p_size=0)
        self.layer8_extra = conv_block(1024, 1024, 1, stride=1, p_size=0)
        self.layerD = nn.Conv3d(in_channels=1024, out_channels=d_count, kernel_size=1, stride=1, padding=0)

    def forward(self, img0, img1, img2):
        feature0 = self.branch0(img0)
        feature1 = self.branch1(img1)
        feature2 = self.branch2(img2)
        feature = torch.cat([feature0, feature1, feature2], dim=1)
        feature = self.layer8(feature)
        feature = F.dropout(feature, p=0.5)
        feature = self.layer8_extra(feature)
        d_predict = self.layerD(feature)
        return d_predict.reshape((d_predict.size(0), -1))


class SKNetDir(nn.Module):
    def __init__(self, d_count, c_count=1):
        super(SKNetDir, self).__init__()
        branch_d_out = 64
        self.branch0 = SKBranch(branch_d_out)
        self.branch1 = SKBranch(branch_d_out)
        self.branch2 = SKBranch(branch_d_out)
        self.layer8 = conv_block(3 * branch_d_out, 2 * d_count, 1, stride=1, p_size=0)
        self.layer8_extra = conv_block(2 * d_count, 2 * d_count, 1, stride=1, p_size=0)
        self.layerD = nn.Conv3d(in_channels=2 * d_count, out_channels=d_count, kernel_size=1, stride=1, padding=0)
        self.layer9 = conv_block(d_count, 256, 1, stride=1, p_size=0)
        self.layer10 = conv_block(256, 128, 1, stride=1, p_size=0)
        self.layerC = nn.Conv3d(in_channels=128, out_channels=c_count, kernel_size=1, stride=1, padding=0)

    def forward(self, img0, img1, img2):
        feature0 = self.branch0(img0)
        feature1 = self.branch1(img1)
        feature2 = self.branch2(img2)
        feature = torch.cat([feature0, feature1, feature2], dim=1)
        feature = self.layer8(feature)
        feature = F.dropout(feature, p=0.5)
        feature = self.layer8_extra(feature)
        d_predict = self.layerD(feature)
        feature = self.layer9(d_predict)
        feature = self.layer10(feature)
        c_predict = self.layerC(feature)
        return d_predict.reshape((d_predict.size(0), -1)), c_predict.reshape((c_predict.size(0), -1)),


@BACKBONES.register_module
class StnSKNet(nn.Module):
    def __init__(self, d_count, c_count=1):
        super(StnSKNet, self).__init__()
        self.stn = SKNetLoc(d_count=3 * 3 * 4)
        self.dir_net = SKNetDir(d_count, c_count)

    def forward(self, img0, img1, img2):
        img = [img0, img1, img2]
        theta = self.stn(*img)
        theta = theta.reshape(theta.size(0), 3, 3, 4)
        theta = [theta[:, 0], theta[:, 1], theta[:, 2]]
        grid = [F.affine_grid(t, im.size()) for t, im in zip(theta, img)]
        img = [F.grid_sample(im, g, align_corners=True) for im, g in zip(img, grid)]
        ret = self.dir_net(*img)
        return ret
