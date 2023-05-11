'''VGG11/13/16/19 in Pytorch.'''
import sys
sys.path.append('./model/')
import torch
import torch.nn as nn
import os
import numpy as np
from scipy.io import loadmat
import torch.nn.functional as F


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class Probe(nn.Module):
    def __init__(self, in_ch, layer_num=2, num_class=10):
        super(Probe, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
        )
        if layer_num == 2:
            self.convs = nn.Sequential(
                nn.Conv2d(in_ch, in_ch * 2, 3, 2, 1),
                nn.Conv2d(in_ch * 2, in_ch * 2, 3, 2, 1),
                nn.BatchNorm2d(in_ch * 2),
                nn.ReLU(),
                nn.Conv2d(in_ch * 2, in_ch * 4, 3, 1, 1),
                nn.Conv2d(in_ch * 4, in_ch * 4, 3, 1, 1),
                nn.BatchNorm2d(in_ch * 4),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.fc = nn.Linear(in_ch * 4, num_class)
        elif layer_num == 1:
            self.convs = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, 1, 1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                nn.Conv2d(in_ch, in_ch, 3, 1, 1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.fc = nn.Linear(in_ch, num_class)

    def forward(self, x):
        feat = self.features(x)
        feat = self.convs(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        return out


class CNN_MNIST(nn.Module):
    def __init__(self, num_channels=3):
        super(CNN_MNIST, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.probe1 = Probe(16, layer_num=1)
        self.probe2 = Probe(32, layer_num=1)
        self.adp = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(32, 10)

    def forward(self, x, probe=False):
        if not probe:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.adp(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
        else:
            x = self.layer1(x)
            out1 = self.probe1(x)
            x = self.layer2(x)
            out2 = self.probe2(x)
            x = self.adp(x)
            x = x.view(x.size(0), -1)
            out3 = self.classifier(x)
            return out1, out2, out3


class CNN_6(nn.Module):
    def __init__(self, num_channels=3):
        super(CNN_6, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=1, stride=1)
        )
        self.probe1 = Probe(64)
        self.probe2 = Probe(256)

        self.classifier = nn.Linear(65536, 10)

    def forward(self, x, probe=False):
        if not probe:
            x = self.layers(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
        else:
            x1 = self.layers[:7](x)
            x2 = self.layers[:13](x)
            out1 = self.probe1(x1)
            out2 = self.probe2(x2)

            f3 = self.layers(x)
            f3 = f3.view(x.size(0), -1)
            out3 = self.classifier(f3)
            return out1, out2, out3


class Flatten(nn.Module):
    # 构造函数，没有什么要做的
    def __init__(self):
        # 调用父类构造函数
        super(Flatten, self).__init__()

    # 实现forward函数
    def forward(self, input):
        return input.view(input.size(0), -1)


class VGG(nn.Module):
    def __init__(self, vgg_name, num_class=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_class)

    def forward(self, x, mid_input=False):
        if not mid_input:
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out
        else:
            for i in range(len(self.features)):
                if i < mid_input:
                    continue
                else:
                    x = self.features[i](x)
            x = x.view(x.size(0), -1)
            out = self.classifier(x)
            return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG13_2(nn.Module):  # for ai lancet
    def __init__(self, vgg_name):
        super(VGG13_2, self).__init__()
        self.in_channels = 3
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:3])
        self.features2 = self._make_layers(cfg[vgg_name][3:6])
        self.features3 = self._make_layers(cfg[vgg_name][6:9])
        self.features4 = self._make_layers(cfg[vgg_name][9:12])
        self.features5 = self._make_layers(cfg[vgg_name][12:])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x, probe=False):
        f1 = self.features1(x)
        f2 = self.features2(f1)
        f3 = self.features3(f2)
        f4 = self.features4(f3)
        f5 = self.features5(f4)
        out = f5.view(f5.size(0), -1)
        out = self.classifier(out)
        return out
    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.in_channels = 3
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.adv = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        f1 = self.features1(x)
        f = self.adv(f1).view(f1.size(0), -1)
        out = self.classifier(f)
        return out


class VGG13DensMask(nn.Module):  # for robust erpair
    def __init__(self, vgg_name):
        super(VGG13DensMask, self).__init__()
        self.in_channels = 3
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:3])
        self.features2 = self._make_layers(cfg[vgg_name][3:6])
        self.features3 = self._make_layers(cfg[vgg_name][6:9])
        self.features4 = self._make_layers(cfg[vgg_name][9:12])
        self.features5 = self._make_layers(cfg[vgg_name][12:])
        self.dense1 = nn.Linear(512, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 10)
    def forward(self, x, mid_input=None, mid_idx=None):
        if mid_idx == 1:
            f1 = self.features1(x)
            f1 = mid_input * f1
            f2 = self.features2(f1)
            f3 = self.features3(f2)
            f4 = self.features4(f3)
            f5 = self.features5(f4)
            f5 = f5.view(f5.size(0), -1)
            d1 = F.relu(self.dense1(f5))
            d2 = F.relu(self.dense2(d1))
            out = d2.view(f5.size(0), -1)
            out = self.classifier(out)
        elif mid_idx == 2:
            f1 = self.features1(x)
            f2 = self.features2(f1)
            f2 = mid_input * f2
            f3 = self.features3(f2)
            f4 = self.features4(f3)
            f5 = self.features5(f4)
            f5 = f5.view(f5.size(0), -1)
            d1 = F.relu(self.dense1(f5))
            d2 = F.relu(self.dense2(d1))
            out = d2.view(f5.size(0), -1)
            out = self.classifier(out)
        elif mid_idx == 3:
            f1 = self.features1(x)
            f2 = self.features2(f1)
            f3 = self.features3(f2)
            f3 = mid_input * f3
            f4 = self.features4(f3)
            f5 = self.features5(f4)
            f5 = f5.view(f5.size(0), -1)
            d1 = F.relu(self.dense1(f5))
            d2 = F.relu(self.dense2(d1))
            out = d2.view(f5.size(0), -1)
            out = self.classifier(out)
        elif mid_idx == 4:
            f1 = self.features1(x)
            f2 = self.features2(f1)
            f3 = self.features3(f2)
            f4 = self.features4(f3)
            f4 = mid_input * f4
            f5 = self.features5(f4)
            f5 = f5.view(f5.size(0), -1)
            d1 = F.relu(self.dense1(f5))
            d2 = F.relu(self.dense2(d1))
            out = d2.view(f5.size(0), -1)
            out = self.classifier(out)
        elif mid_idx == 5:
            f1 = self.features1(x)
            f2 = self.features2(f1)
            f3 = self.features3(f2)
            f4 = self.features4(f3)
            f5 = self.features5(f4)
            f5 = mid_input * f5
            f5 = f5.view(f5.size(0), -1)
            d1 = F.relu(self.dense1(f5))
            d2 = F.relu(self.dense2(d1))
            out = d2.view(f5.size(0), -1)
            out = self.classifier(out)
        elif mid_idx == 6:
            f1 = self.features1(x)
            f2 = self.features2(f1)
            f3 = self.features3(f2)
            f4 = self.features4(f3)
            f5 = self.features5(f4)
            f5 = f5.view(f5.size(0), -1)
            d1 = F.relu(self.dense1(f5))
            d1 = d1 * mid_input
            d2 = F.relu(self.dense2(d1))
            out = d2.view(f5.size(0), -1)
            out = self.classifier(out)
        elif mid_idx == 7:
            f1 = self.features1(x)
            f2 = self.features2(f1)
            f3 = self.features3(f2)
            f4 = self.features4(f3)
            f5 = self.features5(f4)
            f5 = f5.view(f5.size(0), -1)
            d1 = F.relu(self.dense1(f5))
            d2 = F.relu(self.dense2(d1))
            d2 = mid_input * d2
            out = d2.view(f5.size(0), -1)
            out = self.classifier(out)
        else:
            f1 = self.features1(x)
            f2 = self.features2(f1)
            f3 = self.features3(f2)
            f4 = self.features4(f3)
            f5 = self.features5(f4)
            f5 = f5.view(f5.size(0), -1)
            d1 = F.relu(self.dense1(f5))
            d2 = F.relu(self.dense2(d1))
            out = d2.view(f5.size(0), -1)
            out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG16_dense_no_probe(nn.Module):  # for ai lancet
    def __init__(self, vgg_name='VGG16', num_class=10):
        super(VGG16_dense_no_probe, self).__init__()
        self.in_channels = 3
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:3])
        self.features2 = self._make_layers(cfg[vgg_name][3:6])
        self.features3 = self._make_layers(cfg[vgg_name][6:10])
        self.features4 = self._make_layers(cfg[vgg_name][10:14])
        self.features5 = self._make_layers(cfg[vgg_name][14:])
        self.dense1 = nn.Linear(25088, 4096)
        self.dense2 = nn.Linear(4096, 4096)
        self.classifier = nn.Linear(4096, num_class)
    def forward(self, x, probe=False):
        f1 = self.features1(x)
        f2 = self.features2(f1)
        f3 = self.features3(f2)
        f4 = self.features4(f3)
        f5 = self.features5(f4)
        f5 = f5.view(f5.size(0), -1)
        d1 = F.relu(self.dense1(f5))
        d2 = F.relu(self.dense2(d1))
        out = d2.view(f5.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG16_dense(nn.Module):  # for ai lancet
    def __init__(self, vgg_name='VGG16', num_class=10):
        super(VGG16_dense, self).__init__()
        self.in_channels = 3
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:3])
        self.features2 = self._make_layers(cfg[vgg_name][3:6])
        self.features3 = self._make_layers(cfg[vgg_name][6:10])
        self.features4 = self._make_layers(cfg[vgg_name][10:14])
        self.features5 = self._make_layers(cfg[vgg_name][14:])
        self.dense1 = nn.Linear(25088, 4096)
        self.dense2 = nn.Linear(4096, 4096)
        self.classifier = nn.Linear(4096, num_class)
        self.probe1 = Probe(64, 2, num_class=num_class)
        self.probe2 = Probe(128, 2, num_class=num_class)
        self.probe3 = Probe(256, 1, num_class=num_class)
        self.probe4 = Probe(512, 1, num_class=num_class)
        self.probe5 = nn.Linear(25088, num_class)
        self.probe6 = nn.Linear(4096, num_class)
        self.probe7 = nn.Linear(4096, num_class)
    def forward(self, x, probe=False):
        if probe:
            f1 = self.features1(x)
            p1 = self.probe1(f1)
            f2 = self.features2(f1)
            p2 = self.probe2(f2)
            f3 = self.features3(f2)
            p3 = self.probe3(f3)
            f4 = self.features4(f3)
            p4 = self.probe4(f4)
            f5 = self.features5(f4)
            f5 = f5.view(f5.size(0), -1)
            p5 = self.probe5(f5)
            d1 = F.relu(self.dense1(f5))
            p6 = self.probe6(d1)
            d2 = F.relu(self.dense2(d1))
            p7 = self.probe7(d2)
            out = d2.view(f5.size(0), -1)
            out = self.classifier(out)
            return p1, p2, p3, p4, p5, p6, p7, out
        else:
            f1 = self.features1(x)
            f2 = self.features2(f1)
            f3 = self.features3(f2)
            f4 = self.features4(f3)
            f5 = self.features5(f4)
            f5 = f5.view(f5.size(0), -1)
            d1 = F.relu(self.dense1(f5))
            d2 = F.relu(self.dense2(d1))
            out = d2.view(f5.size(0), -1)
            out = self.classifier(out)
            return out
        # f1_ = self.features1[0:2](x)
        # f1 = self.features1[2:](f1_)
        # f2_ = self.features2[0:2](f1)
        # f2 = self.features2[2:](f2_)
        # f3_ = self.features3[0:3](f2)
        # f3 = self.features3[3:](f3_)
        # f4_ = self.features4[0:3](f3)
        # f4 = self.features4[3:](f4_)
        # f5_ = self.features5[0:3](f4)
        # f5 = self.features5[3:](f5_)
        # f5 = f5.view(f5.size(0), -1)
        # d1 = F.relu(self.dense1(f5))
        # d2 = F.relu(self.dense2(d1))
        # out = d2.view(d2.size(0), -1)
        # out = self.classifier(out)
        # if feature:
        #     return out
        # else:
        #     return [f5, d1, d2, out]


    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG13_dense(nn.Module):  # for ai lancet
    def __init__(self, vgg_name='VGG13', num_class=10):
        super(VGG13_dense, self).__init__()
        self.in_channels = 3
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:3])
        self.features2 = self._make_layers(cfg[vgg_name][3:6])
        self.features3 = self._make_layers(cfg[vgg_name][6:9])
        self.features4 = self._make_layers(cfg[vgg_name][9:12])
        self.features5 = self._make_layers(cfg[vgg_name][12:])
        self.dense1 = nn.Linear(512, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, num_class)
        self.probe1 = Probe(64, 2, num_class=num_class)
        self.probe2 = Probe(128, 2, num_class=num_class)
        self.probe3 = Probe(256, 1, num_class=num_class)
        self.probe4 = Probe(512, 1, num_class=num_class)
        self.probe5 = nn.Linear(512, num_class)
        self.probe6 = nn.Linear(1024, num_class)
        self.probe7 = nn.Linear(1024, num_class)
    def forward(self, x, probe=False):

        if probe:
            f1 = self.features1(x)
            p1 = self.probe1(f1)
            f2 = self.features2(f1)
            p2 = self.probe2(f2)
            f3 = self.features3(f2)
            p3 = self.probe3(f3)
            f4 = self.features4(f3)
            p4 = self.probe4(f4)
            f5 = self.features5(f4)
            f5 = f5.view(f5.size(0), -1)
            p5 = self.probe5(f5)
            d1 = F.relu(self.dense1(f5))
            p6 = self.probe6(d1)
            d2 = F.relu(self.dense2(d1))
            p7 = self.probe7(d2)
            out = d2.view(f5.size(0), -1)
            out = self.classifier(out)
            return p1, p2, p3, p4, p5, p6, p7, out
        else:
            f1 = self.features1(x)
            f2 = self.features2(f1)
            f3 = self.features3(f2)
            f4 = self.features4(f3)
            f5 = self.features5(f4)
            f5 = f5.view(f5.size(0), -1)
            d1 = F.relu(self.dense1(f5))
            d2 = F.relu(self.dense2(d1))
            out = d2.view(f5.size(0), -1)
            out = self.classifier(out)
            return out
    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG13(nn.Module):  # for ai lancet
    def __init__(self, vgg_name):
        super(VGG13, self).__init__()
        self.in_channels = 3
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:3])
        self.features2 = self._make_layers(cfg[vgg_name][3:6])
        self.features3 = self._make_layers(cfg[vgg_name][6:9])
        self.features4 = self._make_layers(cfg[vgg_name][9:12])
        self.features5 = self._make_layers(cfg[vgg_name][12:])
        self.classifier = nn.Linear(512, 10)
        self.probe1 = Probe(64, 2)
        self.probe2 = Probe(128, 2)
        self.probe3 = Probe(256, 1)
        self.probe4 = Probe(512, 1)
        self.probe5 = nn.Linear(512, 10)
    def forward(self, x, probe=False):

        if probe:
            f1 = self.features1(x)
            p1 = self.probe1(f1)
            f2 = self.features2(f1)
            p2 = self.probe2(f2)
            f3 = self.features3(f2)
            p3 = self.probe3(f3)
            f4 = self.features4(f3)
            p4 = self.probe4(f4)
            f5 = self.features5(f4)
            out = f5.view(f5.size(0), -1)
            p5 = self.probe5(out)
            out = self.classifier(out)
            return p1, p2, p3, p4, p5, out
        else:
            f1 = self.features1(x)
            f2 = self.features2(f1)
            f3 = self.features3(f2)
            f4 = self.features4(f3)
            f5 = self.features5(f4)
            out = f5.view(f5.size(0), -1)
            out = self.classifier(out)
            return out
    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


# >>>>>>>>AI Lancet model loading


def get_mask_mine(ratio, seed, layer, device, set, rep_type, arch='vgg13_dense'):
    if '_' in set:
        set = set.split('_')[-1]
    base_dir = f'/public/czh/repair/checkpoints/{set}'

    blocks = ['features1.3', 'features2.3', 'features3.3', 'features4.3', 'features5.3', 'dense1', 'dense2',
              'classifier']
    PROBE_NUM = 7
    neuron_file = os.path.join(base_dir, '%s/seed%s/%s/fault_%s_top%s_%s_%s.npy' % (
    arch, seed, rep_type, blocks[layer], 100, 0.0, 1000))
    if not os.path.exists(neuron_file):
        raise ValueError(neuron_file)
    fault_neuron = np.load(neuron_file, allow_pickle=True)
    if arch == 'vgg13_dense':
        model = VGG13_dense('VGG13')
    else:
        model = VGG16_dense()

    for n, v in model.named_parameters():
        if blocks[layer] in n and 'bias' not in n:
            neuron = torch.zeros_like(v.data)
            break
    neuron_view = neuron.contiguous().view(-1)
    ratio_num = fault_neuron[: int(ratio*len(fault_neuron))]
    for id in ratio_num:
        neuron_view[id] = 1
    neuron_view = neuron_view.reshape(neuron.size())
    mask = neuron_view >= 1
    mask = mask.to(device)
    return mask


def get_mask(ratio, seed, layer, device, set, rep_type):
    root = '/public/czh/AILancet/'
    if rep_type == 'adv':
        set = set.split('_')[-1]

    neuron_load=loadmat(root + 'mask_ablation_vgg13_dense_seed%s_%s_%s/layer'%(seed, set, rep_type)+str(layer)+'.mat')
    neuron=torch.from_numpy(neuron_load['y']).to(device)
    neuron_view=neuron.contiguous().view(1, -1)
    neuron_sort=torch.argsort(neuron_view,1,descending=True)
    ratio_num=neuron_sort[0,int(np.ceil(ratio*neuron_view.shape[1]))]
    boarder=neuron_view[0,int(ratio_num)]
    mask=neuron<boarder
    return mask


def convert_net(pre_net, net, ratio, seed, layer, device, set, rep_type, mine=False):  # pretrained model dict,new define model
    net_items = net.state_dict().items()
    net_state_dict = {}
    for k, v in net.state_dict().items():
        if 'probe' in k:
            continue
        else:
            net_state_dict[k] = v
    pre_vgg_items = pre_net.items()
    new_dict_model = {}
    j = 0

    if mine:
        dict_idx = [ 7, 21, 35, 49, 63, 70, 72]
        for k, v in net_state_dict.items():  #
            v = list(pre_vgg_items)[j][1]  # weights(pretrained model)
            k = list(net_items)[j][0]  # dict names(new model)
            if j == dict_idx[layer]:
                mask = get_mask(ratio, seed, layer, device, set, rep_type)
                v_f = v * mask - v * (~mask)
                new_dict_model[k] = v_f
            else:
                new_dict_model[k] = v
            j += 1
        return new_dict_model
    else:
        dict_idx = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 72, 74]
        for k, v in net_state_dict.items():  #
            v = list(pre_vgg_items)[j][1]  # weights(pretrained model)
            k = list(net_items)[j][0]  # dict names(new model)

            if j == dict_idx[layer]:
                mask = get_mask(ratio, seed, layer, device, set, rep_type)
                v_f = v * mask - v * (~mask)
                new_dict_model[k] = v_f
            else:
                new_dict_model[k] = v
            j += 1
        return new_dict_model


def get_mask_model_vgg(model, ratio, layer, pretrained, device, seed, arch='vgg13'):
    # model = VGG13('VGG13')
    if os.path.isfile(pretrained):
        pretrained_dict = torch.load(pretrained)
        for k in model.state_dict().keys():
            if k not in pretrained_dict.keys():
                pretrained_dict.pop(k)
        pretrained_dict = convert_net(pretrained_dict, model, ratio, layer, seed, device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # wipe out
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.to(device)
    return model
# >>>>>>>>>

# >>>ai lancet

class VGG13_dense_noProbe(nn.Module):  # for ai lancet
    def __init__(self, vgg_name='VGG13', nc=10):
        super(VGG13_dense_noProbe, self).__init__()
        self.in_channels = 3
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:3])
        self.features2 = self._make_layers(cfg[vgg_name][3:6])
        self.features3 = self._make_layers(cfg[vgg_name][6:9])
        self.features4 = self._make_layers(cfg[vgg_name][9:12])
        self.features5 = self._make_layers(cfg[vgg_name][12:])
        self.dense1 = nn.Linear(512, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, nc)

    def forward(self, x, feature=False):
        f1_ = self.features1[0:2](x)
        f1 = self.features1[2:](f1_)
        f2_ = self.features2[0:2](f1)
        f2 = self.features2[2:](f2_)
        f3_ = self.features3[0:2](f2)
        f3 = self.features3[2:](f3_)
        f4_ = self.features4[0:2](f3)
        f4 = self.features4[2:](f4_)
        f5_ = self.features5[0:2](f4)
        f5 = self.features5[2:](f5_)
        f5 = f5.view(f5.size(0), -1)
        d1 = F.relu(self.dense1(f5))
        d2 = F.relu(self.dense2(d1))
        out = d2.view(d2.size(0), -1)
        out = self.classifier(out)

        return out

    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def get_mask_vgg_dense(ratio, star, layer, device):
    project_root = '/public/czh/' # '/public/home/czh_1112103010'
    file_parttern = 'AILancet/mask_ablation_vgg13_dense_seed2022/layer'+str(layer)+'.mat'
    neuron_load=loadmat()
    neuron=torch.from_numpy(neuron_load['y']).to(device)
    neuron_view=neuron.contiguous().view(1, -1)
    neuron_sort=torch.argsort(neuron_view,1,descending=True)
    ratio_num=neuron_sort[0,int(np.ceil(ratio*neuron_view.shape[1]))]
    boarder=neuron_view[0,int(ratio_num)]
    mask=neuron<boarder
    return mask


def convert_net_vgg_dense(pre_net, net, ratio, star, layer, device):  # pretrained model dict,new define model
    net_items = net.state_dict().items()
    pre_vgg_items = pre_net.items()
    new_dict_model = {}
    j = 0

    for k, v in net.state_dict().items():  #
        v = list(pre_vgg_items)[j][1]  # weights(pretrained model)
        k = list(net_items)[j][0]  # dict names(new model)
        dict_idx = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 72, 74]
        if j == dict_idx[layer]:
            mask = get_mask_vgg_dense(ratio, star, layer, device)
            v_f = v * mask - v * (~mask)
            new_dict_model[k] = v_f
        else:
            new_dict_model[k] = v
        j += 1
    return new_dict_model


def get_mask_vgg_probe(model, probe, ratio, seed, layer, pretrained, device, nc=10, set='cifar10', rep_type='bd'):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained, map_location=device)
    pretrained_dict = convert_net(pretrained_dict, model, ratio, seed, layer, device, set, rep_type)
    for p in probe:
        if os.path.isfile(p):
            probes_weight = torch.load(p, map_location=device)
            pretrained_dict.update(probes_weight)
            print("=> loading probe weights from '{}'".format(p))
        else:
            raise ValueError(f'{p} not exit')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)


def convert_vgg16(pre_net, net, ratio, seed, layer, device, set, rep_type):  # pretrained model dict,new define model
    net_items = net.state_dict().items()
    pre_vgg_items = pre_net.items()
    new_dict_model = {}
    j = 0
    for k, v in net.state_dict().items():  #
        v = list(pre_vgg_items)[j][1]  # weights(pretrained model)
        k = list(net_items)[j][0]  # dict names(new model)
        dict_idx = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84]
        if j == dict_idx[layer]:
            mask = get_mask_vgg16(ratio, seed, layer, device, set, rep_type)
            v_f = v * mask - v * (~mask)
            new_dict_model[k] = v_f
        else:
            new_dict_model[k] = v
        j += 1
    return new_dict_model


def get_mask_vgg16(ratio, seed, layer, device, set, rep_type):
    root = '/public/czh/AILancet/'
    if rep_type == 'adv':
        set = set.split('_')[-1]
    neuron_load=loadmat(root+'/mask_ablation_vgg16_dense_seed%s_%s_%s/layer'%(seed, set, rep_type)+str(layer)+'.mat')
    neuron=torch.from_numpy(neuron_load['y']).to(device)
    neuron_view=neuron.contiguous().view(1, -1)
    neuron_sort=torch.argsort(neuron_view,1,descending=True)
    ratio_num=neuron_sort[0,int(np.ceil(ratio*neuron_view.shape[1]))]
    boarder=neuron_view[0,int(ratio_num)]
    mask=neuron<boarder
    return mask

def get_mask_model_vgg16_dense(ratio, seed, layer, pretrained, device, nc=10, set='imagenet', rep_type='bd'):
    model = VGG16_dense_no_probe(num_class=nc)
    model.to(device)
    if os.path.isfile(pretrained):
        pretrained_dict = torch.load(pretrained, map_location=device) #
        pretrained_dict = convert_vgg16(pretrained_dict, model, ratio, seed, layer, device, set, rep_type)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # wipe out
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.to(device)
    else:
        raise ValueError(f'{pretrained} not exit')
    return model

def get_mask_model_vgg_dense(ratio, seed, layer, pretrained, device, nc=10, set='cifar10', rep_type='bd'):
    model = VGG13_dense_noProbe('VGG13', nc=nc)
    model.to(device)
    if os.path.isfile(pretrained):
        pretrained_dict = torch.load(pretrained, map_location=device)
        pretrained_dict = convert_net(pretrained_dict, model, ratio, seed, layer, device, set, rep_type)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # wipe out
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.to(device)
    else:
        raise ValueError(f'{pretrained} not exit')
    return model



if __name__ == "__main__":
    model = VGG13("VGG13")

    inputt = torch.ones(1, 3, 32, 32)
    a = model(inputt, probe=True)
    print(a)
    # print(model)
    # for n,m in model.named_modules():
    #     print(n)
    #     for n_,p in m.named_parameters():
    #         print(n_)
    #         print(p)
    #
    # for n,m in model.named_modules():
    #     print(n)
    #     print(m)

