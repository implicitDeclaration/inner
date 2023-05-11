import os
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.utils.data import DataLoader, Dataset
from tensorflow.keras.datasets import cifar10, mnist
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from utils.make_dataset import get_standard, get_backdoor
import numpy as np
import h5py
from tensorflow.keras.models import load_model


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


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




class CAREProbeVGG13_dense(nn.Module):  # for ai lancet
    def __init__(self, r_weight, rep_index, vgg_name='VGG13', num_class=10):
        super(CAREProbeVGG13_dense, self).__init__()
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
        self.probe1 = Probe(64, 2, num_class=num_class)
        self.probe2 = Probe(128, 2, num_class=num_class)
        self.probe3 = Probe(256, 1, num_class=num_class)
        self.probe4 = Probe(512, 1, num_class=num_class)
        self.probe5 = nn.Linear(512, 10)
        self.probe6 = nn.Linear(1024, 10)
        self.probe7 = nn.Linear(1024, num_class)
        self.r_weight = r_weight
        self.rep_index = rep_index
    def forward(self, x, probe=False):
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

        f5_flatten = f5.view(f5.size(0), -1)
        for i in range(0, len(self.rep_index)):
            rep_idx = int(self.rep_index[i])
            f5_flatten[:, rep_idx] = (self.r_weight[i]) * f5_flatten[:, rep_idx]
        f5_new = f5_flatten.reshape(f5.shape)

        p5 = self.probe5(f5_new)

        d1 = F.relu(self.dense1(f5_new))
        # d1 = F.relu(self.dense1(f5))
        # d1_flatten = d1.view(d1.size(0), -1)

        # for i in range(0, len(self.rep_index)):
        #     rep_idx = int(self.rep_index[i])
        #     d1_flatten[:, rep_idx] = (self.r_weight[i]) * d1_flatten[:, rep_idx]
        # d1_new = d1_flatten.reshape(d1.shape)
        p6 = self.probe6(d1)
        d2 = F.relu(self.dense2(d1))
        p7 = self.probe7(d2)
        out = d2.view(d2.size(0), -1)
        out = self.classifier(out)
        if probe:
            return p1, p2, p3, p4, p5, p6, p7, out
        else:
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


class CAREVGG16_dense(nn.Module):  #
    def __init__(self, r_weight, rep_index,vgg_name='VGG16', num_class=10):
        super(CAREVGG16_dense, self).__init__()
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
        self.r_weight = r_weight
        self.rep_index = rep_index
    def forward(self, x, feature=False):
        f1 = self.features1(x)
        f2 = self.features2(f1)
        f3 = self.features3(f2)
        f4 = self.features4(f3)
        f5 = self.features5(f4)
        f5 = f5.view(f5.size(0), -1)
        for i in range(0, len(self.rep_index)):
            rep_idx = int(self.rep_index[i])
            f5[:, rep_idx] = (self.r_weight[i]) * f5[:, rep_idx]

        d1 = F.relu(self.dense1(f5))
        d2 = F.relu(self.dense2(d1))
        out = d2.view(f5.size(0), -1)
        out = self.classifier(out)
        if feature:
            return f5
        else:
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



class CAREExchangeVGG13_dense(nn.Module):
    def __init__(self, r_weight, rep_index, nc=10, vgg_name='VGG13'):
        super(CAREExchangeVGG13_dense, self).__init__()
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
        self.r_weight = r_weight
        self.rep_index = rep_index
        self.block_name = 'dense1'
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
    def apply_reweight(self):
        for name, params in self.named_parameters():  # initialize routing
            if 'bias' in name or 'probe' in name or '_fc' in name:
                continue
            if self.block_name in name:
                flat_param = params.clone().view(-1)
                for i in range(0, len(self.rep_index)):
                    rep_idx = int(self.rep_index[i])
                    flat_param[rep_idx] = self.r_weight[i]
                flat_param = flat_param.reshape(params.size())
                params.data = flat_param

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



class CAREVGG13_dense(nn.Module):  # for ai lancet
    def __init__(self, r_weight, rep_index, nc=10, vgg_name='VGG13'):
        super(CAREVGG13_dense, self).__init__()
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
        self.r_weight = r_weight
        self.rep_index = rep_index
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
        for i in range(0, len(self.rep_index)):
            rep_idx = int(self.rep_index[i])
            f5[:, rep_idx] = (self.r_weight[i]) * f5[:, rep_idx]
        d1 = F.relu(self.dense1(f5))
        # d1_flatten = d1.view(d1.size(0), -1)
        # for i in range(0, len(self.rep_index)):
        #     rep_idx = int(self.rep_index[i])
        #     d1_flatten[:, rep_idx] = (self.r_weight[i]) * d1_flatten[:, rep_idx]
        # d1_new = d1_flatten.reshape(d1.shape)
        d2 = F.relu(self.dense2(d1))
        out = d2.view(d2.size(0), -1)
        out = self.classifier(out)
        if feature:
            return f5
        else:
            return out
    def get_neuron(self, x):
        pass

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




def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PTFlatten(nn.Module):
    def __init__(self):
        super(PTFlatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.planes = planes
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = norm_layer(planes)
        self.flat = PTFlatten()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion * planes, stride=stride),
                norm_layer(self.expansion * planes),
            )

    def forward(self, x, al_output=-1):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_Dense(nn.Module):
    def __init__(self, block, num_blocks, r_weight, rep_index, base_width=64, num_classes=10):
        super(ResNet_Dense, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.in_planes = base_width
        self.conv1 = nn.Conv2d(3, base_width, 3, 1, 1, bias=False)
        self.bn1 = norm_layer(base_width)
        self.layer1 = self._make_layer(block, base_width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_width * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8, num_blocks[3], stride=2)
        self.fc1 = nn.Conv2d(base_width * 8 * block.expansion, 1024, 1)  # 512 * 1024
        self.fc2 = nn.Conv2d(1024, 1024, 1)
        self.fc3 = nn.Conv2d(1024, num_classes, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Conv2d(base_width * 8 * block.expansion, num_classes, 1)
        self.block = block
        self.r_weight = r_weight
        self.rep_index = rep_index
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))

            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        f0 = self.avgpool(out)
        f0_flatten = f0.view(f0.size(0), -1)
        for i in range(0, len(self.rep_index)):
            rep_idx = int(self.rep_index[i])
            f0_flatten[:, rep_idx] = (self.r_weight[i]) * f0_flatten[:, rep_idx]
        f0_new = f0_flatten.reshape(f0.shape)

        f1 = self.fc1(f0_new)
        # f1_flatten = f1.view(f1.size(0), -1)
        # for i in range(0, len(self.rep_index)):
        #     rep_idx = int(self.rep_index[i])
        #     f1_flatten[:, rep_idx] = (self.r_weight[i]) * f1_flatten[:, rep_idx]
        # f1_new = f1_flatten.reshape(f1.shape)
        #
        f2 = self.fc2(f1)
        # f2_flatten = f2.view(f2.size(0), -1)
        # for i in range(0, len(self.rep_index)):
        #     rep_idx = int(self.rep_index[i])
        #     f2_flatten[:, rep_idx] = (self.r_weight[i]) * f2_flatten[:, rep_idx]
        # f2_new = f2_flatten.reshape(f2.shape)

        out = self.fc3(f2)
        if feature:
            return f0_flatten
        else:

            return out.flatten(1)



def CAREResNet18(r_weight, rep_index, num_classes=10):
    return ResNet_Dense(BasicBlock, [2, 2, 2, 2], r_weight, rep_index, num_classes=num_classes)


def load_care_model(arch, ):
    if arch == 'vgg13':
        r_weight = np.load('/public/home/czh_1112103010/care-main/ckpts/vgg13/seed2022/layer38_alpha_1_r_weight.npy', allow_pickle=True)
        r_idx = np.load('/public/home/czh_1112103010/care-main/ckpts/vgg13/seed2022/layer38_alpha_1_r_idx.npy', allow_pickle=True)
        split_index = 38
    else:
        r_weight = np.load('/public/home/czh_1112103010/care-main/ckpts/vgg13_dense/seed2022/good_result_bkp/layer39_alpha_1.0_n_5_r_weight.npy', allow_pickle=True)
        r_idx = np.load('/public/home/czh_1112103010/care-main/ckpts/vgg13_dense/seed2022/good_result_bkp/layer39_alpha_1.0_n_5_r_idx.npy', allow_pickle=True)
        split_index = 39
    print(r_weight)
    print(r_idx)
    if arch == 'vgg13':
        model_file = '/public/home/czh_1112103010/care-main/ckpts/vgg13/seed2022/bd/cifar10_whole_model.hdf5'
    else:
        model_file = '/public/home/czh_1112103010/care-main/ckpts/vgg13_dense/seed2022/bd/cifar_whole_model.hdf5'
    model = load_model(model_file)

    model1, model2 = split_keras_model(model, index=split_index)
    return model1, model2


def get_r_w_id(arch):
    if arch == 'vgg13':
        r_weight = np.load('/public/home/czh_1112103010/care-main/ckpts/vgg13/seed2022/layer38_alpha_1_r_weight.npy', allow_pickle=True)
        r_idx = np.load('/public/home/czh_1112103010/care-main/ckpts/vgg13/seed2022/layer38_alpha_1_r_idx.npy', allow_pickle=True)
        split_index = 38
    else:
        r_weight = np.load('/public/home/czh_1112103010/care-main/ckpts/vgg13_dense/seed2022/good_result_bkp/layer39_alpha_1.0_n_5_r_weight.npy', allow_pickle=True)
        r_idx = np.load('/public/home/czh_1112103010/care-main/ckpts/vgg13_dense/seed2022/good_result_bkp/layer39_alpha_1.0_n_5_r_idx.npy', allow_pickle=True)
        split_index = 39
    return r_weight, r_idx


def split_keras_model(lmodel, index):
    model1 = Model(inputs=lmodel.inputs, outputs=lmodel.layers[index - 1].output)
    model2_input = Input(lmodel.layers[index].input_shape[1:])
    model2 = model2_input
    for layer in lmodel.layers[index:]:
        model2 = layer(model2)
    model2 = Model(inputs=model2_input, outputs=model2)

    return (model1, model2)


def validate_keras(model, data_loader, per_class=True, r_idx=None, r_weight=None, nc=10):
    total,correct = 0, 0

    class_correct = list(0. for i in range(nc))
    class_total = list(0. for i in range(nc))

    for data, target in data_loader:
        bs = data.size(0)
        data, target = data.cpu().detach().numpy().transpose(0, 2, 3, 1), target.cpu().detach().numpy()
        if r_idx is not None:
            p_prediction = model[0].predict(data)
            l_shape = p_prediction.shape
            _p_prediction = np.reshape(p_prediction, (len(p_prediction), -1))
            do_hidden = _p_prediction.copy()

            for i in range(0, len(r_idx)):
                rep_idx = int(r_idx[i])
                do_hidden[:, rep_idx] = (r_weight[i]) * _p_prediction[:, rep_idx]
            outputs = model[1].predict(do_hidden.reshape(l_shape))
        else:
            outputs = model.predict(data)
        outputs = outputs.reshape(bs, nc)
        predicted = np.argmax(outputs, axis=1)

        c = (predicted == target).squeeze()

        if per_class:
            for i in range(bs):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        else:
            total += len(target)
            correct += (predicted == target).sum().item()

    if per_class:
        acc_per_class = []
        for i in range(nc):
            acc_per_class.append(class_correct[i] / class_total[i])
        return acc_per_class
    else:
        accuracy = correct / total
        return accuracy
