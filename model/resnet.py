import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
from scipy.io import loadmat
from torchvision.models import resnet

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

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
        #  add a fc layer for
        # self._fc1 = nn.Conv2d(self.expansion * planes, 512, 1)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self._fc2 = nn.Conv2d(512, 10, 1)
        # self.flat = Flatten()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion * planes, stride=stride),
                norm_layer(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    # def forward(self, x, al_output=-1):
    #     if al_output == -1:  # normal mode, do not run branch fc layers
    #         out = F.relu(self.bn1(self.conv1(x)))
    #         out = self.bn2(self.conv2(out))
    #         out += self.shortcut(x)
    #         out = F.relu(out)
    #         return out
    #     elif al_output == 1:  # branch fc training, only run branch fc layers
    #         out = F.relu(self.bn1(self.conv1(x)))
    #         out = self.bn2(self.conv2(out))
    #         branch1 = self._fc1(out)
    #         branch2 = self.avgpool(branch1)
    #         branch = self._fc2(branch2)
    #         branch = self.flat(branch)  # branch.flatten(1)
    #         return branch
    #     else:  # get inner output, run all layers but output normally
    #
    #         out = F.relu(self.bn1(self.conv1(x)))
    #         out = self.bn2(self.conv2(out))
    #         branch1 = self._fc1(out)
    #         branch2 = self.avgpool(branch1)
    #         branch = self._fc2(branch2)
    #         branch = self.flat(branch)  # branch.flatten(1)
    #         out += self.shortcut(x)
    #         out = F.relu(out)
    #         return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, self.expansion * planes)
        self.bn3 = norm_layer(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion * planes, stride=stride),
                norm_layer(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Probe(nn.Module):
    def __init__(self, in_ch, fc_ch, layer_num=2, num_class=10):
        super(Probe, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )
        if layer_num == 2:
            self.convs = nn.Sequential(
                nn.Conv2d(in_ch, in_ch*2, 3, 2, 1),
                nn.Conv2d(in_ch*2, in_ch*2, 3, 2, 1),
                nn.BatchNorm2d(in_ch*2),
                nn.ReLU(),
                nn.Conv2d(in_ch*2, in_ch*4, 3, 1, 1),
                nn.Conv2d(in_ch*4, in_ch*4, 3, 1, 1),
                nn.BatchNorm2d(in_ch*4),
                nn.ReLU(),
                nn.AvgPool2d(4, 4)
            )
        elif layer_num == 1:
            self.convs = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, 1, 1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                nn.Conv2d(in_ch, in_ch, 3, 1, 1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                nn.AvgPool2d(4, 4)
            )
        self.fc = nn.Linear(fc_ch, num_class)

    def forward(self, x):
        feat = self.features(x)
        feat = self.convs(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, base_width=64, num_classes=10):
        super(ResNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.in_planes = base_width
        self.conv1 = nn.Conv2d(3, base_width, 3, 1, 1, bias=False)
        self.bn1 = norm_layer(base_width)
        self.layer1 = self._make_layer(block, base_width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_width*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_width*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_width*8, num_blocks[3], stride=2)
        self.probe1 = Probe(base_width, base_width*16, 2, num_classes)
        self.probe2 = Probe(base_width*2, base_width*8, 2, num_classes)
        self.probe3 = Probe(base_width*4, base_width*16, 1, num_classes)
        self.probe4 = Probe(base_width*8, base_width*8, 1, num_classes)
        self.fc1 = nn.Conv2d(base_width * 8 * block.expansion, 1024, 1)  # 512 * 1024
        self.fc2 = nn.Conv2d(1024, 1024, 1)
        self.fc3 = nn.Conv2d(1024, num_classes, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(base_width * 8 * block.expansion, num_classes, 1)
        self.block = block

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
        
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, probe=False):
        if not probe:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = self.fc(out)
            return out.flatten(1)
        else:
            input = F.relu(self.bn1(self.conv1(x)))
            l1 = self.layer1(input)
            out1 = self.probe1(l1)
            l2 = self.layer2(l1)
            out2 = self.probe2(l2)
            l3 = self.layer3(l2)
            out3 = self.probe3(l3)
            l4 = self.layer4(l3)
            out4 = self.probe4(l4)
            pred = self.avgpool(l4)
            pred = self.fc(pred)
            return out1, out2, out3, out4, pred.flatten(1)


class ResNet_Dense(nn.Module):
    def __init__(self, block, num_blocks, base_width=64, num_classes=10):
        super(ResNet_Dense, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.in_planes = base_width
        self.conv1 = nn.Conv2d(3, base_width, 3, 1, 1, bias=False)
        self.bn1 = norm_layer(base_width)
        self.layer1 = self._make_layer(block, base_width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_width * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8, num_blocks[3], stride=2)
        self.probe1 = Probe(base_width, base_width * 16, 2, num_classes)
        self.probe2 = Probe(base_width * 2, base_width * 8, 2, num_classes)
        self.probe3 = Probe(base_width * 4, base_width * 16, 1, num_classes)
        self.probe4 = Probe(base_width * 8, base_width * 8, 1, num_classes)
        self.probe5 = nn.Linear(1024, num_classes)
        self.probe6 = nn.Linear(1024, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(base_width * 8 * block.expansion, 1024, 1) # 512 * 1024
        self.fc2 = nn.Conv2d(1024, 1024, 1)
        self.fc3 = nn.Conv2d(1024, num_classes, 1)
        self.block = block

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))

            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, probe=False):
        if not probe:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = self.fc1(out)
            out = self.fc2(out)
            out = self.fc3(out)
            return out.flatten(1)
        else:
            input = F.relu(self.bn1(self.conv1(x)))
            l1 = self.layer1(input)
            out1 = self.probe1(l1)
            l2 = self.layer2(l1)
            out2 = self.probe2(l2)
            l3 = self.layer3(l2)
            out3 = self.probe3(l3)
            l4 = self.layer4(l3)
            out4 = self.probe4(l4)
            pool = self.avgpool(l4)
            dense1 = self.fc1(pool)
            out5 = self.probe5(dense1.view(dense1.size(0), -1))
            dense2 = self.fc2(dense1)
            out6 = self.probe6(dense2.view(dense2.size(0), -1))
            pred = self.fc3(dense2)
            return out1, out2, out3, out4, out5, out6, pred.flatten(1)


def cResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def cResNet18_Dense(num_classes=10):
    return ResNet_Dense(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def cResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def cResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def cResNet101(num_classes=50):
    return ResNet_Dense(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def cResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


# >>>>>>>>>> AI Lancet model loading

class ResNet18(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], base_width=64, num_classes=10):
        super(ResNet18, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.in_planes = base_width
        self.conv1 = nn.Conv2d(3, base_width, 3, 1, 1, bias=False)
        self.bn1 = norm_layer(base_width)
        self.layer1 = self._make_layer(block, base_width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_width * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(base_width * 8 * block.expansion, num_classes, 1)
        self.block = block

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))

            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, feature=False):
        c1 = F.relu(self.bn1(self.conv1(x)))
        l1_ = self.layer1[0](c1)
        l1 = self.layer1[1](l1_)
        l2_ = self.layer2[0](l1)
        l2 = self.layer2[1](l2_)
        l3_ = self.layer3[0](l2)
        l3 = self.layer3[1](l3_)
        l4_ = self.layer4[0](l3)
        l4 = self.layer4[1](l4_)

        out = self.avgpool(l4)
        out = self.fc(out)
        if feature:
            return out
        else:
            return out, [c1, l1_, l1, l2_, l2, l3_, l3, l4_, l4, out]



def get_mask(ratio, seed, layer, device, set, rep_type):
    root = '/public/czh/AILancet/'
    if rep_type == 'adv':
        set = set.split('_')[-1]
    neuron_load = loadmat(root + 'mask_ablation_res18_dense_seed%s_%s_%s/layer' % (seed, set, rep_type) + str(layer) + '.mat')
    neuron=torch.from_numpy(neuron_load['y']).to(device)
    neuron_view=neuron.contiguous().view(1, -1)
    neuron_sort=torch.argsort(neuron_view,1,descending=True)
    ratio_num=neuron_sort[0,int(np.ceil(ratio*neuron_view.shape[1]))]
    boarder=neuron_view[0,int(ratio_num)]
    mask=neuron<boarder
    return mask


def convert_net(pre_net, net, ratio, seed, layer, device, set, rep_type):  # pretrained model dict,new define model
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
    # >>>
    # mask = get_mask(ratio, star, layer, device)
    # print(f'layer {layer}, mask shape {mask.shape}, ')
    # >>>
    for k, v in net_state_dict.items():  #
        v = list(pre_vgg_items)[j][1]  # weights(pretrained model)
        k = list(net_items)[j][0]  # dict names(new model)
        dict_idx = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 122, 124]
        #[0, 12, 28, 44, 66, 82, 104, 120, 142, 152, 154, 156]

        if j == dict_idx[layer]:
            mask = get_mask(ratio, seed, layer, device, set, rep_type)
            v_f = v * mask - v * (~mask)
            new_dict_model[k] = v_f
        else:
            new_dict_model[k] = v
        j += 1
    return new_dict_model



class ResNet18_Dense(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], base_width=64, num_classes=10):
        super(ResNet18_Dense, self).__init__()
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
        self.probe1 = Probe(base_width, base_width * 16, 2, num_classes)
        self.probe2 = Probe(base_width * 2, base_width * 8, 2, num_classes)
        self.probe3 = Probe(base_width * 4, base_width * 16, 1, num_classes)
        self.probe4 = Probe(base_width * 8, base_width * 8, 1, num_classes)
        self.probe5 = nn.Linear(1024, 10)
        self.probe6 = nn.Linear(1024, 10)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # self.fc = nn.Conv2d(base_width * 8 * block.expansion, num_classes, 1)
        self.block = block

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))

            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, probe=False):
        if not probe:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = self.fc1(out)
            out = self.fc2(out)
            out = self.fc3(out)
            return out.flatten(1)
        else:
            input = F.relu(self.bn1(self.conv1(x)))
            l1 = self.layer1(input)
            out1 = self.probe1(l1)
            l2 = self.layer2(l1)
            out2 = self.probe2(l2)
            l3 = self.layer3(l2)
            out3 = self.probe3(l3)
            l4 = self.layer4(l3)
            out4 = self.probe4(l4)
            pool = self.avgpool(l4)
            dense1 = self.fc1(pool)
            out5 = self.probe5(dense1.view(dense1.size(0), -1))
            dense2 = self.fc2(dense1)
            out6 = self.probe6(dense2.view(dense2.size(0), -1))
            pred = self.fc3(dense2)
            return out1, out2, out3, out4, out5, out6, pred.flatten(1)


def get_mask_model_res_probe(probe, ratio, seed, layer, pretrained, device, nc=10, set='cifar10', rep_type='bd'):
    model = ResNet18_Dense(num_classes=nc)
    pretrained_dict = torch.load(pretrained, map_location=device)
    pretrained_dict = convert_net(pretrained_dict, model, ratio, seed, layer, device, set, rep_type)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'probe' not in k}  # wipe out
    for p in probe:
        if os.path.isfile(p):
            probes_weight = torch.load(p, map_location=device)
            pretrained_dict.update(probes_weight)
            print("=> loading probe weights from '{}'".format(p))
        else:
            raise ValueError(f'{p} not exit')
    #print(pretrained_dict.keys())
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    return model

def get_mask_model_res(ratio, seed, layer, pretrained, device, nc=10, set='cifar10', rep_type='bd'):
    model = ResNet18_Dense(num_classes=nc)
    if os.path.isfile(pretrained):
        pretrained_dict = torch.load(pretrained, map_location=device)
        pretrained_dict = convert_net(pretrained_dict, model, ratio, seed, layer, device, set, rep_type)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # wipe out
        # print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.to(device)
    return model


# >>>>>>>>>>>>>>>>>
if __name__ == "__main__":
    import torch

    inter_feature = []
    def hook(module, input, output):
        inter_feature.append(output.clone().detach())


    def cResNet30(num_classes=10):
        return ResNet(Bottleneck, [3, 4, 10, 3], num_classes=num_classes)
    input = torch.randn(2, 3, 32, 32)
    model = cResNet30()
    print(model)
    output = model(input)
    print(output)








