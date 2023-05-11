import os
import sys

sys.path.append('/public/czh/repair/')
import torch
import torchvision.transforms as transforms
import numpy as np
#from utils.make_dataset import get_standard, get_cifar10C, get_backdoor
from model.cifar10_vgg import VGG13_dense, VGG16_dense
from model.resnet import cResNet18_Dense
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from utils.utils import *
import args


def get_mis(model, set, save_dir):
    file_path, file_name = os.path.split(save_dir)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    clean = get_standard(set=set, num=20000, train=False, seed=2333, process=['std'])
    nor_loader = DataLoader(clean, batch_size=1, shuffle=False, num_workers=0)
    mis_classified = None
    true_labels = None
    for images, labels in nor_loader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images)
        _, idx = torch.max(pred, dim=1)
        if idx != labels:
            if mis_classified is None:
                mis_classified = images
                true_labels = labels
            else:
                mis_classified = torch.concat((mis_classified, images), dim=0)
                true_labels = torch.concat((true_labels, labels), dim=0)

    mis_classified = mis_classified.cpu().detach().numpy()
    true_labels = true_labels.cpu().detach().numpy()
    np.save(save_dir, mis_classified)
    np.save(save_dir.replace('.npy', '_label.npy'), true_labels)


class MyDataset4Misc(torch.utils.data.Dataset):
    def __init__(self, root, transforms, random_seed=2023):
        #self.max_size = max_size
        images = np.load(os.path.join(root, 'wp.npy'))
        labels = np.load(os.path.join(root, 'wp_label.npy'))
        assert len(labels) == len(images)
        load_imgs = images
        load_labels = labels

        # print(f'load image shape is {load_imgs.shape} label shape {load_labels.shape}')
        self.transforms = transforms
        self.data = load_imgs
        self.labels = load_labels
        # self.predicted_labels = np.array(predicted_labels)

         # ref: https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        img = np.transpose(img, axes=(1, 2, 0))
        # print('---<<<<<<<<<')
        # print(f'img shape {img.shape} label {target.shape}, type {type(img)}\n')
        img = self.transforms(img)

        target = torch.tensor(target)
        return img, target

    def __len__(self):
        return len(self.data)


def get_misclassified(set, process, num, arch, seed, validate=False, root='./datasets/'):
    pro_lib = {'flip': transforms.RandomHorizontalFlip(), 'rota': transforms.RandomRotation(30),
               'crop': transforms.RandomCrop((32, 32)),
               'tt': transforms.ToTensor()}

    tfs = [pro_lib['tt']]
    for p in process:
        if p == 'std':
            continue
        else:
            tfs.append(pro_lib[p])
    tfs = transforms.Compose(tfs)

    prefix = '%s_misclassified/%s_%s.npy' % (set, arch, seed)
    data_path = os.path.join(root, prefix)
    if validate:
        random_seed = 2023
    else:
        random_seed = 20233
    misclassified = MyDataset4Misc(data_path, tfs, random_seed, max_size=num)
    return misclassified


def test(device):
    model_file = './checkpoints/final/%s/seed%s/std/model-best.pt' % ('res18_dense', 2022)
    model = cResNet18_Dense()
    pretrained(model_file, model, device, [])
    model.eval()
    dd = get_misclassified('cifar10', ['std', ], 1000, 'res18_dense', 2022) # 'rota', 'crop', 'flip'
    loader = DataLoader(dd, batch_size=10, shuffle=False)
    for img, label in loader:
        img, label = img.to(device), label.to(device)
        pred = model(img)
        pred = torch.argmax(pred, dim=0)
        print(pred)
        print(label)
        break


def get_mis_batch():
    # archs = ['vgg13_dense', 'res18_dense'] #'vgg16_dense'
    # sets = ['cifar10', 'gtsrb']
    # num_classes = {'gtsrb': 43, 'cifar10': 10, 'imagenet': 10, }
    # for a in archs:
    #     for s in sets:
    #         for seed in range(2022, 2032):
    #             set_random_seed(seed)
    #             save_dir = f'/public/czh/repair/checkpoints/{s}/{a}/seed{seed}/std/model-best.pt'
    #             if a == 'vgg13_dense':
    #                 model = VGG13_dense('VGG13', num_classes[s])
    #             elif a == 'vgg16_dense':
    #                 model = VGG16_dense()
    #             else:
    #                 model = cResNet18_Dense(num_classes[s])
    #             pretrained(save_dir, model, device, [])
    #             model.eval()
    #             prefix = f'{s}/wp/{a}_{seed}/wp.npy'
    #             data_path = os.path.join('/public/czh/repair/datasets/', prefix)
    #             get_mis(model, s, data_path)

    for seed in range(2022, 2032):
        save_dir = f'/public/czh/repair/checkpoints/imagenet/vgg16_dense/seed{seed}/std/model-best.pt'
        prefix = f'imagenet/wp/vgg16_dense_{seed}/wp.npy'
        data_path = os.path.join('/public/czh/repair/datasets/', prefix)
        model = VGG16_dense()
        pretrained(save_dir, model, device, [])
        model.eval()
        get_mis(model, 'imagenet', data_path)



if __name__ == '__main__':
    gpu = 1
    device = 'cuda:%s' % gpu if torch.cuda.is_available() else 'cpu'
    data_root = '/public/czh/repair/datasets/cifar10/wp/res18_dense_2022/'

    dataset = MyDataset4Misc(data_root, transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    for img, label in loader:
        print(img.shape)

    # get_mis_batch()

    #
    # for i in range(10):
    #     set_random_seed(2022 + i)
    #     seed = 2022 + i
    #     model_file = './checkpoints/final/%s/seed%s/std/model-best.pt' % ('vgg13_dense', 2022)
    #     model = VGG13_dense('VGG13')
    #     pretrained(model_file, model, device, [])
    #     model.eval()
    #     prefix = '%s_misclassified/%s_%s.npy' % ('cifar10', 'vgg13_dense', seed)
    #     data_path = os.path.join('./datasets/', prefix)
    #     get_mis(model, data_path)
    # test(device)
