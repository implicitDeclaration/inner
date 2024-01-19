"""

"""
# import sys
# sys.path.append('~/utils')
import torch
import torchvision
import os
import numpy as np
import copy
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import random
from PIL import Image
from data.misclassification import MyDataset4Misc
from utils.PoisonedDataset import PoisonedCifar, PoisonedMNIST, CleanMNIST, PoisonedSVHN, PoisonedGTSRB, \
    PoisonedCifarPair, PoisonedImageNet, WanetCifar, Dataset, SSBACifar, BlendCifar, SSBAGTSRB, BlendGTSRB, WanetGTSRB,BlendImageNet
import pandas as pd



class Subset(Dataset):
    def __init__(self,dataset,indices):
        super(Subset, self).__init__()
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def save_npy(file_name: str, a):
    np.save(file_name, a)


def load_npy(file_name):
    a = np.load(file_name)
    return a

# 自定义添加椒盐噪声的 transform
class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    # transform 会调用该方法
    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        # 如果随机概率小于 seld.p，则执行 transform
        if random.uniform(0, 1) < self.p:
            # 把 image 转为 array
            img_ = np.array(img).copy()
            # 获得 shape
            h, w, c = img_.shape
            # 信噪比
            signal_pct = self.snr
            # 椒盐噪声的比例 = 1 -信噪比
            noise_pct = (1 - self.snr)
            # 选择的值为 (0, 1, 2)，每个取值的概率分别为 [signal_pct, noise_pct/2., noise_pct/2.]
            # 椒噪声和盐噪声分别占 noise_pct 的一半
            # 1 为盐噪声，2 为 椒噪声
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   # 盐噪声
            img_[mask == 2] = 0     # 椒噪声
            # 再转换为 image
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        # 如果随机概率大于 seld.p，则直接返回原图
        else:
            return img


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img = Image.fromarray(img.astype('uint8'))# .convert('RGB')
            return img
        else:
            return img


# for standard model
# 直接取一部分的训练集数据作为参照
def get_standard(process, num=1000, seed=0, train=False,
                 set='cifar10'):
    if 'cifar10' in set:
        set = 'cifar10'
    roots = {'cifar10': "./datasets/cifar10", 'svhn': "./datasets/svhn", 'gtsrb': "./datasets/gtsrb", 'imagenet': "./datasets/imagenet"}
    root = roots[set]
    if not os.path.exists(root):
        os.makedirs(root)
    cifar10_norm = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262] )
    shift_norm= transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.4, 0.4, 0.4])
    svhn_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    gtsrb_norm = transforms.Normalize([0, 0, 0], [1, 1, 1])
    mnist_norm = transforms.Normalize((0.1307,), (0.3081,))
    imgset_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    std_normalize = {'cifar10': cifar10_norm, 'svhn': svhn_norm, 'mnist': mnist_norm,
                     'gtsrb': gtsrb_norm, 'imagenet':imgset_norm}

    pro_lib = {'flip': transforms.RandomHorizontalFlip(), 'rota': transforms.RandomRotation(30),
               'gau': AddGaussianNoise(), 'gtsrb_size': transforms.Resize([32, 32]),
               'pep': AddPepperNoise(0.95, 0.05), 'crop': transforms.RandomCrop((32, 32), padding=4),
               'shift': shift_norm, 'tt': transforms.ToTensor(), 'std': std_normalize[set],
               'img_size': transforms.Resize(256), 'img_crop': transforms.CenterCrop(224),
               'jit': transforms.ColorJitter(brightness=0.2)
    }
    process.insert(0, 'tt')
    if set == 'gtsrb':
        process.insert(0, 'gtsrb_size')
    elif set == 'imagenet':
        process.insert(0, 'img_crop')
        process.insert(0, 'img_size')
    my_trans = [pro_lib[i] for i in process]
    train_transform = transforms.Compose(my_trans)
    if set == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=train_transform)

    elif set == 'mnist':
        dataset = CleanMNIST(root=root, train=train, download=True, transform=train_transform)
    elif set == 'svhn':
        sp = 'train' if train else 'test'
        dataset = torchvision.datasets.SVHN(root, split=sp, download=True, transform=train_transform)
    elif set.lower() == 'gtsrb':
        dataset = GTSRB(root=root, train=train, transform=train_transform)
    elif set.lower() == 'imagenet':
        if train:
            imagenet_dir = os.path.join(root, 'train')
        else:
            imagenet_dir = os.path.join(root, 'val')
        dataset = torchvision.datasets.ImageFolder(imagenet_dir, train_transform)
    elif set == 'flower':
        dataset = Flower()
        if train:
            dataset = dataset.train_set
        else:
            dataset = dataset.val_set
    else:
        raise ValueError(f'set {set} not support')
    random.seed(seed)
    if num > len(dataset):
        print(f'set {set} length {len(dataset)}')
        num = len(dataset)
    if seed == 0:
        indices = list(range(0, num))
    else:
        indices = random.sample(list(range(len(dataset))), num)

    return Subset(dataset, indices)


# for backdoor model
# 因为backdoor攻击成功率一般是100% 因此这里就不考虑失败的情况了
def get_backdoor(process, num=1000, train=True, mode='ptest', model_seed=2022, arch=None,
                 seed=0, RTL=False, avoid_trg_class=False, corruption_root=None, name=None, set='cifar10'):
    cifar10_norm = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    shift_norm = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.4, 0.4, 0.4])
    svhn_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    gtsrb_norm = transforms.Normalize([0, 0, 0], [1, 1, 1])
    mnist_norm = transforms.Normalize((0.1307,), (0.3081,))
    imgset_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    root_set = set.split('_')[-1]
    roots = {'cifar10': "./datasets/cifar10", 'svhn': "./datasets/svhn", 'gtsrb': "./datasets/gtsrb",
            'imagenet': './datasets/imagenet', }
    root = roots[root_set]
    if not os.path.exists(root):
        os.makedirs(root)
    std_normalize = {'cifar10': cifar10_norm, 'svhn': svhn_norm, 'mnist': mnist_norm,
                     'gtsrb': gtsrb_norm, 'imagenet': imgset_norm}

    pro_lib = {'flip': transforms.RandomHorizontalFlip(), 'rota': transforms.RandomRotation(30),
               'gau': AddGaussianNoise(), 'gtsrb_size': transforms.Resize([32, 32]),
               'pep': AddPepperNoise(0.95, 0.05), 'crop': transforms.RandomCrop((32, 32)),
               'shift': shift_norm, 'tt': transforms.ToTensor(), 'std': std_normalize[root_set],
               'img_size': transforms.Resize(256), 'img_crop': transforms.CenterCrop(224),
               'jit': transforms.ColorJitter(brightness=0.2)
               }
    process.insert(0, 'tt')
    if root_set == 'gtsrb':
        process.insert(0, 'gtsrb_size')
    elif root_set == 'imagenet':
        process.insert(0, 'img_crop')
        process.insert(0, 'img_size')
        if 'crop' in process:
            process = ['img_crop' if i == 'crop' else i for i in process]

    my_trans = [pro_lib[i] for i in process]
    train_transform = transforms.Compose(my_trans)
    if set == 'bd_cifar10':
        dataset = PoisonedCifar(root=root, train=train, transform=train_transform, trigger_label=0, mode=mode,
                                return_true_label=RTL, corruption_root=corruption_root, name=name,
                                avoid_trg_class=avoid_trg_class)
    elif set == 'mnist':
        dataset = PoisonedMNIST(root=root, train=train, transform=train_transform, trigger_label=9, mode=mode,
                                return_true_label=RTL, corruption_root=corruption_root, name=name)
    elif set == 'svhn':
        sp = 'train' if train else 'test'
        dataset = PoisonedSVHN(root=root, split=sp, transform=train_transform, trigger_label=0, mode=mode,
                                return_true_label=RTL, corruption_root=corruption_root, name=name)
    elif set == 'bd_gtsrb':
        dataset = PoisonedGTSRB(root=root, train=train, transform=train_transform, trigger_label=0, mode=mode,
                                return_true_label=RTL, avoid_trg_class=avoid_trg_class)
    elif set == 'pair_cifar10':
        dataset = PoisonedCifarPair(root=root, train=train, transform=train_transform, trigger_label=0, mode=mode,
                                return_true_label=RTL, corruption_root=corruption_root, name=name,
                                avoid_trg_class=avoid_trg_class)
    elif set == 'wanet_cifar10':
        root = f'/public/czh/BackdoorBench-main/attack/checkpoints/cifar10/wanet/{arch}/seed{model_seed}/'
        dataset = WanetCifar(root, transform=train_transform, avoid_trg_class=avoid_trg_class, return_true_label=RTL, mode=mode)
    elif set == 'ssba_cifar10':
        dataset = SSBACifar(root, train, transform=train_transform, avoid_trg_class=avoid_trg_class, return_true_label=RTL, mode=mode)
    elif set == 'blend_cifar10':
        dataset = BlendCifar(root=root, train=train, transform=train_transform, trigger_label=0, mode=mode,
                             return_true_label=RTL, corruption_root=corruption_root, name=name,
                             avoid_trg_class=avoid_trg_class)
    elif set == 'wanet_gtsrb':
        root = f'/public/czh/BackdoorBench-main/attack/checkpoints/gtsrb/wanet/{arch}/seed{model_seed}/'
        dataset = WanetGTSRB(root, transform=train_transform, avoid_trg_class=avoid_trg_class, return_true_label=RTL, mode=mode)
    elif set == 'ssba_gtsrb':
        dataset = SSBAGTSRB(root, train, transform=train_transform, avoid_trg_class=avoid_trg_class, return_true_label=RTL, mode=mode)
    elif set == 'blend_gtsrb':
        dataset = BlendGTSRB(root=root, train=train, transform=train_transform, trigger_label=0, mode=mode,
                             return_true_label=RTL, corruption_root=corruption_root, name=name,
                             avoid_trg_class=avoid_trg_class)
    elif set == 'bd_imagenet':
        if train:
            imagenet_dir = os.path.join(root, 'train')
        else:
            imagenet_dir = os.path.join(root, 'val')
        patt_trans = [pro_lib['tt'], pro_lib['std']]
        pt = transforms.Compose(patt_trans)
        dataset = PoisonedImageNet(root=imagenet_dir, transform=train_transform, pattern_transform=pt, trigger_label=0, mode=mode,
                                   return_true_label=RTL)
    elif set == 'blend_imagenet':
        if train:
            imagenet_dir = os.path.join(root, 'train')
        else:
            imagenet_dir = os.path.join(root, 'val')
        patt_trans = [pro_lib['tt'], pro_lib['std']]
        pt = transforms.Compose(patt_trans)
        dataset = BlendImageNet(root=imagenet_dir, transform=train_transform, pattern_transform=pt, trigger_label=0,
                                   mode=mode, return_true_label=RTL)
    elif 'wp' in set:
        root_set = set.split('_')[-1]
        data_root = f'/public/czh/repair/datasets/{root_set}/wp/{arch}_{model_seed}/'
        dataset = MyDataset4Misc(data_root, transforms.ToTensor())
    else:
        raise ValueError(f'{set} not support')
    random.seed(seed)
    if num > len(dataset):
        num = len(dataset)
    if seed == 0:
        indices = list(range(0, num))
    else:
        indices = random.sample(list(range(len(dataset))), num)

    return Subset(dataset, indices)


class GTSRB(Dataset):
    def __init__(self, root, train=True, transform=None,):
        self.root = root
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])


        if train:
            csv_path = os.path.join(root, "Train.csv")
        else:
            csv_path = os.path.join(root, "Test.csv")

        df = pd.read_csv(csv_path)

        self.img_paths = list(df["Path"])
        self.class_ids = list(df["ClassId"])

    def __len__(self):
        return len(self.class_ids)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.img_paths[index])
        img = Image.open(img_path)
        label = self.class_ids[index]
        label = torch.tensor(label).long()
        img = self.transform(img)

        return img,label


class Flower:
    def __init__(self, data_root='./dataset/flower_data'):
        super(Flower, self).__init__()

        data_dir = data_root
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        name_json = data_dir + '/cat_to_name.json'
        use_cuda = torch.cuda.is_available()
        # Data Normalization code
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

        valid_transforms = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

        image_datasets = {}
        image_datasets['train'] = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)

        image_datasets['valid'] = torchvision.datasets.ImageFolder(valid_dir, transform=valid_transforms)

        self.train_set = image_datasets['train']
        self.val_set = image_datasets['valid']


if __name__ == '__main__':
    nor_dataset = get_standard(set='imagenet', process=['std'], num=50000, train=True, seed=23)
    backdoor_dataset = get_backdoor(set='imagenet', process=['std'], num=50000, train=True, mode='train', seed=23)

    loader_nor = DataLoader(nor_dataset, batch_size=128, shuffle=True, num_workers=0)
    loader_bd = DataLoader(backdoor_dataset, batch_size=128, shuffle=True, num_workers=0)

    for i, l in loader_nor:
        print(1)
        break
    for i, l in loader_bd:
        print(1)
        break
