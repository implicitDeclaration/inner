import os
import sys
sys.path.append('~/data/attack_type')
sys.path.append('~/data/')
import torch
import logging
import numpy as np
from PIL import Image
from imageio.v2 import imsave,imread
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
from attack_type.fgsm import FGSM
from attack_type.carlinl2 import CarliniL2
# from data.attack_type.jsma import *
#from data.attack_type.deepfool import DeepFool
from attack_type.onepixel import *
#from data.attack_util import *
from utils.make_dataset import get_standard
from utils.utils import pretrained
from model.cifar10_vgg import VGG13_dense, VGG16_dense
from model.resnet import cResNet18_Dense
import os
# from advertorch.attacks import LocalSearchAttack
import random
import pandas as pd


random_seed = 5566

normalize_mnist = transforms.Normalize((0.1307,), (0.3081,))
normalize_svhn = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
normalize_cifar10 = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
gtsrb_norm = transforms.Normalize([0, 0, 0], [1, 1, 1])
imgset_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def generate_adversarial(model, adv_type, data_type, norm_type, eps, target, device, save_path):
    sourceDataPath = '/public/czh/repair/datasets/' + data_type
    if not os.path.exists(sourceDataPath):
        raise ValueError(f'dataset path >{sourceDataPath}< not exists!')
    if adv_type == 'fgsm':
        genereate_fgsm_samples(model, sourceDataPath, save_path, data_type=data_type,
                               device=device, eps=eps, is_exclude_wr=True, target=target)
    # elif adv_type == 'jsma':
    #     genereate_jsma_samples(model, sourceDataPath, save_path, data_type=data_type,
    #                            max_distortion=0.12, dim_features=dim_features,
    #                            num_out=num_out, device=device, img_shape=img_shape)
    elif adv_type == 'cw':
        genereate_cw_samples(model, sourceDataPath, save_path, data_type=data_type, device=device,
                             c=0.6, iter=1000, batch_size=1)

    # elif adv_type == 'deepfool':
    #     genereate_deepfool_samples(model, sourceDataPath, save_path, data_type=data_type, overshoot=0.02,
    #                                num_out=10, max_iter=50, device=device)

    # elif adv_type == 'localsearch':
    #     genereate_local_search_samples(model, sourceDataPath, save_path, data_type=data_type, device=device)
    #
    elif adv_type == 'op':
        genereate_one_pixel_samples(model, sourceDataPath, save_path, data_type=data_type, device=device)

    else:
        raise Exception("{} is not supported".format(adv_type))


def genereate_one_pixel_samples(model, source_data, save_path, data_type, device="cpu"):
    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True

    if data_type == 'cifar10':
        test_data, channel = load_data_set(data_type, source_data, train=False)
    elif data_type == 'svhn':
        _, test_data = load_svhn(source_data, split=True, normalize=normalize_svhn)

    test_data = exclude_wrong_labeled(model, test_data, device)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    OnePixel = onepixel(model, pixels=5, maxiter=100, popsize=200, target=0, verbose=True, device=device)
    count = 0

    for batch_idx, (input, target) in enumerate(test_loader):

        targets = [0]

        for target_calss in targets:
            if (targets):
                if (target_calss == target[0]):
                    continue

            flag, x, y = OnePixel.attack(input.to(device), target[0])
            if flag == 0:
                continue
            count += flag
            if is_save:
                # save_imgs_tensor([x.to('cpu')], [target], [y], save_path, 'onepixel', no_batch=count,
                #                  channels=3)
                save_imgs_as_npy(x, target, targets, save_path, 'op', no_batch=batch_idx,
                                 batch_size=1, channels=3)
        if count > 5000:
            break


def genereate_fgsm_samples(model, source_data, save_path, eps, is_exclude_wr=True,
                           data_type='cifar10', device="cpu", target=0):
    '''

    :param model_path:
    :param source_data:
    :param save_path:
    :param eps:
    :param is_save:
    :param is_exclude_wr:  exclude the wrong labeled or not
    :return:
    '''

    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True
    if data_type == 'svhn':
        _,test_data = load_svhn(source_data,split=True,normalize=normalize_svhn)
    else:
        test_data, channel = load_data_set(data_type, source_data, train=False)

    test_data = exclude_wrong_labeled(model, test_data, device)
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    fgsm = FGSM(model, eps=eps, device=device, target_type=target)
    adv_samples, y = fgsm.do_craft_target_batch(test_loader, target_label=0, device=device) #
    adv_loader = DataLoader(TensorDataset(adv_samples, y), batch_size=1, shuffle=False)
    succeed_adv_samples = samples_filter(model, adv_loader, "Eps={}".format(eps), target_label=target, device=device)
    num_adv_samples = len(succeed_adv_samples)
    print('successful samples', num_adv_samples)
    if is_save:
        # save_imgs(succeed_adv_samples, TensorDataset(adv_samples, y), save_path, 'fgsm', channel)
        save_imgs_set_as_npy(succeed_adv_samples, TensorDataset(adv_samples, y), save_path, 'fgsm', channel)
    print('Done!')


def genereate_cw_samples(target_model, source_data, save_path,c=0.8, iter=10000, batch_size=1,
                         data_type='cifar10', device='cpu', target=0):
    # at present, only  cuda0 suopport

    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True

    if data_type == 'svhn':
        _,test_data = load_svhn(source_data,split=True,normalize=normalize_svhn)
    else:
        test_data, channel = load_data_set(data_type, source_data, train=False)

    test_data = exclude_wrong_labeled(target_model, test_data, device=device)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    l2Attack = CarliniL2(target_model=target_model, max_iter=iter, c=c, k=0, device=device, targeted=True)
    print("Generating adversarial sampels...")
    for i, data_pair in enumerate(test_loader):
        i += 1
        data, real_label = data_pair
        data, real_label = data.to(device), real_label.to(device)
        scores = target_model(data)
        normal_preidct = torch.argmax(scores, dim=1, keepdim=True)
        target_labels = torch.tensor([target] * real_label.size(0)).to(device)
        adv_samples = l2Attack.do_craft(data, target_labels)  # ***
        success_samples, normal_labels, adv_label = l2Attack.check_adversarial_samples(l2Attack.target_model,
                                                                                       adv_samples, normal_preidct)
        if is_save:
            if isinstance(normal_labels, torch.Tensor):
                # save_imgs_tensor(success_samples, normal_labels, adv_label, save_path, 'cw', no_batch=i,
                #                  batch_size=batch_size, channels=3)
                save_imgs_as_npy(success_samples, normal_labels, adv_label, save_path, 'cw', no_batch=i,
                                 batch_size=batch_size, channels=3)
        logging.info('batch:{}'.format(i))
        if i > 5000:
            break


def load_data_set(data_type, source_data, train=False):
    if data_type == 'mnist':
        data = torchvision.datasets.MNIST(root=source_data, train=train, transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
                normalize_mnist]
        ))
        channels = 1
    elif data_type == 'cifar10':
        data = torchvision.datasets.CIFAR10(root=source_data, train=train,
                                                 transform=torchvision.transforms.Compose(
                                                     [torchvision.transforms.ToTensor(),
                                                         normalize_cifar10]
                                                 ))
        channels = 3
    elif data_type == 'gtsrb':
        data = GTSRB(root=source_data, train=train, transform=torchvision.transforms.Compose(
                                                     [transforms.Resize([32, 32]),
                                                      torchvision.transforms.ToTensor(),
                                                         gtsrb_norm] ))
        channels = 3
    elif data_type == 'imagenet':
        if train:
            imagenet_dir = os.path.join(source_data, 'train')
        else:
            imagenet_dir = os.path.join(source_data, 'val')
        train_transform = torchvision.transforms.Compose(
                                                     [transforms.Resize(256), transforms.CenterCrop(224),
                                                      torchvision.transforms.ToTensor(), imgset_norm] )
        data = torchvision.datasets.ImageFolder(imagenet_dir, train_transform)
        channels = 3
    elif data_type == 'svhn':
        data = torchvision.datasets.ImageFolder(
            source_data,
            transforms.Compose([
                transforms.ToTensor(),
                normalize_svhn,
            ]))
        channels = 3
    else:
        raise Exception('Unknown data source')

    return data, channels


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



def load_svhn(data_path, split, normalize=normalize_svhn):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    train_data = torchvision.datasets.SVHN(data_path,
                                            split='train',
                                            download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                normalize
                                            ]))

    test_data = torchvision.datasets.SVHN(data_path,
                                           split='test',
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               normalize
                                           ]))

    if split:
        return train_data, test_data
    else:
        dataset = ConcatDataset([train_data, test_data])
        return dataset


def samples_filter_simple(model, loader, target_label=0, device='cpu'):
    model.eval()
    model.to(device)
    index = 0
    target_adv_samples = []

    for data, label in loader:
        data, label = data.to(device), label.to(device)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if target_label != label and pred == target_label:
            target_adv_samples.append((index, label.item(), pred.item()))

        index += 1
    return target_adv_samples


def samples_filter(model, loader, name, return_type="adv", target_label=None, size=0, show_progress=False,
                   device='cpu', is_verbose=False, show_accuracy=True):
    '''
    :param model:
    :param loader:
    :param name:
    :param return_type:
    :param use_adv_ground: use the adv label as the desired label
    :param size:
    :param show_progress:
    :param device:
    :param is_verbose:
    :return:
    '''
    assert loader.batch_size == 1
    model.eval()
    test_loss = 0
    correct = 0
    index = 0
    adv_samples = []  # index:pred
    target_adv_samples = []
    normal_sample = []

    total_sample = size if size > 0 else len(loader.dataset)
    model = model.to(device)
    for data_tuple in loader:
        if len(data_tuple) == 2:
            data, target = data_tuple
        elif len(data_tuple) == 3:
            data, target, adv_label = data_tuple
        elif len(data_tuple) == 4:
            data, target, adv_label, file_name = data_tuple

        data, target = data.to(device), target.to(device)

        if is_verbose:
            if len(data_tuple) == 4:
                logging.info(
                    '{}>>>True Label:{},adv_label:{}'.format(file_name.item(), target.item(), adv_label.item()))
            elif len(data_tuple) == 3:
                logging.info('>>>True Label:{},adv_label:{}'.format(target.item(), adv_label.item()))
            else:
                logging.info('>>>True Label:{}'.format(target.item()))
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

        if is_verbose:
            logging.info('Predicted Label:{}<<<'.format(pred.item()))
        rst = pred.eq(target).sum().item()
        correct += rst
        is_adv_success = 0 if rst == 1 else 1  # if the predict label is equal to the target label,then the attack is failed
        #  add target class filter
        if target_label is not None:
            if target != target_label and pred == target_label:
                target_adv_samples.append((index, target.item(), pred.item()))

        if is_adv_success:
            adv_samples.append((index, target.item(), pred.item()))
        else:
            normal_sample.append((index, target.item(), pred.item()))
        index += 1

        if size > 0 and index == size:
            break

        if show_progress:
            sys.stdout.write("\r Test: %d of %d" % (index + 1, total_sample))
            sys.stdout.flush()

    size = size if size > 0 else len(loader.dataset)
    test_loss /= size
    if show_accuracy:
        print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            name, test_loss, correct, size, 100. * correct / size))

    if target_label is not None:
        return target_adv_samples
    if return_type == 'adv':
        return adv_samples
    else:
        return normal_sample


def datasetMutiIndx(dataset, indices):
    x_list = []
    y_list = []
    for idx in indices:
        x, y = dataset[idx]
        old_shape = [d for d in x.size()]
        old_shape.insert(0, 1)
        x = x.view(old_shape)
        x_list.append(x)
        y_list.append(y)
    return torch.utils.data.TensorDataset(torch.cat(x_list, 0), torch.LongTensor(y_list))


def exclude_wrong_labeled(model, dataset, device):
    dataloader = DataLoader(dataset=dataset)
    correct_labeled = samples_filter(model, dataloader, return_type='normal', name='targeted model', device=device)
    return datasetMutiIndx(dataset, [idx for idx, _, _ in correct_labeled])


def save_imgs_as_npy(adv_samples, normal_preds, adv_preds, save_path, file_prefix, no_batch=1, batch_size=1, channels=1,
                     adv_count=-1):
    '''
    The difference from the "save_imgs" is that the adv_samples are image tensors, not the indicies
    :param adv_samples:
    :param normal_preds:
    :param adv_preds:
    :param save_path:
    :param file_prefix:
    :param no_batch: The number of batch. 1-index
    :return:
    '''

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if adv_count == -1:
        adv_count = (no_batch - 1) * batch_size
    for data, normal_pred, adv_pred in zip(adv_samples, normal_preds, adv_preds):
        adv_count += 1
        if isinstance(normal_pred, torch.Tensor):
            normal_pred = normal_pred.item()
        if isinstance(adv_pred, torch.Tensor):
            adv_pred = adv_pred.item()
        # normal_pred = normal_pred.cpu().detach().numpy()
        adv_path = os.path.join(save_path, file_prefix + '_' + str(adv_count) + '_' + str(normal_pred) + '_' + '.npy')
        # label_path = os.path.join(save_path, file_prefix + '_' + str(adv_count) + 'label.npy')
        img = data.squeeze().detach().cpu().numpy()

        np.save(adv_path, img)
       #  np.save(label_path, normal_preds)


def save_imgs_tensor(adv_samples, normal_preds, adv_preds, save_path, file_prefix, no_batch=1, batch_size=1, channels=1,
                     adv_count=-1):
    '''
    The difference from the "save_imgs" is that the adv_samples are image tensors, not the indicies
    :param adv_samples:
    :param normal_preds:
    :param adv_preds:
    :param save_path:
    :param file_prefix:
    :param no_batch: The number of batch. 1-index
    :return:
    '''

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if adv_count == -1:
        adv_count = (no_batch - 1) * batch_size
    for data, normal_pred, adv_pred in zip(adv_samples, normal_preds, adv_preds):
        adv_count += 1
        if isinstance(normal_pred, torch.Tensor):
            normal_pred = normal_pred.item()
        if isinstance(adv_pred, torch.Tensor):
            adv_pred = adv_pred.item()
        adv_path = os.path.join(save_path,
                                file_prefix + '_' + str(adv_count) + '_' + str(normal_pred) + '_' + str(
                                    adv_pred) + '_.png')

        if channels == 1:
            img = data.detach().cpu().numpy()
            img = img.reshape(28, 28)
            imsave(adv_path, img)
        elif channels == 3:
            img = data.squeeze().detach().cpu().numpy()
            img = np.transpose(img, axes=(1, 2, 0))
            imsave(adv_path, img)


def save_imgs_set_as_npy(adv_samples, dataset, save_path, file_prefix, channels=1, batch_size=1, batch_no=0):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    adv_count = batch_size * batch_no
    for idx, normal_pred, adv_pred in adv_samples:
        adv_count += 1
        data, true_label = dataset[int(idx)]
        assert true_label.item() == normal_pred
        adv_path = os.path.join(save_path, file_prefix + '_' + str(adv_count) + '_' + str(true_label.item()) + '_' +'.npy')
        # label_path = os.path.join(save_path, file_prefix + '_' + str(adv_count) + 'label.npy')
        img, label = data.squeeze().cpu().numpy(), true_label.cpu().numpy()
        np.save(adv_path, img)
       #  np.save(label_path, label)


def save_imgs(adv_samples, dataset, save_path, file_prefix, channels=1, batch_size=1, batch_no=0):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    adv_count = batch_size * batch_no
    for idx, normal_pred, adv_pred in adv_samples:
        adv_count += 1
        data, true_label = dataset[int(idx)]
        assert true_label.item() == normal_pred
        adv_path = os.path.join(save_path,
                                file_prefix + '_' + str(adv_count) + '_' + str(true_label.item()) + '_' + str(
                                    adv_pred) + '_.png')
        if channels == 1:
            img = data.cpu().numpy()
            img = img.reshape(28, 28)
            imsave(adv_path, img)
        elif channels == 3:
            img = data.cpu().numpy()
            img = np.transpose(img, axes=(1, 2, 0))
            imsave(adv_path, img)


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, img_mode=None, show_extral_label=False,
                 show_file_name=False, max_size=1000):

        image_list = []
        real_labels = []
        predicted_labels = []
        img_names = []
        data_count = 1
        self.max_size = max_size
        all_files = np.array([img_file for img_file in os.listdir(root)])

        if len(all_files) < self.max_size:
            load_files = all_files
        else:
            np.random.seed(random_seed)
            load_indices = np.random.choice(len(all_files), self.max_size, replace=False)  # not put back
            load_files = all_files[load_indices]

        assert len(load_files) <= self.max_size
        for img_file in load_files:
            if img_file.endswith('.png'):
                img_file_split = img_file.split('_')
                real_label = int(img_file_split[-3])
                predicted_label = int(img_file_split[-2])
                real_labels.append(real_label)
                predicted_labels.append(predicted_label)
                img = imread(root + os.sep + img_file)
                image_list.append(img)
                img_names.append(img_file)

        self.data = np.array(image_list)
        self.labels = np.array(real_labels)
        self.predicted_labels = np.array(predicted_labels)
        self.transform = transform
        self.target_transform = target_transform
        self.show_extral_label = show_extral_label
        self.show_file_name = show_file_name
        self.img_names = img_names
        self.img_mode = img_mode  # ref: https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode=self.img_mode)

        if self.transform is not None:
            img = self.transform(img)

        if self.show_extral_label:
            predict = self.predicted_labels[index]

        if self.target_transform is not None:
            target = self.target_transform(target)
            predict = self.target_transform(predict)

        if self.show_extral_label:
            if self.show_file_name:
                return img, target, predict, self.img_names[index]
            else:
                return img, target, predict
        else:
            return img, target

    def __len__(self):
        return len(self.data)


class MyDatasetNpy(torch.utils.data.Dataset):
    def __init__(self, root, transforms, show_file_name=False, max_size=1000, random_seed=222):

        image_list = []
        real_labels = []
        predicted_labels = []
        img_names = []
        data_count = 1
        self.max_size = max_size
        self.transforms = transforms
        all_npy_files = np.array([img_file for img_file in os.listdir(root) if '.npy' in img_file])
        # label_files = np.array([lf for lf in all_npy_files if 'label' in lf])
        # img_files = np.array(list(set(all_npy_files) - set(label_files)))
       #  print(f'load {len(all_npy_files)} npy files\n')

        if len(all_npy_files) < self.max_size:
            load_imgs = all_npy_files
        else:
            np.random.seed(random_seed)
            load_indices = np.random.choice(len(all_npy_files), self.max_size, replace=False)  # not put back

            load_imgs = all_npy_files[load_indices]

        for img_file in load_imgs:
            img_file_split = img_file.split('_')
            real_label = int(img_file_split[-2])
            predicted_label = int(img_file_split[-3])
            real_labels.append(real_label)
            predicted_labels.append(predicted_label)
            img_path = os.path.join(root, img_file)
            image_list.append(np.load(img_path))

        self.data = image_list
        self.labels = real_labels
        # self.predicted_labels = np.array(predicted_labels)

        self.show_file_name = show_file_name
        self.img_names = img_names
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
        img = self.transforms(img)
        target = torch.tensor(target)
        return img, target

    def __len__(self):
        return len(self.data)


def get_adv_mix(set, arch, num, process, root, seed=2022, validate=False, rtl=True, filter_unsucceessed=False):
    available_attacks = ['fgsm', 'cw',]
    pro_lib = {'flip': transforms.RandomHorizontalFlip(), 'rota': transforms.RandomRotation(30),'crop': transforms.RandomCrop((32, 32)),
               'tt': transforms.ToTensor(), 'jit': transforms.ColorJitter(brightness=0.2)}
    tfs = [pro_lib['tt']]
    for p in process:
        if p == 'std':
            continue
        else:
            tfs.append(pro_lib[p])
    tfs = transforms.Compose(tfs)
    # if set == 'cifar10':
    #      # >>>>>>for npy load>>>>>>
    #     tfs = [pro_lib['tt']]
    #     for p in process:
    #         if p == 'std':
    #             continue
    #         else:
    #             tfs.append(pro_lib[p])
    #     tfs = transforms.Compose(tfs)
    # elif set == 'cifar10':
    #     tfs = [pro_lib['tt']]
    #     for p in process:
    #         if p == 'std':
    #             continue
    #         else:
    #             tfs.append(pro_lib[p])
    #     tfs = transforms.Compose(tfs)
    #     pass
    adv_sets = []

    for aa in available_attacks:
        data_path = os.path.join(root, f'{set}/{arch}_{seed}_{aa}')
        if validate:
            adv_set = MyDatasetNpy(data_path, tfs, max_size=num, random_seed=333)
        else:
            adv_set = MyDatasetNpy(data_path, tfs, max_size=num, random_seed=3333)
        adv_sets.append(adv_set)

    mixed_set = ConcatDataset(adv_sets)
    if validate:
        random.seed(333)
    else:
        random.seed(3333)
    if num > len(mixed_set):
        num = len(mixed_set)
    if seed == 0:
        indices = list(range(0, num))
    else:
        indices = random.sample(list(range(len(mixed_set))), num)

    return Subset(mixed_set, indices)


class Subset(Dataset):
    def __init__(self,dataset,indices):
        super(Subset, self).__init__()
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def get_adv(set, num, process, root, device, model, validate=False, rtl=False, filter_unsucceessed=False):
    pro_lib = {'flip': transforms.RandomHorizontalFlip(), 'rota': transforms.RandomRotation(30),'crop': transforms.RandomCrop((32, 32)),
               'tt': transforms.ToTensor()}

    if set == 'cifar10':
        # normalize_cifar10 = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
        # tfs = [transforms.ToTensor(), normalize_cifar10]
        # for p in process:
        #     if p == 'std':
        #         continue
        #     tfs.append(pro_lib[p])
        # tf = transforms.Compose(tfs)
         # >>>>>>for npy load>>>>>>
        tfs = [pro_lib['tt']]
        for p in process:
            if p == 'std':
                continue
            else:
                tfs.append(pro_lib[p])
        tfs = transforms.Compose(tfs)
    else:
        # todo add tf of other sets
        pass
    # adv_set1 = MyDataset(root, transform=tf, show_extral_label=False)

    adv_set1 = MyDatasetNpy(root, tfs, max_size=10000, random_seed=333)
    raw_adv_loader = DataLoader(adv_set1, batch_size=1, shuffle=False, num_workers=0)
    # print(f'raw data len {len(raw_adv_loader.dataset)}')
    # # >>>>>>
    # cnt = 0
    # for data, target in raw_adv_loader:
    #     cnt+=1
    #     data, target = data.to(device), target.to(device)
    #     print(f'target is {target}')
    #     print(f'if target == 0 {target == 0}')
    #     output = model(data)
    #     pred = output.data.max(1, keepdim=True)[1]
    #     print(f'pred is {pred}')
    #     print(f' pred == target label {pred==0}')
    #     if cnt == 10:
    #         return
    # #>>>>>>
    if filter_unsucceessed:
        target_successed = samples_filter_simple(model, raw_adv_loader, device=device)
        # target_successed = samples_filter(model, raw_adv_loader, return_type='adv', target_label=0,
        #                                   name='targeted attack', device=device)
        print(f'after filtering, adv set length is {len(target_successed)}')
        suc_len = len(target_successed)
        if num > suc_len:
            print(f'>>>warning! adv samples insufficient, needed {num}, got {suc_len}')
        max_num = num if num < suc_len else suc_len  # avoid out of index
        min_num = max_num if num < suc_len else 0
        all_idx = [idx for idx, _, _ in target_successed]
        sample_num = len(all_idx)
        if validate:
            return_idx = all_idx[:max_num]
        else:
            return_idx = all_idx[min_num:sample_num]
        suc_set = datasetMutiIndx(adv_set1, return_idx)
        return suc_set
    else:
        if validate:
            adv_set1 = MyDatasetNpy(root, tfs, max_size=num,
                                    random_seed=222)  # some sample may fail to attack, load more
        else:
            adv_set1 = MyDatasetNpy(root, tfs, max_size=num, random_seed=333)
        return adv_set1


def data_test(model, data_path, device):

    ## _>>>>>following is saving test
    save_path = './datasets/test_/'
    batch_size = 1
    test_data, channel = load_data_set('cifar10', './datasets/cifar10', train=False)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    for i, data_pair in enumerate(test_loader):
        i += 1
        data, real_label = data_pair
        data, real_label = data.to(device), real_label.to(device)
        save_imgs_as_npy(data, real_label, real_label, save_path, 'cw', no_batch=i,
                         batch_size=batch_size, channels=3)
        if i > 10:
            break
    print(f'successfully generate 10 samples on {save_path}\n')



    print('load test 1 ___>>>\n')
    adv_set1 = MyDatasetNpy('./datasets/test_/', transforms.ToTensor(), max_size=100, random_seed=222)
    adv_loader = DataLoader(adv_set1, batch_size=100)
    for img, label in adv_loader:
        print(label)

    ## ->>>>>>

    # data_path = './datasets/%s_%s_%s/' % ('vgg13_dense', 2022, 'cw')
    # print(f'\n <<<<<root is {data_path}>>>>>>\n')
    # adv_set = get_adv('cifar10', 1000, ['std'], data_path, device, model, validate=False)
    # print('<<<<>')
    # print(len(adv_set))
    # adv_set = get_adv('cifar10', 1000, ['std'], data_path, device, model, validate=True)
    # print('<<<<>')
    # print(len(adv_set))
    # adv_loader = DataLoader(adv_set, batch_size=10)
    # for images, labels in adv_loader:
    #     images, labels = images.float().to(device), labels.to(device)
    #     pred = model(images)
    #     pred = torch.argmax(pred, dim=1)
    #     print(f'true label is {labels}')
    #     print(f'pred is {pred}')
    #     break



if __name__ == '__main__':
    gpu = 0
    device = 'cuda:%s' % gpu if torch.cuda.is_available() else 'cpu'
    num_classes = {'gtsrb': 43, 'imagenet': 10, 'cifar10': 10, 'svhn': 10, 'mnist': 10}
    sets = ['gtsrb']  #  'imagenet'
    # clean = get_standard(num=1000, train=False, seed=2023, process=['std'])
    # nor_loader = DataLoader(clean, batch_size=5, shuffle=False, num_workers=0)

    for ss in sets:
        for seed in range(2028, 2032):
            check_dir1 = f'/public/czh/repair/datasets/{ss}/res18_dense_{seed}_fgsm/'
            check_dir2 = f'/public/czh/repair/datasets/{ss}/vgg13_dense_{seed}_fgsm/'
            if os.path.exists(check_dir1) and os.path.exists(check_dir2):
                print(check_dir1 + '\n exits, continue')
                continue

            c_num = num_classes[ss]
            model_file = '/public/czh/repair/checkpoints/%s/%s/seed%s/std/model-best.pt' % (ss, 'res18_dense', seed)
            model = cResNet18_Dense(num_classes=c_num)
            pretrained(model_file, model, device, [])
            model.eval()
            generate_adversarial(model, 'cw', ss, None, eps=0.1, target=0, device=device,
                                 save_path=f'./datasets/{ss}/res18_dense_{seed}_cw/')
            generate_adversarial(model, 'fgsm', ss, None, eps=0.1, target=0, device=device,
                                 save_path=f'./datasets/{ss}/res18_dense_{seed}_fgsm/')

            model_file = '/public/czh/repair/checkpoints/%s/%s/seed%s/std/model-best.pt' % (ss, 'vgg13_dense', seed)
            model = VGG13_dense('VGG13', num_class=c_num)
            pretrained(model_file, model, device, [])
            model.eval()
            generate_adversarial(model, 'cw', ss, None, eps=0.1, target=0, device=device,
                                 save_path=f'./datasets/{ss}/vgg13_dense_{seed}_cw/')
            generate_adversarial(model, 'fgsm', ss, None, eps=0.1, target=0, device=device,
                                 save_path=f'./datasets/{ss}/vgg13_dense_{seed}_fgsm/')

    # following is imagenet adv samples
    # for seed in range(2022, 2032):
    #     model_file = '/public/czh/repair/checkpoints/imagenet/%s/seed%s/std/model-best.pt' % ('vgg16_dense', seed)
    #     model = VGG16_dense()
    #     pretrained(model_file, model, device, [])
    #     model.eval()
    #     generate_adversarial(model, 'cw', 'imagenet', None, eps=0.1, target=0, device=device,
    #                          save_path=f'./datasets/imagenet/vgg16_dense_{seed}_cw/')
    #     generate_adversarial(model, 'fgsm', 'imagenet', None, eps=0.1, target=0, device=device,
    #                          save_path=f'./datasets/imagenet/vgg16_dense_{seed}_fgsm/')

    #     model_file = f'/public/czh/repair/checkpoints/{ss}/res18_dense/seed2022/std/model-best.pt'
    #     model = cResNet18_Dense(num_classes=c_num)
    #     pretrained(model_file, model, device, [])
    #     model.eval()
    #     generate_adversarial(model, 'op', ss, None, eps=0.1, target=0, device=device,
    #                          save_path='/public/czh/repair/datasets/cifar10/res18_dense_2022_op/')
    #     model_file = f'/public/czh/repair/checkpoints/{ss}/vgg13_dense/seed2022/std/model-best.pt'
    #     model = VGG13_dense('VGG13', num_class=c_num)
    #     pretrained(model_file, model, device, [])
    #     model.eval()
    #     generate_adversarial(model, 'op', ss, None, eps=0.1, target=0, device=device,
    #                          save_path='/public/czh/repair/datasets/cifar10/vgg13_dense_2022_op/')

    # data_test(model, '', device)