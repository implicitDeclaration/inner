"""
1. fault weight localization;
2. save weight mask as .npy file;
3. different from train step use threshold, here use rate by sort score.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import copy
import os

from model.cifar10_vgg import VGG
from utils.cifarLoader import Subset
from utils.make_dataset import get_standard, get_backdoor


class WeightLocalization:
    """
    get mask for AdamChildOptim
    """

    def __init__(self, model, clean_test, repair_test, gpu):
        self.model = model
        self.clean_test = clean_test
        self.repair_test = repair_test
        self.device = torch.device(gpu)

        self.model.to(self.device)
        self.model.eval()

    def load_data(self, dataset):
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        return loader

    def init_mask(self):
        """
        init a mask with params as key, and a zeros_like params tensor as value
        """
        mask = {}
        for name, params in self.model.named_parameters():
            mask[params] = params.new_zeros(params.size())
        return mask

    def get_mask(self, data_loader):
        self.model.train()
        loss_ce = nn.CrossEntropyLoss()
        mask = self.init_mask()
        N = len(data_loader)

        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            outputs = self.model(data)

            loss = loss_ce(outputs, target)
            loss.backward()

            for name, params in self.model.named_parameters():
                mask[params] += params.grad / N
            self.model.zero_grad()

        return mask

    @torch.no_grad()
    def get_negative_subset(self, dataset):
        self.model.eval()
        # get the classification false sample as a subset
        indices = []
        for index, (data, target) in enumerate(dataset):
            img = data.unsqueeze(0).to(self.device)
            target = torch.tensor(target)
            target = target.to(self.device)

            outputs = self.model(img)
            _, pre = torch.max(outputs, 1)

            if pre != target:
                indices.append(index)
        return Subset(dataset, indices)

    def get_score_mask(self, gamma=1):

        score_mask = self.init_mask()
        clean_loader = self.load_data(self.clean_test)
        clean_mask = self.get_mask(clean_loader)

        repair_subset = self.get_negative_subset(self.repair_test)
        repair_loader = self.load_data(repair_subset)
        repair_mask = self.get_mask(repair_loader)

        for name, params in self.model.named_parameters():
            score_mask[params] = 1 / gamma * (abs(repair_mask[params]) + 1e-6) / (abs(clean_mask[params]) + 1e-6)

        for k in score_mask:
            score_mask[k] = score_mask[k] >= 1
        return score_mask

    def get_rate_mask(self, rate=0.2):
        score_mask = self.init_mask()
        clean_loader = self.load_data(self.clean_test)
        clean_mask = self.get_mask(clean_loader)

        repair_subset = self.get_negative_subset(self.repair_test)
        repair_loader = self.load_data(repair_subset)
        repair_mask = self.get_mask(repair_loader)

        for name, params in self.model.named_parameters():
            score_mask[params] = (abs(repair_mask[params]) + 1e-6) / (abs(clean_mask[params]) + 1e-6)

        r = None
        for k, v in score_mask.items():
            v = v.view(-1).cpu().numpy()
            if r is None:
                r = v
            else:
                r = np.append(r, v)

        polar = np.percentile(r, (1 - rate) * 100)
        for k in score_mask:
            score_mask[k] = score_mask[k] >= polar

        print("Polar => {}".format(polar))

        return score_mask

    def save_mask_npy(self, mask, file_name):
        npy = []
        for p in mask:
            m = mask[p].cpu().numpy()
            npy.append(m)

        npy = np.array(npy)
        np.save(file_name, npy)

    def load_mask_npy(self, file_name):
        a = np.load(file_name, allow_pickle=True)
        return a


def get_mask_file(rate, model, clean_test, bd_test, file_name):
    weight_local = WeightLocalization(
        model=model,
        clean_test=clean_test,
        repair_test=bd_test,
        gpu=0
    )

    mask_rate = weight_local.get_rate_mask(rate=rate)
    print(mask_rate)

    weight_local.save_mask_npy(mask_rate, file_name=file_name)


if __name__ == "__main__":
    a = np.load("./weight_mask_npy/hui_rate_backdoor.npy", allow_pickle=True)
    print(a)
    for i in a:
        print(i.shape)

    # npy_save = "./weight_mask_npy"
    # if not os.path.exists(npy_save):
    #     os.mkdir(npy_save)
    #
    # model_rate = VGG("VGG16")
    # model_rate.load_state_dict(torch.load("../weight_mask/model-00100.pt", map_location="cpu"))
    #
    # model_standard_backdoor = torch.load("../weight/backdoor_model.pth", map_location="cpu")
    #
    # positive_dataset = get_standard(root="../dataset", num=1000)
    # backdoor_dataset = get_backdoor(root="../dataset", num=1000)
    #
    # # for rate model
    # get_mask_file(rate=0.2, model=model_rate, clean_test=positive_dataset, bd_test=backdoor_dataset,
    #               file_name="./weight_mask_npy/hui_rate_backdoor.npy")
    #
    # get_mask_file(rate=0.2, model=model_standard_backdoor, clean_test=positive_dataset, bd_test=backdoor_dataset,
    #               file_name="./weight_mask_npy/standard_backdoor.npy")


