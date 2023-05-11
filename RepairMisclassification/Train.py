import torch
import torch.nn as nn
import time
import numpy as np
import torchvision.transforms as transforms
from torch.optim import Optimizer
import sys
sys.path.append('./RepairMisclassification/')
from MaskOptimizer import ChildTuningAdamW
from torch.utils.data import DataLoader, Dataset
from utils.make_dataset import Subset


class Train:
    def __init__(self, model, clean_test, repair_test, gpu, iters,save_name, gamma=1):
        self.model = model
        self.clean_test = clean_test
        self.repair_test = repair_test
        self.device = torch.device(gpu)
        self.iters = iters
        self.save_name = save_name
        self.gamma = gamma
        self.mask_time = 0
        self.repair_time = 0
        self.model.to(self.device)

    def load_data(self, dataset):
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        return loader

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

    @torch.no_grad()
    def get_positive_subset(self, dataset):
        self.model.eval()
        # get the classification true sample as a subset
        indices = []
        for index, (data, target) in enumerate(dataset):
            img = data.unsqueeze(0).to(self.device)
            target = torch.tensor(target)
            target = target.to(self.device)

            outputs = self.model(img)
            _, pre = torch.max(outputs, 1)

            if pre == target:
                indices.append(index)
        return Subset(dataset, indices)

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
                if params.grad is None:
                    continue
                mask[params] += params.grad / N
            self.model.zero_grad()

        return mask

    def get_score_mask(self):

        score_mask = self.init_mask()
        clean_loader = self.load_data(self.clean_test)
        clean_mask = self.get_mask(clean_loader)

        repair_subset = self.get_negative_subset(self.repair_test)
        repair_loader = self.load_data(repair_subset)
        repair_mask = self.get_mask(repair_loader)

        for name, params in self.model.named_parameters():
            if params.grad is None:
                continue
            score_mask[name] = (1 / self.gamma) * (abs(repair_mask[name]) + 1e-6) / (abs(clean_mask[name]) + 1e-6)

        for k in score_mask:
            score_mask[k] = score_mask[k] >= 1
        return score_mask, repair_loader

    def train(self):
        optimizer = ChildTuningAdamW(
            self.model.parameters(),
        )

        loss_ce = nn.CrossEntropyLoss()

        # test #
        clean_loader = self.load_data(self.clean_test)
        repaired_loader = self.load_data(self.repair_test)

        clean_accuracy = self.eval(clean_loader)
        repaired_accuracy = self.eval(repaired_loader)

        print("clean    accuracy", clean_accuracy)
        print("repaired accuracy", repaired_accuracy)

        for i in range(1, self.iters+1):
            print("iter  {}".format(i))
            score_mask, repair_loader = self.get_score_mask()
            self.model.train()
            if len(repair_loader) == 0:
                print("repair success! len repair loader is 0 ")
                break
            optimizer.set_gradient_mask(score_mask)
            for index,(data, target) in enumerate(repair_loader):
                data,target = data.to(self.device),target.to(self.device)
                outputs = self.model(data)
                loss = loss_ce(outputs, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # test #
            clean_loader = self.load_data(self.clean_test)
            repaired_loader = self.load_data(self.repair_test)

            clean_accuracy = self.eval(clean_loader)
            repaired_accuracy = self.eval(repaired_loader)

            print("clean    accuracy", clean_accuracy)
            print("repaired accuracy", repaired_accuracy)


        torch.save(self.model, self.save_name)

    def get_backdoor_score_mask(self):

        score_mask = self.init_mask()
        clean_loader = self.load_data(self.clean_test)
        clean_mask = self.get_mask(clean_loader)

        repair_loader = self.load_data(self.repair_test)
        repair_mask = self.get_mask(repair_loader)

        for name, params in self.model.named_parameters():
            score_mask[params] = (1 / self.gamma) * (abs(repair_mask[params]) + 1e-6) / (abs(clean_mask[params]) + 1e-6)

        for k in score_mask:
            score_mask[k] = score_mask[k] >= 1
        return score_mask

    def backdoor_train(self, assign_mask=None):
        optimizer = ChildTuningAdamW(
            self.model.parameters(), lr=0.0001
        )

        loss_ce = nn.CrossEntropyLoss()

        # test #
        clean_loader = self.load_data(self.clean_test)
        update_rep_set = self.repair_test
        repair_loader = self.load_data(update_rep_set)
        rep_test_loader = self.load_data(self.repair_test)

        clean_accuracy = self.eval(clean_loader)
        repaired_accuracy = self.eval(repair_loader)

        print("clean    accuracy", clean_accuracy)
        print("repaired accuracy", repaired_accuracy)

        for i in range(1, self.iters + 1):
            print("iter  {}".format(i))
            mask_time_start = time.time()
            if assign_mask:
                score_mask = assign_mask
            else:
                score_mask = self.get_backdoor_score_mask()
            mask_time_end = time.time()
            self.mask_time += mask_time_end - mask_time_start
            self.model.train()

            repair_time_start = time.time()
            optimizer.set_gradient_mask(score_mask)
            for index, (data, target) in enumerate(repair_loader):
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss = loss_ce(outputs, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            update_rep_set = self.get_negative_subset(update_rep_set)
            repair_time_end = time.time()
            self.repair_time += repair_time_end - repair_time_start
            if len(update_rep_set) == 0:
                print("repair success! len repair loader is 0 ")
                break
            # test #
            clean_loader = self.load_data(self.clean_test)
            repair_loader = self.load_data(update_rep_set)

            clean_accuracy = self.eval(clean_loader)
            repaired_accuracy = self.eval(rep_test_loader)

            print("clean    accuracy", clean_accuracy)
            print("repaired accuracy", repaired_accuracy)


        return self.model
        # torch.save(self.model, self.save_name)

    def fine_tune_train(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(),lr=1e-3,betas=(0.9,0.999),eps=1e-6,weight_decay=0.0)
        loss_ce = nn.CrossEntropyLoss()

        # test #
        clean_loader = self.load_data(self.clean_test)
        repaired_loader = self.load_data(self.repair_test)

        clean_accuracy = self.eval(clean_loader)
        repaired_accuracy = self.eval(repaired_loader)

        print("clean    accuracy", clean_accuracy)
        print("repaired accuracy", repaired_accuracy)

        for i in range(1, self.iters + 1):
            print("iter  {}".format(i))
            self.model.train()
            for index, (data, target) in enumerate(repaired_loader):
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss = loss_ce(outputs, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # test #
            clean_loader = self.load_data(self.clean_test)
            repaired_loader = self.load_data(self.repair_test)

            clean_accuracy = self.eval(clean_loader)
            repaired_accuracy = self.eval(repaired_loader)

            print("clean    accuracy", clean_accuracy)
            print("repaired accuracy", repaired_accuracy)
        if self.save_name is not None:
            torch.save(self.model, self.save_name)
        else:
            return self.model

    @torch.no_grad()
    def eval(self, data_loader):
        self.model.eval()
        total = 0
        correct = 0
        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            outputs = self.model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        accuracy = correct / total

        print("Test Accuracy: {}/ {} = {} ".format(correct, total, accuracy))

        self.model.train()

        return accuracy


if __name__ == "__main__":
    pass
