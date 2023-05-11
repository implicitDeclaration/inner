"""
######## for standard model ########
standard accuracy is 0.9077
accuracy of gaussian_noise is 0.46142
######## for repaired model ########
参数设置如下：iters=10，gamma=1000
standard accuracy is 0.8996
accuracy of gaussian_noise is 0.75986
######## for finetune model ########
iter=1
standard accuracy is 0.3753
accuracy of gaussian_noise is 0.34984
"""

import torch
import torchvision
import numpy as np
import os
import sys
import argparse

from Train import Train
from utils.make_dataset import get_standard, get_cifar10C


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--savename', type=str, default="cifar10c_gn_fintune.pth")
    parser.add_argument("--gamma", type=float, default=1000.0)
    args = parser.parse_args()
    return args

# ============================= SET ============================= #
args = get_args()
gpu = args.gpu
save_name = args.savename
iters = args.iters
gamma = args.gamma

to_paired_model_path = "../weight/standard.pth"
model = torch.load(to_paired_model_path, map_location="cpu")
positive_dataset = get_standard(root="../dataset", num=1000)
# backdoor_dataset = get_backdoor(root="../dataset", num=1000)
repair_dataset = get_cifar10C(model=model,root="../dataset/CIFAR-10-C",name="gaussian_noise")
# ============================= SET ============================= #

BACKDOOR_REPAIR = Train(
    model=model,
    clean_test=positive_dataset,
    repair_test=repair_dataset,
    gpu=gpu,
    iters=iters,
    save_name=save_name,
    gamma=gamma
)

BACKDOOR_REPAIR.fine_tune_train()