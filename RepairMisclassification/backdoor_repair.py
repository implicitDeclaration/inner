"""
原始模型的测试精度：在挑选的1000个样本上。
这里精度差异的原因可能是因为在这边加载的时候是有做数据增强处理的，而在测试集上则没有。
clean    accuracy 0.866
repaired accuracy 0.117
在测试集上的结果
standard accuracy is 0.879
backdoor attack success rate is 1.0
############################################
对于使用 grad_mask 之后进行fine-tune
参数设置：gamma=20，iter=1
clean    accuracy 0.842
repaired accuracy 0.988
在测试集上的结果：
standard accuracy is 0.8717
backdoor attack success rate is 0.1016
############################################
对于直接fine-tune
clean    accuracy 0.332
repaired accuracy 0.468
在测试集上的结果
standard accuracy is 0.6503
backdoor attack success rate is 0.0849
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse

from Train import Train
from utils.make_dataset import get_standard, get_backdoor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--savename', type=str, default="standard_finetune.pth")
    parser.add_argument("--gamma", type=float, default=20.0)
    args = parser.parse_args()
    return args


# ============================= SET ============================= #
args = get_args()
gpu = args.gpu
save_name = args.savename
iters = args.iters
gamma = args.gamma

backdoor_model_path = "../weight/backdoor_model.pth"
model = torch.load(backdoor_model_path, map_location="cpu")
positive_dataset = get_standard(root="../dataset", num=1000)
backdoor_dataset = get_backdoor(root="../dataset", num=1000)

# ============================= SET ============================= #

BACKDOOR_REPAIR = Train(
    model=model,
    clean_test=positive_dataset,
    repair_test=backdoor_dataset,
    gpu=gpu,
    iters=iters,
    save_name=save_name,
    gamma=gamma
)

BACKDOOR_REPAIR.fine_tune_train()
