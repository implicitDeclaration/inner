import os
import time
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from args import args
from utils.care_eval import CAREVGG13_dense, CAREResNet18, CAREProbeVGG13_dense, CAREVGG16_dense
from utils.utils import *
from model.cifar10_vgg import get_mask_model_vgg, get_mask_vgg_probe, VGG13_dense, get_mask_model_vgg_dense, VGG16_dense
from model.resnet import cResNet18, get_mask_model_res, cResNet18_Dense, get_mask_model_res_probe
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from data.adversarial import get_adv, get_adv_mix
from utils.make_dataset import get_standard, get_backdoor


def get_model(args, rep, device, seed=2022):
    '''

    :return: if per class is true, return the acc of each class
    '''
    accs = []
    num_class = {'gtsrb': 43, 'cifar10': 10, 'imagenet': 10, }

    if 'vgg' in args.arch:
        if args.arch == 'vgg13_dense':
            model = VGG13_dense('VGG13', num_class=num_class[args.set])
        else:
            model = VGG16_dense()
        if rep == 'ai':  #  delete inner if you want ai-lancet original
            ai_result = f'/public/czh/AILancet/result_per_seed/{args.set}_{args.arch}_{args.rep_type}_{seed}.pt'
            ai_result = torch.load(ai_result)
            if args.rep_type == 'adv':
                pretrained_weight = './checkpoints/%s/%s/seed%s/std/model-best.pt' % (
                args.set, args.arch, seed)
            else:
                pretrained_weight = './checkpoints/%s/%s/seed%s/%s/model-best.pt' % (args.set, args.arch, seed, args.rep_type)
            layer, ratio = ai_result['layer'], ai_result['ratio']
            model = get_mask_model_vgg_dense(ratio, seed, layer, pretrained_weight, device, num_class[args.set],
                                             '%s_%s' % (args.rep_type, args.set), args.rep_type)
        elif rep == 'hyb':
            w = f'./checkpoints/Hybrid_2/seed{seed}/{args.arch}_{args.set}_{args.rep_type}/repaired.pt'
            pretrained(w, model, device, [''])
        elif rep == 'care':
            if args.rep_type == 'adv':
                pretrained_weight = './checkpoints/%s/%s/seed%s/std/model-best.pt' % (
                args.set, args.arch, seed)
            else:
                pretrained_weight = './checkpoints/%s/%s/seed%s/%s/model-best.pt' % (args.set, args.arch, seed, args.rep_type)
            care_result = f'/public/czh/care-main/ckpts/{args.set}/{args.arch}/seed{seed}/bd_{args.repair_sample_num}/{args.rep_type}/result.pt'

            care_result = torch.load(care_result)
            rw, ri = care_result['repair weight'], care_result['repair index'].reshape(-1)
            if 'vgg16' in args.arch:
                model = CAREVGG16_dense(rw, ri)
            else:
                model = CAREVGG13_dense(rw, ri, num_class[args.set])
            pretrained(pretrained_weight, model, device, [''])
        elif rep == 'rm':
            prefix ='RM/%s_%s_%s_%s_%s/' % (args.arch, seed, args.set, args.rep_type, args.repair_sample_num)+ 'model-best.pt'
            pretrained_weight = os.path.join(args.rep_dir, prefix)

            pretrained(pretrained_weight, model, device, [''])
        elif rep == 'ours':
            save_dir = os.path.join(args.rep_dir, '%s/%s/seed%s/' %
                                    (args.arch, args.rep_type, seed))

            save_name = f'type_{args.rep_type}-rep_layer_{args.rep_layer_num}-rep_num_{args.repair_sample_num}-ratio_{args.ratio}' \
                        f'-neuron_{args.neuron_num}-probe_{args.probe_train_num}'
            file_name = os.path.join(save_dir, save_name)
            file_name = os.path.join(file_name, 'model-best.pt')
            pretrained(file_name, model, device, [''])
        else:
            if rep == 'adv':
                weight_file = os.path.join(args.save_dir, '%s/seed%s/std/model-best.pt' %
                                           (args.arch, seed))
            else:
                weight_file = os.path.join(args.save_dir, '%s/seed%s/%s/model-best.pt' %
                                           (args.arch, seed, rep))
            pretrained(weight_file, model, device, [''])

    elif args.arch == 'res18_dense':
        model = cResNet18_Dense(num_classes=num_class[args.set])
        if rep == 'ai':
            ai_result = f'/public/czh/AILancet/result_per_seed/{args.set}_{args.arch}_{args.rep_type}_{seed}.pt'
            ai_result = torch.load(ai_result)
            layer, ratio = ai_result['layer'], ai_result['ratio']
            if args.rep_type == 'adv':
                pretrained_weight = './checkpoints/%s/res18_dense/seed%s/std/model-best.pt' % (
                args.set, seed)
            else:
                pretrained_weight = './checkpoints/%s/res18_dense/seed%s/%s/model-best.pt' % (args.set, seed, args.rep_type)
            model = get_mask_model_res(ratio, seed, layer, pretrained_weight, device, num_class[args.set], '%s_%s' % (args.rep_type, args.set), args.rep_type)
        elif rep == 'care':
            if args.rep_type == 'adv':
                pretrained_weight = './checkpoints/%s/res18_dense/seed%s/std/model-best.pt' % (
                    args.set, seed)
            else:
                pretrained_weight = './checkpoints/%s/res18_dense/seed%s/%s/model-best.pt' % (
            args.set, seed, args.rep_type)
            care_result = f'/public/czh/care-main/ckpts/{args.set}/{args.arch}/seed{seed}/bd_1000/{args.rep_type}/result.pt'
            care_result = torch.load(care_result)
            rw, ri = care_result['repair weight'], care_result['repair index'].reshape(-1)
            model = CAREResNet18(rw, ri, num_classes=num_class[args.set])
            pretrained(pretrained_weight, model, device, [''])
        elif rep == 'rm':

            prefix ='RM/%s_%s_%s_%s_%s/' % (args.arch, seed, args.set, args.rep_type, args.repair_sample_num)+ 'model-best.pt'
            pretrained_weight = os.path.join(args.rep_dir, prefix)
            pretrained(pretrained_weight, model, device, [''])
        elif rep == 'hyb':
            w = f'./checkpoints/Hybrid_2/seed{seed}/{args.arch}_{args.set}_{args.rep_type}/repaired.pt'
            pretrained(w, model, device, [''])
        elif rep == 'ours':
            save_dir = os.path.join(args.rep_dir, '%s/%s/seed%s/' %
                                    ('res18_dense', args.rep_type, seed))


            save_name = f'type_{args.rep_type}-rep_layer_{args.rep_layer_num}-rep_num_{args.repair_sample_num}-ratio_{args.ratio}' \
                        f'-neuron_{args.neuron_num}-probe_{args.probe_train_num}'

            file_name = os.path.join(save_dir, save_name)
            file_name = os.path.join(file_name, 'model-best.pt')
            pretrained(file_name, model, device, [''])
        else:
            if rep == 'adv':
                weight_file = os.path.join(args.save_dir, '%s/seed%s/std/model-best.pt' %
                                           ('res18_dense', seed))
            else:
                weight_file = os.path.join(args.save_dir, '%s/seed%s/%s/model-best.pt' %
                                       ('res18_dense', seed, rep))
            pretrained(weight_file, model, device, [''])
    elif args.arch == 'vgg16_dense':
        model = VGG16_dense()
        weight_file = os.path.join(args.save_dir, '%s/seed%s/%s/model-best.pt' %
                                   ('res18_dense', seed, rep))
        if rep == 'rm':
            prefix = '%s_%s_%s_%s_%s/' % (
            args.arch, seed, args.set, args.rep_type, args.repair_sample_num) + 'model-best.pt'
            pretrained_weight = os.path.join(args.rep_dir, prefix)
            pretrained(pretrained_weight, model, device, [''])
    else:
        raise ValueError('arch %s is not included' % args.arch)

    return model


def get_care_mask(args, seed, device):
    num_classes = {'gtsrb': 43, 'cifar10': 10, 'imagenet': 10, }
    care_result = f'/public/czh/care-main/ckpts/{args.set}/{args.arch}/seed{seed}/bd_1000/{args.rep_type}/result.pt'
    if not os.path.exists(care_result):
        raise ValueError(f'{care_result} \nnot exist!')
    care_result = torch.load(care_result)
    neuron = care_result['repair index']
    rw, ri = care_result['repair weight'], care_result['repair index'].reshape(-1)
    index1, weight1 = [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]
    pretrained_file = './checkpoints/%s/%s/seed%s/%s/model-best.pt' % (args.set, args.arch, seed, args.rep_type)
    if args.arch == 'vgg13_dense':
        model1 = CAREVGG13_dense(index1, weight1, nc=num_classes[args.set])
        model2 = CAREVGG13_dense(rw, ri, nc=num_classes[args.set])
        target_layer = 'features5.3'
    elif args.arch == 'vgg16_dense':
        model1 = CAREVGG16_dense(index1, weight1)
        model2 = CAREVGG16_dense(rw, ri)
        target_layer = 'features5.3'
    elif args.arch == 'res18_dense':
        model1 = CAREResNet18(index1, weight1, num_classes=num_classes[args.set])
        model2 = CAREResNet18(rw, ri, num_classes=num_classes[args.set])
    else:
        raise ValueError()
    pretrained(pretrained_file, model1, device, [])
    pretrained(pretrained_file, model2, device, [])

    if args.set == 'imagenet':
        input = torch.rand([1, 3, 224, 224]).to(device)
    else:
        input = torch.rand([1, 3, 32, 32]).to(device)

    feature1 = model1(input, True)
    feature2 = model2(input, True)

    feature2 = feature2.view(feature2.size(0), -1)
    for i in range(len(ri)):
        model2.zero_grad()
        feature2 = model2(input, True)
        id = int(ri[i])
        y = feature2[0, id] * 1
        print(y)
        # f2 = torch.ones_like(feature2).to(device)
        # f3 = torch.ones_like(feature2).to(device)
        # f2 = f2.view(f2.size(0), -1)
        # for i in range(len(ri)):
        #     id = int(ri[i])
        #     f2[:, id] = (rw[i]) * f2[:, id]
        # f2 = f2.reshape(feature2.size())
        # y = torch.sum(feature2 * f2)
        y.backward()
        for n, v in model2.named_parameters():
            if target_layer in n:
                print(n)
                nz_id = torch.nonzero(v.grad)
                print(nz_id)
                nz = torch.count_nonzero(v.grad).item()  # 返回tensor中不为0的数据个数
                print(nz)
                print(v.grad.size())
                break
        print('>>>>>>>')


def layer_probe_per_model(bd_model, visclean_loader, visbd_loader, device, set_label2, target_label, num=20):
    successed = 0
    clayers, blayers = [], []
    for img, label in visclean_loader:
        if set_label2 is not None and label.item() != set_label2:  # filter clean label
            continue
        img, label = img.to(device), label
        cpreds = bd_model(img, True)
        cout = torch.argmax(cpreds[-1], 1)
        # print(f'true label: {label.item()}, clean pred: {cout.item()}')
        if cout.item() == label.item():  # only use right prediction
            successed += 1
        elif target_label == None:
            successed += 1
        else:
            continue
        c_per_img = []
        for l in range(len(cpreds)):
            clayer1 = cpreds[l].cpu().detach().numpy().squeeze()
            # cnorm = np.linalg.norm(clayer1)
            # l2 normalization
            clayer1 = softmax(clayer1)
            c_per_img.append(clayer1)
        clayers.append(c_per_img)
        if successed == num:
            break
    for img, label in visbd_loader:
        if set_label2 is not None and label.item() != set_label2:
            continue
        img, label = img.to(device), label
        bpreds = bd_model(img, True)
        bout = torch.argmax(bpreds[-1], 1)
        # print(f'true label: {label.item()}, bd pred: {bout.item()}')
        if target_label != None and bout.item() == target_label:
            successed += 1
        elif target_label == None:
            successed += 1
        else:
            continue
        b_per_img = []
        for l in range(len(bpreds)):
            blayer1 = bpreds[l].cpu().detach().numpy().squeeze()
            # bnorm = np.linalg.norm(blayer1)
            # l2 normalization
            blayer1 = softmax(blayer1)
            b_per_img.append(blayer1)
        blayers.append(b_per_img)
        if successed == num:
            break
    return clayers, blayers


def get_probe_model(arch, rep, seed, attack, dataset, device):
    if arch == 'vgg13_dense':
        prob_bd, prob_std, PROBE_NUM = [], [], 7
    else:
        prob_bd, prob_std, PROBE_NUM = [], [], 6
    std_file = '/public/czh/repair/checkpoints/cifar10/%s/seed%s/std/model-best.pt' % (arch, seed)
    bd_file = '/public/czh/repair/checkpoints/cifar10/%s/seed%s/%s/model-best.pt' % (arch, seed, attack)
    print(f'{arch} {rep} {seed}')
    p = os.path.join('/public/czh/repair/checkpoints/cifar10', '%s/seed%s/%s/model-best.pt')
    for i in range(1, PROBE_NUM + 1):
        prob_bd.append(
            p % (arch, seed, '%s_probe_num_%s_layer_%s' % (attack, 1000, i)))
        prob_std.append(
            p % (arch, seed, '%s_probe_num_%s_layer_%s' % ('std', 1000, i)))

    if rep == 'care':
        care_result = f'/public/czh/care-main/ckpts/{dataset}/{arch}/seed{seed}/{attack}_{1000}/{attack}/result.pt'
        care_result = torch.load(care_result)
        rw, ri = care_result['repair weight'], care_result['repair index'].reshape(-1)
        print(rw)
        print(ri)
        bd_model = CAREProbeVGG13_dense(rw, ri, num_class=10)
        pretrained(bd_file, bd_model, device, prob_bd)
    elif rep == 'rm':
        prefix = 'RM/%s_%s_%s_%s_%s/' % (
        arch, seed, dataset, attack, 1000) + 'model-best.pt'
        pretrained_weight = os.path.join(f'./checkpoints/repaired/{dataset}', prefix)
        bd_model = VGG13_dense('VGG13')
        pretrained(pretrained_weight, bd_model, device, prob_bd)

    elif rep == 'ai':
        ai_result = f'/public/czh/AILancet/result_per_seed/{dataset}_{arch}_{attack}_{seed}.pt'
        ai_result = torch.load(ai_result)
        layer, ratio = ai_result['layer'], ai_result['ratio']
        if arch == 'vgg13_dense':
            bd_model = VGG13_dense('VGG13' )
            get_mask_vgg_probe(bd_model, prob_bd, ratio, seed, layer, bd_file, device, 10,
                               '%s_%s' % (attack, dataset), attack)
        else:
            bd_model = get_mask_model_res_probe(prob_bd, ratio, seed, layer, bd_file, device, 10,
                                                '%s_%s' % (attack, dataset), attack)
    elif rep == 'ours':
        bd_model = VGG13_dense('VGG13')
        ours_file = f'./checkpoints/repaired/{dataset}/{arch}/{attack}/seed{seed}/1000_0.0_100/type_{attack}-rep_layer_3-rep_num_1000-ratio_0.0-neuron_100-probe_1000-epoch_7/model-best.pt'
        pretrained(ours_file, bd_model, device, prob_bd)
    else:
        if arch == 'res18_dense':
            bd_model = cResNet18_Dense()
            pretrained(bd_file, bd_model, device, prob_bd)
        else:
            bd_model = VGG13_dense('VGG13')
            pretrained(bd_file, bd_model, device, prob_bd)
    bd_model.eval()
    bd_model.to(device)
    return bd_model, PROBE_NUM


def anomaly_select(PROBE_NUM, model, bd_loader, nor_loader, set_label2=None, trigger_label=0, device='cuda:0', NO_normalization=True):
    '''
    :return: anomaly score for each layer
    '''

    model.eval()
    layer_score = []
    conf_dif = []
    with torch.no_grad():
        clayers, blayers = [], []
        successed = 0
        for img, label in nor_loader:
            if set_label2 is not None and label.item() != set_label2:  # filter clean label
                continue
            img, label = img.to(device), label
            cpreds = model(img, True)
            cout = torch.argmax(cpreds[-1], 1)

            if cout.item() == label.item():  # only use right prediction
                successed += 1
            elif trigger_label is None:
                successed += 1
            else:
                continue
            c_per_img = []
            for l in range(len(cpreds)):
                clayer1 = cpreds[l].cpu().detach().numpy().squeeze()
                cnorm = np.linalg.norm(clayer1)
                # l2 normalization
                if NO_normalization:
                    clayer1 = softmax(clayer1)
                else:
                    clayer1 = clayer1 / cnorm
                c_per_img.append(clayer1)
            clayers.append(c_per_img)
            if successed >= 30:
                break

        successed = 0
        for img, label in bd_loader:
            if set_label2 is not None and label.item() != set_label2:
                continue
            img, label = img.to(device), label
            bpreds = model(img, True)
            bout = torch.argmax(bpreds[-1], 1)
            # print(f'true label: {label.item()}, bd pred: {bout.item()}')
            if trigger_label is not None and bout.item() == trigger_label:
                successed += 1
            elif trigger_label is None:
                successed += 1
            else:
                continue
            b_per_img = []
            for l in range(len(bpreds)):
                blayer1 = bpreds[l].cpu().detach().numpy().squeeze()
                bnorm = np.linalg.norm(blayer1)
                # l2 normalization
                if NO_normalization:
                    blayer1 = softmax(blayer1)
                else:
                    blayer1 = blayer1 / bnorm

                b_per_img.append(blayer1)
            blayers.append(b_per_img)
            if successed >= 30:
                break

        clayers, blayers = np.array(clayers), np.array(blayers)
        print(f'clean layers {clayers.shape} bd layers {blayers.shape}')
        from sklearn.metrics import mean_squared_error
        for l in range(PROBE_NUM):
            c_means = np.mean(clayers[:, l], axis=0)
            b_means = np.mean(blayers[:, l], axis=0)
            # c_means, b_means = softmax(c_means), softmax(b_means)
            #conf_dif = clayers[:, l] - blayers[:, l]

            # print(conf_dif)
            ce = cross_entropy(c_means, b_means)
            layer_score.append(ce)
            # layer_score.append(mean_squared_error(c_means, b_means))
    return layer_score


def cross_entropy(y, t):
    delta = 1e-7
    y, t = np.abs(y), np.abs(t)
    return -np.sum(t*np.log(y+delta))


def softmax(f):
    # instead: first shift the values of f so that the highest number is 0:
    f -= np.max(f)  # f becomes [-666, -333, 0]
    return np.exp(f) / np.sum(np.exp(f))



def anomaly_vis(arch, set, seed, rep_type, device, set_label2=None, save_dir='/public/czh/repair/checkpoints/'):
    num_classes = {'cifar10': 10, 'gtsrb': 43, 'imagenet': 10}
    save_dir = save_dir + set
    p = os.path.join(save_dir, '%s/seed%s/%s/model-best.pt')
    if rep_type == 'adv' or rep_type == 'wp':
        prob_type = 'std'
    else:
        prob_type = rep_type
    if arch == 'vgg13_dense':
        prob_w, PROBE_NUM = [], 7
        for i in range(1, PROBE_NUM + 1):
            prob_w.append(
                p % (arch, seed, '%s_probe_num_%s_layer_%s' % (prob_type, 1000, i)))
        bd_model = VGG13_dense(num_class=num_classes[set])
        rep_model = VGG13_dense(num_class=num_classes[set])
    elif arch == 'vgg16_dense':
        prob_w, PROBE_NUM = [], 7
        bd_model = VGG16_dense()
        rep_model = VGG16_dense()
        for i in range(1, PROBE_NUM + 1):
            prob_w.append(
                p % (arch, seed, '%s_probe_num_%s_layer_%s' % (prob_type, 1000, i)))
    else:
        bd_model = cResNet18_Dense(num_classes=num_classes[set])
        rep_model = cResNet18_Dense(num_classes=num_classes[set])
        prob_w, PROBE_NUM = [], 6
        for i in range(1, PROBE_NUM + 1):
            prob_w.append(
                p % (arch, seed, '%s_probe_num_%s_layer_%s' % (prob_type, 1000, i)))

    std_file = '/public/czh/repair/checkpoints/%s/%s/seed%s/std/model-best.pt' % (set, arch, seed)
    bd_file = '/public/czh/repair/checkpoints/%s/%s/seed%s/%s/model-best.pt' % (set, arch, seed, rep_type)
    rep_file = '/public/czh/repair/checkpoints/repaired/%s/%s/%s/seed%s/' \
               'type_%s-rep_layer_3-rep_num_1000-ratio_0.0-neuron_100-probe_1000/' \
               'model-best.pt' % (set, arch, rep_type, seed, rep_type)

    if rep_type == 'adv' or rep_type == 'wp':
        pretrained(std_file, bd_model, device, prob_w)
    else:
        pretrained(bd_file, bd_model, device, prob_w)
    pretrained(rep_file, rep_model, device, prob_w)

    if rep_type == 'adv':
        vis_bd = get_adv_mix(set, args.arch, 1000, ['std'], './datasets/', seed, validate=True)
        val_bd = get_adv_mix(set, args.arch, 1000, ['std'], './datasets/', seed, validate=True)
    else:
        vis_bd = get_backdoor(set='%s_%s' % (rep_type, set), num=1000, train=False, mode='ptest', model_seed=seed,
                              arch=arch,
                              seed=2333, RTL=True, process=['std'])
        val_bd = get_backdoor(set='%s_%s' % (rep_type, set), num=1000, train=False, mode='ptest', model_seed=seed,
                              arch=arch,
                              seed=2333, avoid_trg_class=True, process=['std'])
    rep_clean = get_standard(set=set, num=1000, train=False, seed=2333, process=['std'])
    bd_loader = DataLoader(val_bd, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(rep_clean, batch_size=32, shuffle=True, num_workers=0)
    visbd_loader = DataLoader(vis_bd, batch_size=1, shuffle=True, num_workers=0)
    visclean_loader = DataLoader(rep_clean, batch_size=1, shuffle=True, num_workers=0)
    visclean_loader2 = DataLoader(rep_clean, batch_size=1, shuffle=False, num_workers=0)
    asr1 = validate(bd_model, bd_loader, per_class=False, device=device)
    asr2 = validate(rep_model, bd_loader, per_class=False, device=device)
    print(f'asr befor {asr1} after{asr2}')
    score_before = anomaly_select(PROBE_NUM, bd_model, visbd_loader, visclean_loader, set_label2=set_label2,
                           trigger_label=0, device=device, NO_normalization=True)
    score_after = anomaly_select(PROBE_NUM, rep_model, visbd_loader, visclean_loader, set_label2=set_label2,
                                  trigger_label=None, device=device, NO_normalization=True)
    score_baseline = anomaly_select(PROBE_NUM, bd_model, visclean_loader2, visclean_loader, set_label2=set_label2,
                                  trigger_label=set_label2, device=device, NO_normalization=True)
    print(f'score before: {score_before}')
    print(f'score after: {score_after}')
    print(f'score baseline: {score_baseline}')

    if 'vgg' in arch:
        score_before = score_before[:-1]
        score_after = score_after[:-1]

    for ss in range(len(score_before)):
        if score_before[ss] < 0.1:
            score_before[ss] = 0.1
    for ss in range(len(score_after)):
        if score_after[ss] < 0.1:
            score_after[ss] = 0.1

    font_leg = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14, }
    xx = list(range(1, len(score_before)+1))
    xx = np.array(xx)
    xx_ticks = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6' ]

    plt.bar(xx, score_before, width=0.4, color='#FF4500', label='before repair')
    plt.bar(xx+0.42, score_after, width=0.4, color='#98FB98', label='after repair')
    plt.ylabel('Anomaly Score')
    plt.xticks(xx+0.21, xx_ticks)
    plt.legend(loc='lower left', ncol=1, prop=font_leg) #  bbox_to_anchor=(0.9, 1.05)
    save_dir = f'./as_{set}_{arch}_{rep_type}.pdf'
    plt.savefig(save_dir, bbox_inches='tight')
    sys.exit()


def test_per_seed(args, arch, set, rep_type, rep, device, count_mean_var):
    log_file = './acc_asr4all_seeds.txt'
    log_file = open(log_file, 'a')
    a = arch
    s = set
    r = rep_type
    args.arch = a
    args.set = s
    args.rep_type = r


    val_clean = get_standard(set=s, num=500, train=False, seed=23, process=['std'])
    clean_loader = DataLoader(val_clean, batch_size=64, shuffle=True, num_workers=0)
    # bd_loader = DataLoader(val_bd, batch_size=64, shuffle=True, num_workers=0)
    reps = ['rm', 'care', 'ai']
    # for rep in reps:
    args.save_dir = f'./checkpoints/{s}/'
    args.rep_dir = f'./checkpoints/repaired/{args.set}'
    args.repair_sample_num, args.neuron_num, args.probe_train_num, args.ratio = 1000, 100, 1000, 0.0
    if set == 'imagenet':
        args.repair_sample_num = 100

    log_file.write(f'{rep} {rep_type} {arch}, {set}\n')
    for seed in range(2022, 2032):
        if rep_type == 'adv':
            val_bd = get_adv_mix(args.set, args.arch, 500, ['std'], './datasets/', seed, validate=True)
        else:
            val_bd = get_backdoor(set='%s_%s' % (args.rep_type, s), num=500, train=False, mode='ptest', seed=23,
                                  RTL=False, model_seed=seed, arch=args.arch, avoid_trg_class=True, process=['std'])
        bd_loader = DataLoader(val_bd, batch_size=64, shuffle=True, num_workers=0)

        acc = count_mean_var(a, args, rep, clean_loader, device, seed, False)
        asr = count_mean_var(a, args, rep, bd_loader, device, seed, False)
        if rep_type == 'adv' or rep_type == 'wp':
            asr = 1-asr
        log_file.write(f'{seed} acc: {acc} asr: {asr}\n')
        log_file.flush()


def get_inner_lancet_model(model, seed, set='cifar10', arch='vgg13_dense', rep_type='bd', layer=1, top_nuerons=25000):
    base_dir = f'/public/czh/repair/checkpoints/{set}'
    blocks = ['features1.3', 'features2.3', 'features3.3', 'features4.3', 'features5.3', 'dense1', 'dense2',] # 'classifier'

    def reweight(model, indexes, block_name, weight=-1):
        for name, params in model.named_parameters():  # initialize routing
            if 'bias' in name or 'probe' in name or '_fc' in name:
                continue
            if block_name in name:
                flat_param = params.clone().view(-1)
                for idx in indexes:
                    rep_idx = int(idx)
                    flat_param[rep_idx] *= weight
                flat_param = flat_param.reshape(params.size())
                params.data = flat_param

    neuron_file = os.path.join(base_dir, '%s/seed%s/%s/fault_%s_top%s_%s_%s.npy' % (
        arch, seed, rep_type, blocks[layer], 100, 0.0, 1000))
    if not os.path.exists(neuron_file):
        raise ValueError(neuron_file)
    fault_neuron = np.load(neuron_file, allow_pickle=True)
    fault_neurons = fault_neuron[:top_nuerons]
    reweight(model, fault_neurons, blocks[layer])
    return model


def ailancet_by_inner(model, ori_dict, seed, device, rep_type, set='cifar10', arch='vgg13_dense', top_nuerons=10):
    # load inner version AI-lancet model
    if '_' in set:
        set = set.split('_')[-1]
    base_dir = f'/public/czh/repair/checkpoints/{set}'
    blocks = ['features1.3', 'features2.3', 'features3.3', 'features4.3', 'features5.3', 'dense1', 'dense2',] # 'classifier'
    PROBE_NUM = 7

    def reweight(model, indexes, block_name, weight=-1):
        for name, params in model.named_parameters():  # initialize routing
            if 'bias' in name or 'probe' in name or '_fc' in name:
                continue
            if block_name in name:
                flat_param = params.clone().view(-1)
                for idx in indexes:
                    rep_idx = int(idx)
                    flat_param[rep_idx] *= weight
                flat_param = flat_param.reshape(params.size())
                params.data = flat_param

    val_clean = get_standard(set=args.set, num=400, train=False, seed=23, process=['std'])
    clean_loader = DataLoader(val_clean, batch_size=32, shuffle=True, num_workers=0)
    val_bd = get_backdoor(set='%s_%s' % (args.rep_type, args.set), num=400, train=False, mode='ptest', seed=23,
                          RTL=False, model_seed=seed, arch=args.arch, avoid_trg_class=True, process=['std'])
    bd_loader = DataLoader(val_bd, batch_size=32, shuffle=True, num_workers=0)

    log_file = open('./exchange_ai.txt', 'a')
    score_per_layer = []

    for layer in range(len(blocks)):
        if layer !=1:
            continue

        neuron_file = os.path.join(base_dir, '%s/seed%s/%s/fault_%s_top%s_%s_%s.npy' % (
            arch, seed, rep_type, blocks[layer], 100, 0.0, 1000))
        if not os.path.exists(neuron_file):
            raise ValueError(neuron_file)
        fault_neuron = np.load(neuron_file, allow_pickle=True)
        # print(fault_neuron[:top_nuerons])

        fault_neurons = fault_neuron[:top_nuerons]
        reweight(model, fault_neurons, blocks[layer])
        acc = validate(model, clean_loader, device, per_class=False)
        asr = validate(model, bd_loader, device, per_class=False)
        log_file.write('layer:%s-neuron:%s, acc: %.3f asr: %.3f, score: %.3f\n' % (layer, top_nuerons, acc, asr, acc-asr))
        log_file.flush()
        pretrained(ori_dict, model, device, [])
        score_per_layer.append(np.argmax(acc - asr))

    log_file.write(f'------->seed {seed}:\nbest score at layer:{np.argmax(score_per_layer)}')

