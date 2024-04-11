import os
import shutil
import time

import torch
import numpy as np
from model.cifar10_vgg import get_mask_model_vgg, get_mask_vgg_probe, VGG13_dense, get_mask_model_vgg_dense, VGG16_dense, get_mask_model_vgg16_dense
from model.resnet import cResNet18, get_mask_model_res, cResNet18_Dense
from utils.make_dataset import get_standard, get_backdoor
from utils.care_eval import CAREVGG13_dense, CAREResNet18, CAREProbeVGG13_dense, CAREVGG16_dense, CAREExchangeVGG13_dense
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.nn.functional as F
import torch.nn as nn

from utils.my_optim import MySGD
from RepairMisclassification.MaskOptimizer import ChildTuningAdamW
from RepairMisclassification.Train import Train
from utils.utils import *
from data.adversarial import get_adv, get_adv_mix
from data.misclassification import get_misclassified
from args import args

from exp import *
gpu = args.gpu
device = 'cuda:%s'%gpu if torch.cuda.is_available() else 'cpu'
print(f'device :{device}')


def get_weight_mask(args, arch, seed, block, block_num, ratio=0.5, total_num=1000, topk=20, bd=True, update=False):

    base_dir = args.save_dir

    if bd:
        neuron_file = os.path.join(base_dir, '%s/seed%s/%s/fault_%s_top%s_%s_%s.npy' % (arch, seed, args.rep_type,  block, topk, ratio, total_num))
    else:
        neuron_file = os.path.join(base_dir, '%s/seed%s/%s/clean_%s_top%s_%s_%s.npy' % (arch, seed, args.rep_type, block, topk, ratio, total_num))
    # print(neuron_file)
    mask_time = None
    # >>>if have saved, load existing file
    if not update and os.path.exists(neuron_file):
        fault_neuron = np.load(neuron_file, allow_pickle=True)
    else:
        mask_time_start = time.time()
        routs = get_routing_per_layer(args, arch, seed, block_num, block, ratio, total_num, tg_label=0, bd=bd)
        id_count = {}
        for img_e in range(100):
            rout = routs[img_e]
            for k in rout.keys():
                rout_abs = rout[k]
                viewed_rout = rout_abs.view(-1)
                over_avg = torch.where(viewed_rout > torch.mean(viewed_rout))
                # value, indice = torch.sort(viewed_rout, descending=True)
                for id in over_avg[0]:
                    idd = str(id.item())
                    if idd not in id_count.keys():
                        id_count[idd] = 1
                    else:
                        id_count[idd] = id_count[idd] + 1
        fault_neuron = []
        # >>>print the top 10 neuron's location
        cnt = 0
        for k in sorted(id_count, key=id_count.__getitem__, reverse=True):
            # print(k, id_count[k])
            fault_neuron.append(int(k))
            cnt += 1
            # if cnt >= topk:
            #     break
            print(len(fault_neuron))
        mask_time_end = time.time()
        if not os.path.exists(os.path.dirname(neuron_file)):
            os.makedirs(os.path.dirname(neuron_file))
        np.save(neuron_file, fault_neuron)
        mask_time = mask_time_end - mask_time_start
    num_classes = {'gtsrb': 43, 'cifar10': 10, 'imagenet': 10}
    if arch == 'vgg13_dense':
        model = VGG13_dense('VGG13', num_class=num_classes[args.set])
    elif arch == 'vgg16_dense':
        model = VGG16_dense()
    elif arch == 'res18_dense':
        model = cResNet18_Dense(num_classes=num_classes[args.set])
    else:
        raise ValueError('')
    masks = {}
    for n, v in model.named_parameters():
        if block in n and 'bias' not in n:
            # mask = torch.zeros_like(v.data)
            mask = v.new_zeros(v.size())
            ori_shape = v.shape
            mask = mask.view(-1)
            for id in fault_neuron:
                mask[id] = 1
            masks[n] = mask.reshape(ori_shape).to(device)

    return masks, mask_time


def single_img_optim(model, model2, routings, images, ori_weight, epochs, probe_layer, block_name, assign_label=None, alpha=0.01):
    criterion_ce = torch.nn.CrossEntropyLoss()
    def CrossEntropy(outputs, targets, temp=3):
        log_softmax_outputs = F.log_softmax(outputs / temp, dim=1)
        softmax_targets = F.softmax(targets / temp, dim=1)
        return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()
    for e in range(epochs):
        outputs = model2(images, probe=True)
        if assign_label is None:
            self_label = torch.argmax(outputs[probe_layer].detach(), dim=1)
        else:
            self_label = assign_label
        # restore weight
        for name, params in model.named_parameters():
            params.data = ori_weight[name].data
        for name, params in model.named_parameters():  # apply routing to parameters
            if 'bias' in name or 'probe' in name or '_fc' in name:
                continue
            if block_name in name:
                params.data = routings[name] * params.data

        routing_pred = model(images, probe=True)
        #loss = torch.FloatTensor([0.]).to(device)
        loss = criterion_ce(routing_pred[probe_layer], self_label)
        #loss += CrossEntropy(routing_pred[probe_layer], outputs[probe_layer])
        model.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():  # update routing with grad
            if block_name in n and 'weight' in n and p.grad is not None and 'probe' not in n and '_fc' not in n:
                grad = p.grad
                routings[n] = routings[n] - 0.009 * grad - alpha * torch.sign(routings[n])

        for k in routings.keys():
            routings[k].data.clamp_(0, 1)

    for k in routings.keys():
        routings[k] = routings[k].cpu().detach()
    return routings


def get_routing_per_layer(args, arch, seed, probe_layer, block_name, ratio, total_num=1000, tg_label=0, bd=False, save=False):
    if args.rep_type == 'adv' or args.rep_type == 'wp':
        model_type = 'std'
        ptype = 'adv'
    else:
        model_type = args.rep_type
        ptype = args.rep_type
    num_classes = {'gtsrb': 43, 'cifar10': 10, 'imagenet': 10}
    # model_type = 'bd' if args.rep_type == 'bd' else 'std'
    p = os.path.join(args.save_dir, '%s/seed%s/%s/model-best.pt')
    if args.arch == 'vgg13_dense':
        prob_w, PROBE_NUM = [], 7
        base_dir = args.save_dir
        w_file = os.path.join(base_dir, '%s/seed%s/%s/model-best.pt' % (args.arch, seed, model_type))
        for i in range(1, PROBE_NUM + 1):
            prob_w.append(
                p % (args.arch, seed, '%s_probe_num_%s_layer_%s' % (ptype, args.probe_train_num, i)))
        model = VGG13_dense('VGG13', num_classes[args.set])
        pretrained(w_file, model, device, prob_w)
        model2 = VGG13_dense('VGG13', num_classes[args.set])  # it's output as routing training label
        pretrained(w_file, model2, device, prob_w)
    elif args.arch == 'vgg16_dense':
        prob_w, PROBE_NUM = [], 7
        base_dir = args.save_dir
        w_file = os.path.join(base_dir, '%s/seed%s/%s/model-best.pt' % (args.arch, seed, model_type))
        for i in range(1, PROBE_NUM + 1):
            prob_w.append(
                p % (args.arch, seed, '%s_probe_num_%s_layer_%s' % (ptype, args.probe_train_num, i)))
        model = VGG16_dense()
        pretrained(w_file, model, device, prob_w)
        model2 = VGG16_dense()
        pretrained(w_file, model2, device, prob_w)
    elif args.arch == 'res18_dense':
        prob_w, PROBE_NUM = [], 6
        base_dir = args.save_dir
        w_file = os.path.join(base_dir, '%s/seed%s/%s/model-best.pt' % (args.arch, seed, model_type))
        model = cResNet18_Dense(num_classes[args.set])
        for i in range(1, PROBE_NUM + 1):
            prob_w.append(
                p % (args.arch, seed, '%s_probe_num_%s_layer_%s' % (ptype, args.probe_train_num, i)))
        pretrained(w_file, model, device, prob_w)
        model2 = cResNet18_Dense(num_classes[args.set])
        pretrained(w_file, model2, device, prob_w)


    clean_num, bd_num = 1000, args.repair_sample_num
    if args.rep_type == 'adv':
        mix_set = get_adv_mix(args.set, args.arch, args.repair_sample_num, ['std'], './datasets/', validate=False)
    else: #if args.rep_type == 'bd':
        mix_set = get_backdoor(set='%s_%s'%(args.rep_type,args.set), num=total_num, train=False, model_seed=seed, arch=args.arch,
                               mode='ptest', seed=233, avoid_trg_class=True, RTL=True, process=['std'])

    mix_loader = DataLoader(mix_set, batch_size=1, shuffle=True, num_workers=0)
    ori_weight = model.state_dict()

    def get_route_avg(loader, save_name, probe_layer, block_name, enhance=False, clean_label=None, trg_label=tg_label, savefile=save):
        routings_all = []
        cnt = 1
        for images, labels in loader:  #  get the routing for every image
            if clean_label is not None:
                if labels.item() != clean_label:
                    continue
            routings = {}
            for name, params in model.named_parameters():  # initialize routing
                if 'bias' in name or 'probe' in name or '_fc' in name:
                    continue
                if block_name in name:
                    mask = torch.ones_like(params).to(device)
                    routings[name] = mask  # torch.autograd.Variable(mask, requires_grad=True)
            images, labels = images.float().to(device), labels.to(device)
            if enhance:
                nimage, nlabel = sample_enhance(images, labels, trg_label, device)

                for img in range(nimage.size(0)):
                    eimg, elabel =  nimage[img].unsqueeze(dim=0), nlabel[img].unsqueeze(dim=0)
                    pred = model2(eimg)
                    pred = torch.argmax(pred, dim=1)
                    if elabel != pred:
                        routings_enhance = single_img_optim(model, model2, routings, images, ori_weight,
                                                            35, probe_layer, block_name, alpha=0.008)
                    else:
                        continue
                    routings_all.append(routings_enhance)
                    if savefile:
                        torch.save(routings_enhance, save_name % (cnt, img))

            else:
                routings = single_img_optim(model, model2, routings, images, ori_weight,
                                            40, probe_layer, block_name, alpha=0.005)

                if savefile:
                    torch.save(routings, save_name % cnt)
                routings_all.append(routings)
            cnt += 1

            if cnt > 100:
                return routings_all

    if bd:
        routings = get_route_avg(mix_loader,  '', probe_layer, block_name, enhance=False, trg_label=0)  # use bd samples, get fault neuron
    else:
        routings = get_route_avg(mix_loader,  '', probe_layer, block_name, clean_label=0, enhance=False, trg_label=0)

    return routings


def count_mean_var(arch, args, rep, loader, device, seed=2022, per_class=True, target_label=None):
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
            if args.rep_type == 'adv' or args.rep_type == 'wp':
                pretrained_weight = './checkpoints/%s/%s/seed%s/std/model-best.pt' % (
                args.set, args.arch, seed)
            else:
                pretrained_weight = './checkpoints/%s/%s/seed%s/%s/model-best.pt' % (args.set, args.arch, seed, args.rep_type)
            layer, ratio = ai_result['layer'], ai_result['ratio']

            if args.exchange:
                print('\nload inner version ai-lancet\n')
                pretrained(pretrained_weight, model, device, [])
                model = get_inner_lancet_model(model, seed)
                model.to(device)
            else:
                if args.arch == 'vgg13_dense':
                    model = get_mask_model_vgg_dense(ratio, seed, layer, pretrained_weight, device, num_class[args.set],
                                                 '%s_%s' % (args.rep_type, args.set), args.rep_type)
                else:
                    model = get_mask_model_vgg16_dense(ratio, seed, layer, pretrained_weight, device, num_class[args.set],
                                                 '%s_%s' % (args.rep_type, args.set), args.rep_type)

        elif rep == 'hyb':
            w = f'./checkpoints/Hybrid_2/seed{seed}/{args.arch}_{args.set}_{args.rep_type}/repaired.pt'
            pretrained(w, model, device, [''])
        elif rep == 'care':
            if args.rep_type == 'adv' or args.rep_type == 'wp':
                pretrained_weight = './checkpoints/%s/%s/seed%s/std/model-best.pt' % (
                args.set, args.arch, seed)
            else:
                pretrained_weight = './checkpoints/%s/%s/seed%s/%s/model-best.pt' % (args.set, args.arch, seed, args.rep_type)
            if args.exchange:
                care_root = '/public/czh/care-main/ckpts/inner/'
                print('load from inner located neurons')
                care_result = care_root + f'{args.set}/{args.arch}/seed{seed}/bd_{args.repair_sample_num}/{args.rep_type}/result.pt'

                care_result = torch.load(care_result)
                rw, ri = care_result['repair weight'], care_result['repair index'].reshape(-1)
                model = CAREExchangeVGG13_dense(rw, ri, num_class[args.set])
                pretrained(pretrained_weight, model, device, [''])
                model.apply_reweight()
            else:
                care_root = '/public/czh/care-main/ckpts/'
                care_result = care_root + f'{args.set}/{args.arch}/seed{seed}/bd_{args.repair_sample_num}/{args.rep_type}/result.pt'

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
            if rep == 'adv' or args.rep_type == 'wp':
                weight_file = os.path.join(args.save_dir, '%s/seed%s/std/model-best.pt' %
                                           (args.arch, seed))
            else:
                weight_file = os.path.join(args.save_dir, '%s/seed%s/%s/model-best.pt' %
                                           (args.arch, seed, rep))
            pretrained(weight_file, model, device, [''])
        acc = validate(model, loader, device, per_class=per_class, nc=num_class[args.set], set_target_label=target_label)
    elif args.arch == 'res18_dense':
        model = cResNet18_Dense(num_classes=num_class[args.set])
        if rep == 'ai':
            ai_result = f'/public/czh/AILancet/result_per_seed/{args.set}_{args.arch}_{args.rep_type}_{seed}.pt'
            ai_result = torch.load(ai_result)
            layer, ratio = ai_result['layer'], ai_result['ratio']
            if args.rep_type == 'adv' or args.rep_type == 'wp':
                pretrained_weight = './checkpoints/%s/res18_dense/seed%s/std/model-best.pt' % (
                args.set, seed)
            else:
                pretrained_weight = './checkpoints/%s/res18_dense/seed%s/%s/model-best.pt' % (args.set, seed, args.rep_type)
            model = get_mask_model_res(ratio, seed, layer, pretrained_weight, device, num_class[args.set], '%s_%s' % (args.rep_type, args.set), args.rep_type)
        elif rep == 'care':
            if args.rep_type == 'adv' or args.rep_type == 'wp':
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
            if rep == 'adv' or args.rep_type == 'wp':
                weight_file = os.path.join(args.save_dir, '%s/seed%s/std/model-best.pt' %
                                           ('res18_dense', seed))
            else:
                weight_file = os.path.join(args.save_dir, '%s/seed%s/%s/model-best.pt' %
                                       ('res18_dense', seed, rep))
            pretrained(weight_file, model, device, [''])
        acc = validate(model, loader, device, per_class=per_class, nc=num_class[args.set], set_target_label=target_label)

    else:
        raise ValueError('arch %s is not included' % args.arch)

    return acc


def sys_test(args, rep, head, log_path='./', seed_range=2032):
    #  systematically test models, including 1.clean test set, 2.bd attack success rate
    #  and test accuracy and attack success rate under perturbations, including:
    #
    num_classes = {'cifar10': 10, 'gtsrb': 43}
    reps = ['ours', 'rm', 'care', 'ai', 'hyb']
    file_path = os.path.join(log_path, f'val_log_{rep}.csv')
    logfile = open(file_path, 'a')
    batch_size = 16 if args.set == 'imagenet' else 64
    trg_class = 0
    if head:
        logfile.write('\n%s\n' % time.strftime("%Y-%m-%d, %H:%M:%S"))
        # colunms = f'arch,method,acc(avg_acc|max_acc|fairness before|fairness after|RF(b-a)),bd(acc|trigger_acc|fairness),ASR(mean|min),enhance(acc_before|acc_after|RB(b-a))'
        colunms = f'arch,method,max_acc(acc|asr|RF|RS),min_asr(acc|asr|RF|RS),mean(acc|asr)'
        logfile.write(colunms + '\n')
    logfile.flush()
    row = '%s,%s-%s-%s-%s,' % (args.arch, rep, args.set, args.repair_sample_num,args.rep_type)

    val_clean = get_standard(set=args.set, num=1000, train=False, seed=23, process=['std'])
    clean_loader = DataLoader(val_clean, batch_size=batch_size, shuffle=True, num_workers=0)

    unrepaired_acc, repaired_acc = [], []
    for seed in range(2022, seed_range):
        repaired_acc_per_seed = count_mean_var(args.arch, args, rep, clean_loader, device, seed)
        unrepaired_acc_per_seed = count_mean_var(args.arch, args, args.rep_type, clean_loader, device, seed)
        unrepaired_acc.append(unrepaired_acc_per_seed)
        repaired_acc.append(repaired_acc_per_seed)

    unrepaired_acc, repaired_acc = np.array(unrepaired_acc), np.array(repaired_acc)
    unrep_fair_per_seed = [np.mean(row) - np.min(row) for row in unrepaired_acc]
    rep_fair_per_seed = [np.mean(row) - np.min(row) for row in repaired_acc]
    unrep_acc_per_seed = np.mean(unrepaired_acc, axis=1)
    rep_acc_per_seed = np.mean(repaired_acc, axis=1)
    arg_max_acc = np.argmax(rep_acc_per_seed)
    print(f'repaired acc : {rep_acc_per_seed} max {rep_acc_per_seed[arg_max_acc]}')

    unrepaired_sens, repaired_sens, repaired_asr = [], [], []
    if args.rep_type == 'adv':
        for seed in range(2022, seed_range):
            val_bd = get_adv_mix(args.set, args.arch, 1000, ['std'], './datasets/', seed, validate=True)
            val_enhance = get_adv_mix(args.set, args.arch, 1000, ['std', 'rota', 'jit', ], './datasets/', seed, validate=True)
            bd_loader = DataLoader(val_bd, batch_size=batch_size, shuffle=True, num_workers=0)
            en_loader = DataLoader(val_enhance, batch_size=batch_size, shuffle=True, num_workers=0)
            repaired_acc_per_seed = count_mean_var(args.arch, args, rep, bd_loader, device, seed, False, 0)
            repaired_sens_per_seed = count_mean_var(args.arch, args, rep, en_loader, device, seed, False)
            unrepaired_sens_per_seed = count_mean_var(args.arch, args, args.rep_type, en_loader, device, seed, False)
            repaired_sens_per_seed, unrepaired_sens_per_seed = \
            1-repaired_sens_per_seed, 1-unrepaired_sens_per_seed
            unrepaired_sens.append(unrepaired_sens_per_seed)
            repaired_asr.append(repaired_acc_per_seed)
            repaired_sens.append(repaired_sens_per_seed)
    else:
        for seed in range(2022, seed_range):
            val_bd = get_backdoor(set='%s_%s' % (args.rep_type, args.set), num=1000, train=False, mode='ptest', seed=23,
                              RTL=False, model_seed=seed, arch=args.arch, avoid_trg_class=True, process=['std'])
            val_enhance = get_backdoor(set='%s_%s' % (args.rep_type, args.set), num=1000, train=False, mode='ptest',
                                       seed=23, RTL=True, model_seed=seed, arch=args.arch, process=['std', 'jit', 'rota'])
            bd_loader = DataLoader(val_bd, batch_size=batch_size, shuffle=True, num_workers=0)
            en_loader = DataLoader(val_enhance, batch_size=batch_size, shuffle=True, num_workers=0)
            repaired_asr_per_seed = count_mean_var(args.arch, args, rep, bd_loader, device, seed, False)
            repaired_sens_per_seed = count_mean_var(args.arch, args, rep, en_loader, device, seed, False)
            unrepaired_sens_per_seed = count_mean_var(args.arch, args, args.rep_type, en_loader, device, seed, False)
            if args.rep_type == 'wp':
                repaired_asr_per_seed, repaired_sens_per_seed, unrepaired_sens_per_seed = \
                    1 - repaired_asr_per_seed, 1 - repaired_sens_per_seed, 1 - unrepaired_sens_per_seed
            repaired_asr.append(repaired_asr_per_seed)
            repaired_sens.append(repaired_sens_per_seed)
            unrepaired_sens.append(unrepaired_sens_per_seed)

    arg_min_asr = np.argmin(repaired_asr)
    print(f'bd data length: {len(val_bd)}')
    # print(unrepaired_sens)
    # print(repaired_sens)
    # print(arg_max_acc)
    print(f'repaired asr : {repaired_asr} min {repaired_asr[arg_min_asr]}')
    max_acc_RF = unrep_fair_per_seed[arg_max_acc] - rep_fair_per_seed[arg_max_acc]
    max_acc_RS = unrepaired_sens[arg_max_acc] - repaired_sens[arg_max_acc]
    min_asr_RF = unrep_fair_per_seed[arg_min_asr] - rep_fair_per_seed[arg_min_asr]
    min_asr_RS = unrepaired_sens[arg_min_asr] - repaired_sens[arg_min_asr]

    row = row + '%.3f|%.3f|%.3f|%.3f,' % (rep_acc_per_seed[arg_max_acc], repaired_asr[arg_max_acc], max_acc_RF, max_acc_RS)
    row = row + '%.3f|%.3f|%.3f|%.3f,' % (rep_acc_per_seed[arg_min_asr], repaired_asr[arg_min_asr], min_asr_RF, min_asr_RS)

    mean_acc = np.mean(rep_acc_per_seed)
    mean_asr = np.mean(repaired_asr)
    row = row + '%.3f|%.3f' % (mean_acc, mean_asr)

    row = row + ',Sens: %.3f|Fair: %.3f' % (np.mean(unrepaired_sens), np.mean(unrep_fair_per_seed))
    logfile.write(row + '\n')
    logfile.flush()


def repair_mid_layers(arch, args, w_file, seed, probe_type):
    global blocks
    num_classes = {'cifar10': 10, 'gtsrb': 43, 'imagenet':10}
    p = os.path.join(args.save_dir, '%s/seed%s/%s/model-best.pt')
    if args.arch == 'vgg13_dense':
        save_dir = os.path.join(args.rep_dir, '%s/%s/seed%s/' %
                                ('vgg13_dense', args.rep_type, seed))
        prob_w, PROBE_NUM = [], 7
        for i in range(1, PROBE_NUM + 1):
            prob_w.append(
                p % (args.arch, seed, '%s_probe_num_%s_layer_%s' % (probe_type, args.probe_train_num, i)))

        blocks = ['features1.3', 'features2.3', 'features3.3', 'features4.3', 'features5.3', 'dense1', 'dense2', 'classifier']
        model = VGG13_dense('VGG13', num_class=num_classes[args.set])
        pretrained(w_file, model, device, prob_w)
    elif args.arch == 'vgg16_dense':
        save_dir = os.path.join(args.rep_dir, '%s/%s/seed%s/' %
                                ('vgg16_dense', args.rep_type, seed))
        prob_w, PROBE_NUM = [], 7
        for i in range(1, PROBE_NUM + 1):
            prob_w.append(
                p % (args.arch, seed, '%s_probe_num_%s_layer_%s' % (probe_type, args.probe_train_num, i)))

        blocks = ['features1.3', 'features2.3', 'features3.3', 'features4.3', 'features5.3', 'dense1', 'dense2', 'classifier']
        model = VGG16_dense()
        pretrained(w_file, model, device, prob_w)
    elif args.arch == 'res18_dense':
        save_dir = os.path.join(args.rep_dir, '%s/%s/seed%s/' %
                                ('res18_dense', args.rep_type, seed))
        model = cResNet18_Dense(num_classes=num_classes[args.set])
        blocks = ['layer1.1.conv2', 'layer2.1.conv2', 'layer3.1.conv2', 'layer4.1.conv2', 'fc1', 'fc2']
        prob_w, PROBE_NUM = [], 6
        for i in range(1, PROBE_NUM + 1):
            prob_w.append(
                p % (args.arch, seed, '%s_probe_num_%s_layer_%s' % (probe_type, args.probe_train_num, i)))
        pretrained(w_file, model, device, prob_w)
    else:
        raise ValueError(f'arch <{arch}> not supported')

    #  get x and x', e.g., backdoor sample and corresponding normal sample
    if args.rep_type == 'adv':
        eps = 0.05
        data_path = './datasets/%s_%s_%s/' % (arch, seed, args.attack_type)
        rep_clean = get_standard(set=args.set, num=args.repair_sample_num, train=False, seed=233, process=['std'])
        rep_bd = get_adv_mix(args.set, args.arch, args.repair_sample_num, ['std'], './datasets/', validate=False)
        rep_en_bd = get_adv_mix(args.set, args.arch, args.repair_sample_num, ['std', 'rota', 'jit', ], './datasets/', validate=False)
        rep_en_clean = get_standard(set=args.set, num=1000, train=False, seed=233, process=['std', 'rota', 'jit', ])
        val_nor = get_standard(set=args.set, num=1000, train=False, seed=23, process=['std'])
        val_bd = get_adv_mix(args.set, args.arch, args.repair_sample_num, ['std'], './datasets/', validate=True)
        val_en = get_adv_mix(args.set, args.arch, args.repair_sample_num, ['std', 'rota', 'jit', ], './datasets/', validate=True)

    else:

        rep_clean = get_standard(set=args.set, num=args.repair_sample_num, train=False, seed=233, process=['std'])
        rep_bd = get_backdoor(set='%s_%s'%(args.rep_type,args.set), num=args.repair_sample_num, train=False, model_seed=seed, arch=args.arch,
                              mode='ptest', seed=233, avoid_trg_class=True, RTL=True, process=['std'])
        rep_en_bd = get_backdoor(set='%s_%s'%(args.rep_type,args.set), num=args.repair_sample_num, train=False,model_seed=seed, arch=args.arch,
                                 mode='ptest', seed=233, avoid_trg_class=False, RTL=True,
                                 process=['std', 'rota', 'crop', 'flip'])
        rep_en_clean = get_standard(set=args.set, num=1000, train=False, seed=233, process=['std', 'rota', 'jit' ])
        val_nor = get_standard(set=args.set, num=1000, train=False, seed=23, process=['std'])

        val_bd = get_backdoor(set='%s_%s'%(args.rep_type,args.set), num=1000, train=False, mode='ptest', model_seed=seed, arch=args.arch,
                              seed=23, RTL=False, process=['std'])
        val_en = get_standard(set=args.set, num=1000, train=False, seed=23, process=['std', 'rota', 'jit'])

    #  for repair
    batch_size = 8 if args.set == 'imagenet' else 64
    bd_loader = DataLoader(rep_bd, batch_size=batch_size, shuffle=False, num_workers=0)
    nor_loader = DataLoader(rep_clean, batch_size=batch_size, shuffle=False, num_workers=0)
    bd_loader2 = DataLoader(rep_bd, batch_size=1, shuffle=False, num_workers=0)
    nor_loader2 = DataLoader(rep_clean, batch_size=1, shuffle=False, num_workers=0)
    en_loader = DataLoader(rep_en_clean, batch_size=batch_size, shuffle=False, num_workers=0)
    en_bd_loader = DataLoader(rep_en_bd, batch_size=batch_size, shuffle=False, num_workers=0)
    #  for validation
    val_loader = DataLoader(val_nor, batch_size=batch_size, shuffle=True, num_workers=0)
    vbd_loader = DataLoader(val_bd, batch_size=batch_size, shuffle=True, num_workers=0)
    en_val = DataLoader(val_en, batch_size=batch_size, shuffle=True, num_workers=0)

    all_keys = model.state_dict().keys()
    model_keys = [k for k in all_keys if 'probe' not in k]
    target_label = None if args.rep_type == 'wp' else 0
    layer_scores = anomaly_select(PROBE_NUM, model, bd_loader2, nor_loader2, set_label2=None, trigger_label=target_label, device=device)
    args.rep_layer_num = min(PROBE_NUM, args.rep_layer_num)
    score_index = np.argsort(-np.array(layer_scores))[:args.rep_layer_num]  # descending
    log_file = open('./repv3_%s_%s_%s_%s_log.txt' % (arch, seed, args.set, args.rep_type), 'w')
    log_file.write(f'\n {time.strftime("%Y-%m-%d, %H:%M:%S")} total repair sample num: {args.repair_sample_num} '
                   f'mask ratio: {args.ratio} neuron num: {args.neuron_num} probe train num {args.probe_train_num}\n')
    epoch = 20
    acc_hist = [0]
    rep_time_all = 0
    mask_time_all = 0
    satisfied = False
    pc_acc = validate(model, val_loader, per_class=True, device=gpu, nc=num_classes[args.set])
    std_acc_before = validate(model, val_loader, device=gpu, nc=num_classes[args.set])
    bd_acc = validate(model, vbd_loader, device=gpu, nc=num_classes[args.set])
    bd_acc = 1-bd_acc if args.rep_type == 'adv' or args.rep_type == 'wp' else bd_acc
    prob_acc = probe_val(model, val_loader, PROBE_NUM, device)

    # score_index = [3,4,5]
    if args.rep_type == 'blend':
        satisfy_bd_asr = 0.25
    elif args.rep_type == 'wanet' or  args.rep_type == 'adv':
        satisfy_bd_asr = 0.22
    else:
        satisfy_bd_asr = 0.15

    log_file.write(
        f'before repair, clean acc: {std_acc_before}, class0: {pc_acc[0]}  asr: {bd_acc} '
        f'probe acc : {prob_acc} score index: {score_index}\n')
    print(
        f'before repair, clean acc: {std_acc_before}, class0: {pc_acc[0]}  asr: {bd_acc} '
        f'probe acc : {prob_acc} score index: {score_index}\n')
    log_file.flush()
    for e in range(epoch):
        for b in score_index:
            mask_bd, mask_time = get_weight_mask(args, arch, seed, blocks[b], b, ratio=args.ratio,
                                                 total_num=args.repair_sample_num, topk=args.neuron_num, bd=True, update=False)
            rep_time2_start = time.time()
            layer_rep(model, arch, bd_loader, nor_loader, b, [blocks[b]], num_class=num_classes[args.set], mask=mask_bd)
            rep_time2_end = time.time()
            log_file.write(f'block {b}, repair time2 {rep_time2_end- rep_time2_start}\n')
            rep_time_all += rep_time2_end - rep_time2_start
        #log_file.write('>>>>>>>bd repair\n')
        pc_acc = validate(model, val_loader, per_class=True, device=gpu, nc=num_classes[args.set])
        std_acc = validate(model, val_loader, device=gpu, nc=num_classes[args.set])
        bd_acc = validate(model, vbd_loader, device=gpu, nc=num_classes[args.set])
        bd_acc = 1 - bd_acc if args.rep_type == 'adv' or args.rep_type == 'wp'  else bd_acc
        log_file.write(
            f'epoch {e}, normal set acc is {std_acc}, class0 {pc_acc[0]}  attack success rate is {bd_acc}\n')
        log_file.flush()
        #if bd_acc < 0.15:
        acc_hist.append(pc_acc[0])
        if std_acc >= std_acc_before-0.1 and bd_acc < satisfy_bd_asr:
            satisfied = True
        if satisfied:
            save_name = f'type_{args.rep_type}-rep_layer_{args.rep_layer_num}-rep_num_{args.repair_sample_num}-ratio_{args.ratio}' \
                        f'-neuron_{args.neuron_num}-probe_{args.probe_train_num}'
            save_model_part(model, model_keys, save_name, out_dir=save_dir, acc_rec=None)
            break
        # save_model_part(model, model_keys, save_name, out_dir=save_dir, acc_rec=None)
        ### ——————————————————————————————————————————
        epoch2 = 2
        if args.rep_type == 'blend':
            epoch2 = 1
        if args.rep_type == 'wanet':
            epoch2 = 4
        for e2 in range(epoch2):
            for b in score_index:
                mask_bd, mask_time = get_weight_mask(args, arch, seed, blocks[b], b, ratio=args.ratio,
                                                 total_num=args.repair_sample_num, topk=args.neuron_num, bd=True, update=False)
                # mask_rob, mask_time = get_weight_mask(arch, seed, blocks[b], b, topk=100, bd=False, rob=True, update=False)
                rep_time3_start = time.time()
                layer_rep(model, arch, bd_loader, en_bd_loader, b, [blocks[b]], mask=mask_bd, num_class=num_classes[args.set], clean=True)
                rep_time3_end = time.time()
                log_file.write(f'block {b}, repair time3 {rep_time3_end - rep_time3_start}\n')
                rep_time_all += rep_time3_end - rep_time3_start
        en_acc = validate(model, en_val, per_class=True, device=gpu, nc=num_classes[args.set])
        log_file.write(f'enhance set acc is {np.mean(en_acc)}\n')

        std_acc = validate(model, val_loader, device=gpu, nc=num_classes[args.set])
        bd_acc = validate(model, vbd_loader, device=gpu, nc=num_classes[args.set])
        bd_acc = 1 - bd_acc if args.rep_type == 'adv' or args.rep_type == 'wp'  else bd_acc
        log_file.write(
            f'epoch {e}, normal set acc is {std_acc},  attack success rate is {bd_acc}\n')
        log_file.flush()

        if std_acc >= std_acc_before-0.1 and bd_acc < satisfy_bd_asr:
            satisfied = True
        if satisfied:
            save_name = f'type_{args.rep_type}-rep_layer_{args.rep_layer_num}-rep_num_{args.repair_sample_num}-ratio_{args.ratio}' \
                        f'-neuron_{args.neuron_num}-probe_{args.probe_train_num}'
            save_model_part(model, model_keys, save_name, out_dir=save_dir, acc_rec=None)
            break
        log_file.flush()
    save_name = f'type_{args.rep_type}-rep_layer_{args.rep_layer_num}-rep_num_{args.repair_sample_num}-ratio_{args.ratio}' \
                f'-neuron_{args.neuron_num}-probe_{args.probe_train_num}'
    save_model_part(model, model_keys, save_name, out_dir=save_dir, acc_rec=None)
    log_file.write(f'>>>>>>>>>>>overall repair time : {rep_time_all}')
    log_file.flush()


def probe_train(model, prober_num, nor_loader):
    criterion = nn.CrossEntropyLoss()
    model.train()
    optimizers = []
    for i in range(1, prober_num+1):  # only the parameter of probe are optimized
        param = [p[1] for p in model.named_parameters() if 'probe%s' % i in p[0]]
        optimizers.append(torch.optim.Adam(param, lr=0.01))

    overall_acc = []
    for img, label in nor_loader:
        img, label = img.float().to(device), label.to(device)
        outputs = model(img, probe=True)
        one_iter_acc = []
        for i in range(prober_num):  # there are several probes, I backward them one by one
            loss = criterion(outputs[i], label)
            optimizers[i].zero_grad()
            loss.backward(retain_graph=(i < prober_num-1))  # don't retain in the last iter
            optimizers[i].step()
            one_iter_acc.append(Accuracy(outputs[i], label)[0].cpu().detach().numpy())
        overall_acc.append(one_iter_acc)

    return np.mean(overall_acc, axis=0)


def label_align(loader_bd, loader_clean, num_class): # assume loader1 is smaller
    '''

    :param loader1:
    :param loader2:
    :param num_class:
    :return: return a new loader with images has the identical labels of loader1 at each location
    '''
    loader1 = DataLoader(loader_bd.dataset, batch_size=1, shuffle=False)
    loader2 = DataLoader(loader_clean.dataset, batch_size=1, shuffle=False)

    new_loader = []

    idx_record = [0 for i in range(num_class)]
    for img, label in loader1:
        restart = False
        #  enumerate loader2 to find the image with the same label
        for idx, data_pair in enumerate(loader2):
            if data_pair[1] == label and idx > idx_record[label.item()]:
                new_loader.append((data_pair[0].squeeze(), data_pair[1].item()))
                idx_record[label.item()] = idx
                break
            elif idx == len(loader2)-1:
                idx_record[label.item()] = 0
                restart = True
            else:
                continue
        if restart:  # if some images are exhausted, restart from the beginning to replenish the newloader
            for idx, data_pair in enumerate(loader2):
                if data_pair[1] == label and idx > idx_record[label.item()]:
                    new_loader.append((data_pair[0].squeeze(), data_pair[1].item()))
                    idx_record[label.item()] = idx
                    restart = False
                    break
                elif idx == len(loader2) - 1:
                    raise ValueError(f'{label} not exist in loader2 !!!')
                else:
                    continue

    return torch.utils.data.DataLoader(new_loader, batch_size=loader_bd.batch_size, shuffle=False)


def layer_rep(model, arch, bd_loader, nor_loader, probe_num, layer_list, mask, num_class, clean=False):
    '''

    :param model: model
    :param bd_loader: data loader of back-doored images
    :param nor_loader: data loader of normal images
    :param bs: batch size
    :param
    :param layer_list: the list of actual indexes of conv layers in the model
    :return:
    '''
    model.train()
    alpha = 0.8
    beta = 1
    selected_param = []
    for l in layer_list:
        selected_param += get_target_param(model, l, arch=arch)

    if not selected_param:  # avoid empty list
        raise ValueError(f'for {layer_list}, got empty param')
    def CrossEntropy(outputs, targets, temp=3):
        log_softmax_outputs = F.log_softmax(outputs / temp, dim=1)
        softmax_targets = F.softmax(targets / temp, dim=1)
        return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

    criterion = nn.CrossEntropyLoss()

    if mask is not None:

        optimizer = MySGD(selected_param, lr=0.0007, assign_mask=mask[layer_list[0]+'.weight'], device=device)
    else:
        optimizer = torch.optim.Adam(selected_param, lr=0.01)
    loss_sum = []
    # for bd_image, bd_label in bd_loader:
    #     bd_image, bd_label = bd_image.float().to(device), bd_label.to(device)
    if probe_num == -1:
        if mask is not None:
            # optimizer = ChildTuningAdamW(selected_param, lr=0.01)
            # optimizer.set_gradient_mask(mask)
            optimizer = MySGD(selected_param, lr=0.0007, assign_mask=mask[layer_list[0]+'.weight'], device=device)
        else:
            optimizer = torch.optim.SGD(selected_param, lr=0.001)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for image, label in bd_loader:  # nor_loader: # ***
            image, label = image.float().to(device), label.to(device)
            nimage, nlabel = sample_enhance(image, label, 0, device)
            image = torch.cat((image, nimage), dim=0)
            label = torch.cat((label, nlabel), dim=0)
            nor_out = model(image)
            loss = torch.FloatTensor([0.]).to(device)

            a = criterion(nor_out, label)  # guide of label
            loss += a
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model
    if clean:
        for image, label in nor_loader:  # nor_loader: # ***
            image, label = image.float().to(device), label.to(device)

            nor_out = model(image)
            loss = torch.FloatTensor([0.]).to(device)
            a = criterion(nor_out, label)  # guide of label
            loss += a
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        # new_loader = label_align(bd_loader, nor_loader, num_class)
        # execute in the same loop algorithm.2
        for bd, nor in zip(bd_loader, nor_loader):
            nor_image, bd_image = nor[0].float().to(device), bd[0].float().to(device)
            nor_label, bd_label = nor[1].to(device), bd[1].to(device)
            nor_batch_size = nor_image.size(0)
            bd_batch_size = bd_image.size(0)
            if nor_batch_size > bd_batch_size:
                nor_image = nor_image[:bd_batch_size]
            nor_out = model(nor_image, probe=True)
            nor_out = [n.detach() for n in nor_out]
            bd_out = model(bd_image, probe=True)
            loss = torch.FloatTensor([0.]).to(device)
            a = criterion(bd_out[probe_num], bd_label) * alpha  # guide of label
            loss += a
            b = CrossEntropy(bd_out[probe_num], nor_out[probe_num]) * (1 - alpha)  # guide of probe distribution
            loss += b
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def train_only_probe(args, seed, epochs=50):
    probe_save_dir = os.path.join(args.save_dir, '%s/seed%s' % (args.arch, seed))
    if not os.path.exists(probe_save_dir):
        print('\n ------->new dir <%s>\n' % probe_save_dir)
        os.makedirs(probe_save_dir)
    num_classes = {'gtsrb': 43, 'imagenet': 50, 'cifar10': 10, 'svhn': 10, 'mnist': 10}
    c_num = num_classes[args.set]
    if 'res18_dense' in args.arch:
        model = cResNet18_Dense(num_classes=c_num)
    elif 'vgg13' in args.arch:
        model = VGG13_dense('VGG13', num_class=c_num)
    else:
        model = VGG16_dense()
    rep_clean = get_standard(set=args.set, num=args.probe_train_num, train=False, seed=23, process=['std'])
    clean_loader = DataLoader(rep_clean, batch_size=128, shuffle=True, num_workers=0)

    if args.rep_type == 'adv':
        model_file = os.path.join(args.save_dir, '%s/seed%s/%s/model-best.pt') % (args.arch, seed, 'std')
    else:
        model_file = os.path.join(args.save_dir, '%s/seed%s/%s/model-best.pt') % (args.arch, seed, args.rep_type)
    pretrained(model_file, model, device, [''])

    if 'vgg' in args.arch:
        PROBE_NUM = 7
    else:
        PROBE_NUM = 6
    print('>>>>>><<<<<<<')
    probe_keys, acc_records = [], {}
    for i in range(1, PROBE_NUM + 1):
        probe_keys.append([k for k in model.state_dict().keys() if 'probe%s' % i in k])
        acc_records['probe%s' % i] = [0]
    for e in range(epochs):
        acc = probe_train(model, PROBE_NUM, clean_loader)
        for i in range(1, PROBE_NUM + 1):
            probe_dir = '%s_probe_num_%s_layer_%s' % (args.rep_type, args.probe_train_num, i)
            # probe_dir = p % (args.arch, seed, '%s%s_probe_%s_%s' % (args.arch, args.rep_type, args.probe_train_num, i))
            save_model_part(model, probe_keys[i - 1], probe_dir, out_dir=probe_save_dir,
                            acc_rec=acc_records['probe%s' % i])
            acc_records['probe%s' % i].append(acc[i - 1])


def train_models(arch, args, seed=2222, epochs=100):
    save_dir = os.path.join(args.save_dir, '%s/seed%s' % (arch, seed))
    if not os.path.exists(save_dir):
        print('\n new dir <%s>\n' % save_dir)
        os.makedirs(save_dir)
    log_file = open(os.path.join(save_dir[:-8], '%s_%s_%s_log.txt' % (arch, args.set, seed)), 'a')
    log_info = '%s, seed %s, ' % (arch, seed)
    num_classes = {'gtsrb': 43, 'imagenet': 10, 'cifar10': 10, 'svhn': 10, 'mnist': 10}

    c_num = num_classes[args.set]
    if 'res18_dense' in arch:
        model = cResNet18_Dense(num_classes=c_num)
    elif 'vgg13_dense' in arch:
        model = VGG13_dense('VGG13', num_class=c_num)
    else:
        model = VGG16_dense()
    model.to(device)
    bs = 16 if args.set == 'imagenet' else 128

    nor_dataset = get_standard(set=args.set, process=['std'], num=50000, train=True, seed=23)
    backdoor_dataset = get_backdoor(set='%s_%s' % (args.rep_type, args.set), process=['std'], num=50000, train=True, mode='train', seed=23)
    nor_val = get_standard(set=args.set, process=['std'], num=2000, train=False, seed=23)
    bd_val = get_backdoor(set='%s_%s' % (args.rep_type, args.set), process=['std'], num=2000, train=False, mode='ptest', seed=23)

    loader_nor = DataLoader(nor_dataset, batch_size=bs, shuffle=True, num_workers=0)
    loader_bd = DataLoader(backdoor_dataset, batch_size=bs, shuffle=True, num_workers=0)
    val_loader_nor = DataLoader(nor_val, batch_size=bs, shuffle=True, num_workers=0)
    val_loader_bd = DataLoader(bd_val, batch_size=bs, shuffle=True, num_workers=0)


    if 'res' in args.arch:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=0.0005)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.008, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().cuda()

    all_keys = model.state_dict().keys()
    model_keys = [k for k in all_keys if 'probe' not in k]
    if args.continue_train:
        w = os.path.join(save_dir, 'std/model-best.pt')
        pretrained(w, model, device, [])
        print(f'---->>>continue from {w}')
    else:
        #  train normal model
        acc_hist = []
        for e in range(epochs):
            acc = std_trainer(model, loader_nor, criterion, optimizer, e)
            acc_hist.append(acc)
            save_model_part(model, model_keys, 'std', out_dir=save_dir, acc_rec=acc_hist)

    std_acc = validate(model, val_loader_nor, device=gpu, nc=c_num)
    log_info += 'std acc %s, ' % std_acc

    if 'res' in args.arch:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0008)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    acc_hist = []
    model.train()
    for e in range(15):
        acc = std_trainer(model, loader_bd, criterion, optimizer, e)
        acc_hist.append(acc)
        save_model_part(model, model_keys, args.rep_type, out_dir=save_dir, acc_rec=acc_hist)
    bd_acc = validate(model, val_loader_nor, device=gpu, nc=c_num)
    log_info += 'bd acc on nor set %s, ' % bd_acc
    bd_acc = validate(model, val_loader_bd, device=gpu, nc=c_num)
    log_info += 'bd attack success rate %s \n ' % bd_acc
    log_file.write(log_info)
    log_file.flush()


def rm_repair(args, w_file, seed):
# python main.py --mode rm --set cifar10 --arch vgg13_dense --rep_type bd --repair_sample_num 1000
    num_classes = {'gtsrb': 43, 'imagenet': 50, 'cifar10': 10, 'svhn': 10, 'mnist': 10}
    if args.arch == 'vgg13_dense':
        prob_w, PROBE_NUM = [], 7
        model = VGG13_dense('VGG13', num_class=num_classes[args.set])
        pretrained(w_file, model, device, prob_w)
    elif args.arch == 'res18_dense':
        model = cResNet18_Dense(num_classes=num_classes[args.set])
        prob_w, PROBE_NUM = [], 6
        pretrained(w_file, model, device, prob_w)
    elif args.arch == 'vgg16_dense':
        model = VGG16_dense()
        pretrained(w_file, model, device, [])
    else:
        raise ValueError(f'arch <{args.arch}> not supported')
    all_keys = model.state_dict().keys()
    model_keys = [k for k in all_keys if 'probe' not in k]
    if not os.path.exists(args.rep_dir):
        os.makedirs(args.rep_dir)
    log_file = open(os.path.join(args.rep_dir, 'RM_log.txt'), 'a')
    log_file.write(f'{time.strftime("%Y-%m-%d, %H:%M:%S")}, {args.set}, {args.rep_type}, {args.arch}\n')

    rep_clean = get_standard(set=args.set, num=1000, train=False, seed=233, process=['std'])
    if args.rep_type == 'adv':
        rep_bd = get_adv_mix(args.set, args.arch, 1000, ['std'], './datasets/', validate=False)
    else:
        rep_bd = get_backdoor(set='%s_%s'%(args.rep_type, args.set), num=args.repair_sample_num, train=False,
                              mode='ptest', seed=233, RTL=True, process=['std'], model_seed=seed, arch=args.arch)

    backdoor_rep = Train(
        model=model,
        clean_test=rep_clean,
        repair_test=rep_bd,
        gpu=gpu,
        iters=10,
        save_name=None,
        gamma=0.01
    )
    if args.exchange:
        if 'vgg' in args.arch:
            blocks = ['features1.3', 'features2.3', 'features3.3', 'features4.3', 'features5.3', 'dense1', 'dense2',
                      'classifier']
            PROBE_NUM = 7
        elif args.arch == 'res18_dense':
            blocks = ['layer1.1.conv2', 'layer2.1.conv2', 'layer3.1.conv2', 'layer4.1.conv2', 'fc1', 'fc2']
            prob_w, PROBE_NUM = [], 6
        mask_time_all = 0
        mask_all = {}
        for b in range(PROBE_NUM):
            mask_bd, mask_time = get_weight_mask(args, args.arch, seed, blocks[b], b, ratio=args.ratio,
                                                 total_num=args.repair_sample_num, topk=args.neuron_num, bd=True,
                                                 update=False)
            mask_all.update(mask_bd)
        print('get mask of ours ^^^^')

        rep_model = backdoor_rep.backdoor_train(assign_mask=mask_all)
    else:
        rep_model = backdoor_rep.backdoor_train()
    time_info = '\n mask time %s repair time %s\n' % (backdoor_rep.mask_time, backdoor_rep.repair_time)
    log_file.write(time_info)
    log_file.flush()
    save_dir = os.path.join(args.rep_dir, 'RM')
    save_model_part(rep_model, model_keys, '%s_%s_%s_%s_%s'%(args.arch, seed, args.set, args.rep_type, args.repair_sample_num), out_dir=save_dir)


def std_trainer(model, loader, criterion, optimizer, e):
    acc = []
    for images, targets in loader:
        images, targets = images.float().to(device), targets.long().to(device)
        output = model(images)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc.append(Accuracy(output, targets)[0].cpu().detach().numpy())

    print('end of epoch : {}, top1 train acc : {}'.format(e, np.mean(acc)))
    return np.mean(acc)

# python main.py --save_dir ./checkpoints/final/ --mode train --set cifar10  --arch res18_dense --repair_sample_num 1000 --rep_type bd


if __name__ == '__main__':
    if args.mode == 'test':
        sys_test(args, rep='ours', head=False, seed_range=2032)
    elif args.mode == 'train':
        for i in range(10):
            set_random_seed(2022 + i)
            seed = 2022 + i
            train_models(args.arch, args, seed=2022 + i, epochs=100)
            break
    elif args.mode == 'neuron':
        for i in range(2022, 2032):
            f = open('mask_time_v2.txt', 'a')
            if 'vgg' in args.arch:
                blocks = ['features1.3', 'features2.3', 'features3.3', 'features4.3', 'features5.3', 'dense1', 'dense2',
                          'classifier']
                PROBE_NUM = 7
            elif args.arch == 'res18_dense':
                blocks = ['layer1.1.conv2', 'layer2.1.conv2', 'layer3.1.conv2', 'layer4.1.conv2', 'fc1', 'fc2']
                prob_w, PROBE_NUM = [], 6
            mask_time_all = 0

            for b in range(PROBE_NUM):
                mask_bd, mask_time = get_weight_mask(args, args.arch, i, blocks[b], b, ratio=args.ratio,
                                                 total_num=args.repair_sample_num, topk=args.neuron_num, bd=True,
                                                 update=True)
                mask_time_all += mask_time
                f.write(f'{args.arch},{args.set},{i},{b},{mask_time}\n')
            f.write(f'{args.arch},{args.set},{i},overall,{mask_time_all}\n')
            f.flush()

    elif args.mode == 'probe':
        for i in range(10):
            set_random_seed(2022 + i)
            seed = 2022 + i
            file_path = f'{args.arch}/seed{seed}/{args.rep_type}_probe_num_{args.probe_train_num}_layer_1/model-best.pt'
            file_path = os.path.join(args.save_dir, file_path)
            if os.path.exists(file_path):
                print(f'{file_path} exists!!!\n continue')
                continue
            train_only_probe(args, seed)

    elif args.mode == 'rep':
        for i in range(10):
            set_random_seed(2022 + i)
            seed = 2022 + i
            if args.rep_type == 'adv' or args.rep_type == 'wp':
                w = '%s/seed%s/std/model-best.pt' % (args.arch, seed)
            else:
                w = '%s/seed%s/%s/model-best.pt' % (args.arch, seed, args.rep_type)
            w = os.path.join(args.save_dir, w)

            ff = f'./checkpoints/repaired/{args.set}/{args.arch}/{args.rep_type}/seed{seed}/' \
                 f'type_{args.rep_type}-rep_layer_{args.rep_layer_num}-rep_num_{args.repair_sample_num}' \
                 f'-ratio_0.0-neuron_{args.neuron_num}-probe_{args.probe_train_num}/model-best.pt'

            if os.path.exists(ff) and args.neuron_num!=1:
                print(ff+ '\nexists********')
                continue
            if args.rep_type == 'wp':
                ptype = 'adv'
            else:
                ptype = args.rep_type
            repair_mid_layers(args.arch, args, w, seed, probe_type=ptype)

    elif args.mode == 'rm':
        seed_range = 2032
        for seed in range(2022, seed_range):
            if args.rep_type == 'adv' or args.rep_type == 'wp':
                w = '%s/seed%s/std/model-best.pt' % (args.arch, seed)
            else:
                w = '%s/seed%s/%s/model-best.pt' % (args.arch, seed, args.rep_type)
            w = os.path.join(args.save_dir, w)
            if args.set == 'imagenet':
                args.repair_sample_num = 100
            else:
                args.repair_sample_num = 1000
            rm_repair(args, w, seed)

    else:
        pass


