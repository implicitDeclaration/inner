import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os
import random
from csv import reader
from PIL import Image


def neuron_overlap(save_dir, arch, seed=2022):
    base_dir = save_dir
    if arch == 'vgg13_dense':
        blocks = ['features1.3', 'features2.3', 'features3.3', 'features4.3', 'features5.3', 'dense1', 'dense2',
                  'classifier']
    else:
        blocks = ['layer1.1.conv2', 'layer2.1.conv2', 'layer3.1.conv2', 'layer4.1.conv2', 'fc1', 'fc2']

    ratios = [0.3, 0.5, 0.7, 1.0]
    ol_per_layer = []

    for r in ratios:
        ratio_i = []
        for b in blocks:
            compare_base = os.path.join(base_dir, '%s/seed%s/%s/fault_%s_top%s_%s_%s.npy' % (
                arch, seed, 'bd', b, 100, 0.0, 1000))

            neuron_file = os.path.join(base_dir, '%s/seed%s/%s/fault_%s_top%s_%s_%s.npy' % (
                arch, seed, 'bd', b, 100, r, 1000))
            comp_base = np.load(compare_base)
            comp_with = np.load(neuron_file)
            total = len(comp_base)
            eq = [i for i in comp_with if i in comp_base]
            overlapp_rate = len(eq)/total
            ratio_i.append(overlapp_rate)
        ol_per_layer.append(ratio_i)
    ol_per_layer = np.array(ol_per_layer)
    print(f'overlap per layer:\n{ol_per_layer}')
    print(f'overlap average:\n{np.mean(ol_per_layer, axis=1)}')



def sample_enhance(images, labels, trg_label, device):
    '''

    :param images:
    :param labels:
    :param trg_label:
    :return: enhanced images with the trg_label
    '''
    from torchvision import transforms
    # trg_idx = torch.nonzero(labels == trg_label)
    # print(f'selected image indexes are {trg_idx}')
    # trg_images = images[trg_idx].squeeze()
    trg_images = images.squeeze()


    #print(f'selected images shape is {trg_images.shape}')
    trans = [

        transforms.RandomHorizontalFlip(p=1),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.RandomCrop((32, 32))
    ]
    # transform = transforms.Compose()

    enhanced_images = trg_images.unsqueeze(dim=0)
    for t in trans:
        new_img = t(trg_images).unsqueeze(dim=0)
        enhanced_images = torch.concat((enhanced_images, new_img), dim=0)
        # print(f'after transform, image shape is {enhanced_images.shape}')

    idx = torch.randperm(enhanced_images.size(0))
    shuffled_images = enhanced_images[idx]
    new_labels = torch.ones(shuffled_images.size(0))
    new_labels = new_labels * trg_label
    return shuffled_images.to(device), new_labels.long().to(device)


def val_robust_rep(model, feat_mask, id, data_loader, device, per_class=False, nc=10):
    model.eval()
    total, correct = 0, 0

    class_correct = list(0. for i in range(nc))
    class_total = list(0. for i in range(nc))

    for data, target in data_loader:
        bs = data.size(0)
        data, target = data.float().to(device), target.to(device)
        outputs = model(data, mid_input=feat_mask, mid_idx=id)
        outputs = outputs.view(bs, nc)
        _, predicted = torch.max(outputs.data, 1)

        if per_class:
            c = (predicted == target).squeeze()
            for i in range(bs):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        total += target.size(0)
        correct += (predicted == target).sum().item()
    acc_per_class = []

    if per_class:
        for i in range(nc):
            acc_per_class.append(class_correct[i] / class_total[i])
        return acc_per_class

    accuracy = correct / total
    # print("Test Accuracy: {}/ {} = {} ".format(correct, total, accuracy))
    return accuracy


def validate_target_asr(model, target_class, data_loader, device, nc=10):
    model.eval()
    sucessed, total = 0, 0
    for data, target in data_loader:
        bs = data.size(0)
        data, target = data.float().to(device), target.to(device)
        outputs = model(data)
        outputs = outputs.view(bs, nc)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(bs):
            if predicted[i] == target_class:
                sucessed += 1
            total += 1

    return sucessed/total


def validate(model, data_loader, device, per_class=False, nc=10, set_target_label=None):
    model.eval()
    total,correct = 0, 0

    class_correct = list(0. for i in range(nc))
    class_total = list(0. for i in range(nc))
    with torch.no_grad():
        for data, target in data_loader:
            bs = data.size(0)
            data, target = data.float().to(device), target.to(device)
            if set_target_label is not None:
                attack_target = torch.ones_like(target)
                attack_target = set_target_label * attack_target
                target = attack_target.to(device)
            outputs = model(data)
            outputs = outputs.view(bs, nc)
            _, predicted = torch.max(outputs.data, 1)

            if per_class:
                c = (predicted == target).squeeze()
                for i in range(bs):
                    label = target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

            total += target.size(0)
            correct += (predicted == target).sum().item()
        acc_per_class = []

        if per_class:
            for i in range(nc):
                if class_total[i] == 0:
                    # acc_per_class.append(1)
                    continue
                acc_per_class.append(class_correct[i] / class_total[i])
            return acc_per_class

        accuracy = correct / total
        # print("Test Accuracy: {}/ {} = {} ".format(correct, total, accuracy))
        return accuracy


def probe_val(model, data_loader, probe_num, device):
    model.eval()
    total, correct = {}, {}
    for i in range(probe_num):
        total['probe%s' % (i + 1)] = 0
        correct['probe%s' % (i + 1)] = 0

    for data, target in data_loader:
        data, target = data.float().to(device), target.to(device)
        outputs = model(data, probe=True)
        for i in range(probe_num):
            _, predicted = torch.max(outputs[i].data, 1)

            total['probe%s' % (i + 1)] += target.size(0)
            correct['probe%s' % (i + 1)] += (predicted == target).sum().item()

    accuracy = {}
    for i in range(probe_num):
        accuracy['probe%s' % (i + 1)] = correct['probe%s' % (i + 1)] / total['probe%s' % (i + 1)]
    # print("Test Accuracy: {}/ {} = {} ".format(correct, total, accuracy))
    return accuracy


def get_target_param(model, layer_list, arch):
    selected_param = []

    if 'cnn' in arch:
        for p in model.named_parameters():
            if 'layers' in p[0]:
                layer_id = p[0].split('.')[1]

                if int(layer_id) in layer_list:
                    selected_param.append(p[1])
    if 'res' in arch:
        for p in model.named_parameters():
            if layer_list in p[0]:

                selected_param.append(p[1])
            elif layer_list == 'fc':
                if 'fc' in p[0] and 'probe' not in p[0]:
                    selected_param.append(p[1])
    if 'vgg' in arch:
        for p in model.named_parameters():
            if layer_list in p[0]:
                selected_param.append(p[1])
            if 'fc' in p[0] and 'probe' not in p[0]:
                selected_param.append(p[1])
    return selected_param


def binary_out_acc(outputs, target):
    total0, total1, correct0, correct1 = 0, 0, 0, 0

    idx0, _ = torch.where(outputs < 0.5)
    idx1, _ = torch.where(outputs > 0.5)
    tidx0 = torch.where(target == 0)[0]
    tidx1 = torch.where(target == 1)[0]

    total0 += tidx0.size(0)
    total1 += tidx1.size(0)
    correct0_element = [x for x in idx0 if x in tidx0]
    correct1_element = [x for x in idx1 if x in tidx1]
    correct0 += len(correct0_element)
    correct1 += len(correct1_element)

    overall_acc0 = (correct0 + correct1) / (total0 + total1)
    #print("Overall Accuracy {} ---> label0 acc {}, label1 acc {} ".format(overall_acc0, accuracy0, accuracy1))
    return overall_acc0


def Accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def CrossEntropy(outputs, targets, temp):
    log_softmax_outputs = F.log_softmax(outputs/temp, dim=1)
    softmax_targets = F.softmax(targets/temp, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


def save_model_by_name(model, global_step, name, out_dir='./checkpoints', acc_rec=None):
    save_dir = os.path.join(out_dir, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if type(global_step) == int:
        file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    else:
        file_path = os.path.join(save_dir, 'model-{}.pt'.format(global_step))

    if acc_rec is not None:  # save the best in the history
        if acc_rec[-1] >= max(acc_rec):
            state = model.state_dict()
            torch.save(state, file_path)
            print('New best saved {} to {}'.format(acc_rec[-1], file_path))
        else:
            return
    else:
        state = model.state_dict()
        torch.save(state, file_path)
        print('Saved to {}'.format(file_path))


def save_model_part(model, state_keys, name, out_dir='./checkpoints', acc_rec=None):
    # save part of the model's weights, like the probe
    save_dir = os.path.join(out_dir, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    state_dict = model.state_dict()
    save_dict = {
        k: v for k, v in state_dict.items()
        if k in state_keys
    }
    if acc_rec is None:
        file_path = os.path.join(save_dir, 'model-best.pt')
        torch.save(save_dict, file_path)
        return
    else:
        if acc_rec[-1] >= max(acc_rec):
            file_path = os.path.join(save_dir, 'model-best.pt')
            torch.save(save_dict, file_path)
            print('old best is {} New best saved {} to {}'.format(max(acc_rec), acc_rec[-1], file_path))
        else:
            return


def probe_acc(model,):
    pass

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def pd_tf_convert(src_data, labels=None, target='tf'):
    if target == 'tf':
        if isinstance(src_data, pd.Series) or isinstance(src_data, pd.DataFrame):
            return_data = torch.tensor(src_data.values)
            if labels is not None:
                pass
            else:
                return return_data
        else:
            raise ValueError(f'input must be pd type, got {type(src_data)}')
    else:
        if torch.is_tensor(src_data):
            return_data = src_data.detach().cpu().numpy()
            return_data = pd.DataFrame(return_data)
            if labels is not None:
                label_col = labels.detach().cpu().numpy()
                label_col = pd.DataFrame(label_col)
                return_data['n'] = label_col
            return return_data
        else:
            raise ValueError(f'input must be tf tensor, got {type(src_data)}')


def debiased_select(ori_pred, rep_pred, data, label, ori_inv_pred=None, rep_inv_pred=None):
    '''select biased samples that classified correctly after repaired'''
    debiased_sample = []

    _, ori_predicted = torch.max(ori_pred.data, 1)
    _, rep_predicted = torch.max(rep_pred.data, 1)
    ori_predicted.reshape(1, -1)
    rep_predicted.reshape(1, -1)
    # print(f'original prediction is >{ori_predicted}<\nrepaired prediction is >{rep_predicted}<'
    #       f'\n true label is >{label.reshape(1, -1)}<')
    if ori_inv_pred is not None:  # idependent based fairness
        _, ori_inv = torch.max(ori_inv_pred.data, 1)
        _, rep_inv = torch.max(rep_inv_pred.data, 1)
        ori_inv.reshape(1, -1)
        rep_inv.reshape(1, -1)
        ori_result_change = ori_inv.eq(ori_predicted).squeeze()
        rep_result_change = rep_inv.eq(rep_predicted).squeeze()
        for i in range(len(ori_result_change)):
            if ori_result_change[i] == False and rep_result_change[i] == True:
                whole_data = torch.cat((data[i], label[i]))
                debiased_sample.append(whole_data.cpu().detach().numpy())
        return debiased_sample

    ori_correct = ori_predicted.eq(label.reshape(1, -1)).squeeze()
    rep_correct = rep_predicted.eq(label.reshape(1, -1)).squeeze()
    # print(f'correct predicted of ori and rep is \n{ori_correct}\n{rep_correct}')


    for i in range(len(ori_predicted)):
        if rep_correct[i] == True and ori_correct[i] == False:
            whole_data = torch.cat((data[i], label[i]))
            debiased_sample.append(whole_data.cpu().detach().numpy())

    return debiased_sample


def gender_inverse(tf_data, device):
    np_data = tf_data.cpu().detach().numpy()
    df = pd.DataFrame(np_data)
    df.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']
    # print(f'before inverse : {df.iloc[:10]}')
    genders = df['i']
    df['i'] = np.where(df['i'] == 0, 1, 0)
    # print(f'after inverse : {df.iloc[:10]}')
    tf_result = torch.tensor(df.values).float()
    tf_result = tf_result.to(device)
    return tf_result


def load_npy_data(path, sens_attr=8, num=-1):
    # return iterable data, generated by ADF
    # 8 is gender,
    incomp_data = np.load(path)
    com_data0 = np.insert(incomp_data, sens_attr, 0, axis=1)
    com_data1 = np.insert(incomp_data, sens_attr, 1, axis=1)
    com_data = np.concatenate((com_data0, com_data1), axis=0)
    np.random.seed(2022)
    np.random.shuffle(com_data)

    if num < 0:
        return com_data, com_data1, com_data0
    elif num < 1:
        return com_data[:np.floor(num)]
    else:
        if num > len(com_data):
            num = len(com_data)
        return com_data[:num]


def pretrained(path, model, device, probe):
    if os.path.isfile(path):
        print("=> loading pretrained weights from '{}'".format(path))
        pretrained = torch.load(path, map_location=device) #
        for p in probe:
            if os.path.isfile(p):
                probes_weight = torch.load(p, map_location=device)
                # print(probes_weight.keys())
                pretrained.update(probes_weight)
                print("=> loading probe weights from '{}'".format(p))

        model_state_dict = model.state_dict()
        for k, v in pretrained.items():
            if k not in model_state_dict or v.size() != model_state_dict[k].size():
                print(f"IGNORE: {k}")
                if k in model_state_dict:
                    print(f"IGNORE: {k}")
                    print(f"for {k}, except shape{model_state_dict[k].shape}, got {v.shape}")
                else:
                    print(f"IGNORE: {k}, it is not in model's state dict")
        pretrained = {
            k: v
            for k, v in pretrained.items()
            if (k in model_state_dict and v.size() == model_state_dict[k].size())
        }
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)
        model.to(device)
    else:
        print("=> no pretrained weights found at '{}'".format(path))
        import sys
        sys.exit()