import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_avg_conf_acc():
    pass


def plot_gap(acc, probs, bins, epoch):
    filename = 'ipc50_gap_%d'%epoch
    f = plt.figure(filename, figsize=(8,8))
    plt.clf()
    plt.bar(
        bins,
        probs,
        width=0.1,
        color='red',
        edgecolor='r',
        alpha=0.5,
        align='edge',
        tick_label=['%.1f'%b for b in bins]
    )
    plt.bar(
        bins,
        acc,
        width=0.1,
        color=np.array([60, 60, 255]) / 255,
        edgecolor='b',
        alpha=0.8,
        align='edge',
        tick_label=['%.1f'%b for b in bins]
    )
    plt.xlabel('confidence')
    plt.ylabel('accuracy')
    plt.savefig(filename)
    plt.close(filename)


def calc_ece(
    model,
    loader,
    global_step: int,
    epoch: int,
    device: torch.device,
    is_test_set: bool = False,
    is_train_set: bool = False,
):
    model.eval()

    loss = 0
    correct = 0
    n = 0
    pbar = tqdm(total=len(loader), dynamic_ncols=True)
    with torch.no_grad():
        idx = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            probs = torch.softmax(output, dim=-1).cpu().numpy()
            probs_max = np.reshape(np.max(probs, axis=1), (-1,1))
            preds = np.reshape([np.argmax(probs[m, ]) for m in range(probs.shape[0])], (-1,1))

            if idx == 0:
                probs_list = probs
                probs_max_list = probs_max
                preds_list = preds
                y_true = target.cpu().numpy()
            else:
                probs_list = np.concatenate((probs_list, probs), axis=0)
                probs_max_list = np.concatenate((probs_max_list, probs_max), axis=0)
                preds_list = np.concatenate((preds_list, preds), axis=0)
                y_true = np.concatenate((y_true, target.cpu().numpy()), axis=0)

            pbar.update(1)

            idx += 1

        loss /= len(loader)

    val_or_test = "val" if not is_test_set else "test"
    val_or_test = val_or_test if not is_train_set else 'train'

    probs_list_sum = np.sum(probs_list, axis=1)
    p_sum_mu = np.mean(probs_list_sum == 1)

    interval = 0.1
    bins = np.arange(0, 1+0.001, interval)
    # n_bins = 10
    # bins = np.linspace(0, 1, n_bins)
    idx_list =[[]]
    for idx in range(len(bins)-2):
        idx_list.append([])
    for idx in range(probs_max_list.shape[0]):
        g = int(probs_max_list[idx] * 100 // 10)
        if g > 9:
            g = 9
        idx_list[g].append(idx)

    acc_list = []; conf_list = []; size_list = []
    for i in range(len(idx_list)):
        if len(idx_list[i]) == 0:
            acc = 0
            conf = 0
        else:
            acc = sum(y_true[idx_list[i]] == np.reshape(preds_list[idx_list[i]], (-1))) / len(idx_list[i])
            conf = np.mean(probs_max_list[idx_list[i]])

        acc_list.append(acc)
        conf_list.append(conf)
        size_list.append(len(idx_list[i]))            

    print(acc_list)
    print(conf_list)
    print(size_list)
    print(len(acc_list), len(conf_list), bins.shape, )
    n = sum(size_list)
    ece = 0
    for i in range(len(acc_list)):
        ece += abs(acc_list[i] - conf_list[i]) * size_list[i] / n
    print('ece:', ece, 'len probs_max_list:', len(probs_max_list))

    bins = np.linspace(0, 0.9, 10)
    # plt.xlabel('confidence p(x)')
    plot_gap(acc_list, conf_list, bins, epoch=epoch)

    return ece
