import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy.ma as ma

def plot_avg_conf_acc():
    pass

red = {
    'width': 0.1,
    'color': '#f55d5d',
    'edgecolor': 'r',
    'align': 'edge',
}
blue = {
    'width': 0.1,
    'color': '#5d76f5',
    'edgecolor': 'b',
    'align': 'edge',
}
purple = {
    'width': 0.1,
    'color': '#775df5',
    'edgecolor': 'purple',
    'align': 'edge',
}
green = {
    'width': 0.02,
    'color': '#a8f55d',
    'edgecolor': 'g',
    'align': 'edge',
}
def plot_gap(acc, probs, sizes, bins, ece, actual_acc, is_train=False, filename=''):
    f = plt.figure(filename, figsize=(8,8))
    plt.clf()
    m1 = ma.where(probs > acc)
    m2 = ma.where(acc > probs)
    m3 = ma.where(acc == probs)

    if m1[0].shape[0] > 0:
        plt.bar(
            bins[m1],
            probs[m1],
            label='bin confidence',
            **red,
        )
    plt.bar(
        bins,
        acc,
        label='bin accuracy',
        tick_label=['%.1f'%b for b in bins],
        **blue
    )
    if m2[0].shape[0] > 0:
        plt.bar(
            bins[m2],
            probs[m2],
            label='bin confidence' if m1[0].shape[0] == 0 else None,
            **red,
        )
    if m3[0].shape[0] > 0:
        plt.bar(
            bins[m3],
            probs[m3],
            label='conf == acc',
            **purple,
        )
    plt.bar(
        bins + 0.05,
        sizes / sizes.sum(),
        label=f'% of samples',
        **green
    )
    plt.legend()
    plt.xlabel('confidence')
    plt.ylabel('percentage')
    plt.title('ece %.4f' % ece + ' acc %.4f' % actual_acc)
    plt.savefig(filename)
    plt.close(filename)


def calc_ece(
    model,
    loader,
    global_step: int,
    device: torch.device,
    is_test_set: bool = False,
    is_train_set: bool = False,
    plot=False,
    filename=''
):
    model.eval()

    loss = 0
    correct = 0
    n = 0
    # pbar = tqdm(total=len(loader), dynamic_ncols=True)
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

            # pbar.update(1)

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

    acc_list = np.zeros(len(idx_list))
    conf_list = np.zeros(len(idx_list))
    size_list = np.zeros(len(idx_list))
    actual_acc = 0
    for i in range(len(idx_list)):
        if len(idx_list[i]) == 0:
            acc = 0
            conf = 0
        else:
            acc = sum(y_true[idx_list[i]] == np.reshape(preds_list[idx_list[i]], (-1))) / len(idx_list[i])
            actual_acc += acc * len(idx_list[i])
            conf = np.mean(probs_max_list[idx_list[i]])

        acc_list[i] = acc
        conf_list[i] = conf
        size_list[i] = len(idx_list[i])

    n = sum(size_list)
    ece = 0
    for i in range(len(acc_list)):
        ece += abs(acc_list[i] - conf_list[i]) * size_list[i] / n
    print('ece:', ece, 'len probs_max_list:', len(probs_max_list))

    bins = np.linspace(0, 0.9, 10)
    if plot:
        plot_gap(acc_list, conf_list, size_list, bins, 
                 ece=ece, 
                 actual_acc=actual_acc / n, 
                 is_train=is_train_set,
                 filename=filename)

    return ece


def plot_max_logit(
    model,
    trainloader,
    filename='',
    device='cuda'
):
    logits = []
    with torch.no_grad():
        idx = 0
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            output = model(data).cpu().numpy()
            logits.append(output)
    logits = np.concatenate(logits, axis=0)
    print('all logits:', logits.shape)
    logits = np.max(logits, axis=-1)

    f = plt.figure(filename, figsize=(8,8))
    plt.clf()
    plt.hist(
        logits,
        density=True,
        bins=max(logits.shape[0] // 100, 50),
        color='#5555aa',
        edgecolor='b',
        range=(0, 30)
    )
    plt.legend()
    plt.xlabel('logit value')
    plt.ylabel('percentage')
    plt.title('histogram of maximum logits ' + filename)
    plt.savefig(filename)
    plt.close(filename)
