import os
import argparse
import numpy as np
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import wandb
import copy
import random
from reparam_module import ReparamModule

np.random.seed(0)
torch.manual_seed(0)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
from distill import parse_args

# python max_logit.py --model=ConvNetD4 --dataset=Tiny --ipc=50 --syn_steps=20 --expert_epochs=2 --max_start_epoch=40 --batch_syn=300 --lr_img=1e4 --lr_lr=1e-04 --lr_teacher=1e-02 --buffer_path=t0 --data_path=data/tiny-imagenet-200 --num_eval 1 --epoch_eval_train 1
# python max_logit.py --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path=t0 --data_path=data/ --num_eval 1 --epoch_eval_train 1

args = parse_args()

args.dsa = True if args.dsa == 'True' else False
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()[1:]
channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

dd_val_loader = None
indices = np.arange(len(dst_train))
np.random.shuffle(indices)

T = len(dst_train)
dst_train0 = list(dst_train)
dst_train, dst_valid = dst_train0[: int(T * 0.9)], dst_train0[int(T * 0.9):]

valid_loader = torch.utils.data.DataLoader(dst_valid, batch_size=args.batch_train, shuffle=True, num_workers=0)
x0 = torch.load('data/images_best_%s_%s%s.pt' % (args.dataset.lower(), args.ipc, '_z' if args.zca else ''))
y0 = torch.load('data/labels_best_%s_%s%s.pt' % (args.dataset.lower(), args.ipc, '_z' if args.zca else '')).long()
C = x0.shape[0] // args.ipc
print('x0:', x0.shape, 'y0:', y0.shape, 'C:', C)
train_indices, val_indices = [], []
I = np.arange(x0.shape[0])
for c in range(C):
    class_indices = I[y0 == c]
    assert args.ipc == len(class_indices), '? %s' % class_indices.shape
    np.random.shuffle(class_indices)
    T = int(class_indices.shape[0] * 0.9)
    train_indices.extend(class_indices[: T])
    val_indices.extend(class_indices[T: ])
x_train, x_val = x0[train_indices], x0[val_indices]
y_train, y_val = y0[train_indices], y0[val_indices]

from utils import TensorDataset

dd_original_data = TensorDataset(x0, y0)
dd_original_loader = torch.utils.data.DataLoader(dd_original_data, batch_size=args.batch_train, shuffle=True, num_workers=0)

dd_dst_train = TensorDataset(x_train, y_train)
dd_train_loader = torch.utils.data.DataLoader(dd_dst_train, batch_size=args.batch_train, shuffle=False, num_workers=0)
dd_dst_val = TensorDataset(x_val, y_val)
dd_val_loader = torch.utils.data.DataLoader(dd_dst_val, batch_size=args.batch_train, shuffle=False, num_workers=0)

fd_net = torch.load('fd_%s/before_fd_%s.pt' % (args.dataset.lower(), args.dataset.lower())).to(args.device)
dd_net = torch.load('dd_%s/%s_dd_%s.pt' % (args.dataset.lower(), ['before', 'original'][0], args.dataset.lower())).to(args.device)

fd_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

filename = 'logits_ipc%s_%s' % (args.ipc, args.dataset.lower())
device = args.device

f = plt.figure(filename, figsize=(8,8))
plt.clf()

colors = ['#5555aa', '#f55d5d', '#333388', '#d33c3c']
edges = ['b', 'r', 'b', 'r']
models = [fd_net, dd_net]
loaders = [fd_loader, dd_original_loader]
labels = 'full dataset', 'distilled dataset'
for i in range(len(models)):
    normal_logits = []
    masked_logits = []
    with torch.no_grad():
        idx = 0
        for data, target in loaders[i]:
            data, target = data.to(device), target.to(device)
            output = models[i](data).cpu().numpy()
            normal_logits.append(output)
            if 1 or i == 1:
                threshold = 0.3
                mask = (torch.rand_like(data) > threshold).float()
                data *= mask
                output = models[i](data).cpu().numpy()
                masked_logits.append(output)
    for j, logits in enumerate([normal_logits, masked_logits]):
        logits = np.concatenate(logits, axis=0)
        logits = np.max(logits, axis=-1)
        plt.hist(
            logits,
            density=True,
            bins=max(logits.shape[0] // 100, 50),
            histtype=['bar', 'step'][j],
            linewidth=[0.8, 1.8][j],
            alpha=[0.3, 0.6][j],
            color=colors[i+2*j],
            edgecolor=edges[i+2*j],
            range=(5, 30),
            label=['', 'masked '][j] + labels[i]
        )
        # if i != 1:
        #     break

plt.legend()
plt.xlabel('logit value')
plt.ylabel('percentage')
d = args.dataset.lower()
if d == 'tiny':
    d += ' imagenet'
plt.title('histogram of maximum logits on ' + d)
plt.savefig(filename)
plt.close(filename)


