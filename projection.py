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

class A: 
    ipc = 50
    dataset = 'CIFAR100'
    zca = True
    device = 'cpu'
args = A()

import clip
model, preprocess = clip.load("ViT-B/32", device='cpu')

# full data
num_classes = 100
channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = \
    get_dataset('CIFAR100', '.', 256, '.', args=args)
T = num_classes * args.ipc
# T = len(dst_train)

indices = np.arange(len(dst_train))
np.random.shuffle(indices)
dst_train0 = [dst_train[d] for d in indices[ : T]]

y0_r = np.array([d[1].numpy() for d in dst_train0])
x0 = np.array([d[0].numpy() for d in dst_train0])

v = np.random.random([np.prod(x0.shape[1:]), 2])
v, _ = np.linalg.qr(v)

x0_r = np.matmul(x0.reshape([x0.shape[0], -1]), v)
'''
with torch.no_grad():
    x0_c = []
    for z in dst_train0:
        x0_c.append(preprocess(z[0].cpu()))
    x0_c = torch.stack(x0_c, dim=0)
    x0_r = model.encode_image(x0_c)
    print('x0_r:', x0_r.shape)

tsne = TSNE(n_components=2)
x0_r = tsne.fit_transform(x0_r.numpy().reshape([x0_r.shape[0], -1]))
'''


nc = 4
fig, axes = plt.subplots(nc, 2)
for i, color in zip(range(nc), ['r', 'g', 'blue', 'black', 1,2,3,]):
    t = x0_r[y0_r == i]
    axes[i][1].scatter(t[:, 0], t[:, 1], s=0.5, color=color, label='class %d' % i)
    # axes[0][1].legend()
axes[0][1].set_title('fulldata projection', fontsize=8)


# dd
x0 = torch.load('images_best_%s_%s%s.pt' % (args.dataset.lower(), args.ipc, '_z' if args.zca else ''))
y0 = torch.load('labels_best_%s_%s%s.pt' % (args.dataset.lower(), args.ipc, '_z' if args.zca else ''))
y0_r = y0.numpy()


x0_r = np.matmul(x0.numpy().reshape([x0.shape[0], -1]), v)
'''
with torch.no_grad():
    x0_c = []
    for z in x0:
        x0_c.append(preprocess(z.cpu()))
    x0_c = torch.stack(x0_c, dim=0)
    x0_r = model.encode_image(x0_c)
    print('x0_r:', x0_r.shape)
x0_r = tsne.fit_transform(x0_r.numpy().reshape([x0_r.shape[0], -1]))
print('x0_r:', x0_r.shape)
'''

for i, color in zip(range(nc), ['r', 'g', 'blue', 'black', 1,2,3,]):
    t = x0_r[y0_r == i]
    axes[i][0].scatter(t[:, 0], t[:, 1], s=0.5, color=color, label='class %d' % i)
    # axes[0][0].legend()
axes[0][0].set_title('dd projection', fontsize=8)


plt.legend(loc='best')
plt.savefig('pca_random_proj_qr_%s_ipc%s.jpg' % (args.dataset, args.ipc))

