#!bin/bash

# cifar10 zca ipc 10
# python distill.py --dataset=CIFAR10 --ipc=10 --syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --zca --lr_img=10000 --lr_lr=1e-04 --lr_teacher=0.01 --buffer_path=t0 --data_path=.

# cifar10 zca ipc 50
# python distill.py --dataset=CIFAR10 --ipc=50 --syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --zca --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.001 --buffer_path=t0 --data_path=.

# cifar100 zca ipc 10
# python distill.py --dataset=CIFAR100 --ipc=10 --syn_steps=20 --expert_epochs=2 --max_start_epoch=20 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path=t0 --data_path=.

# cifar100 zca ipc 50
# python distill.py --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path=exp30 --data_path=. --num_eval 1

# cifar100 zca ipc 30
python distill.py --dataset=CIFAR100 --ipc=30 --syn_steps=40 --expert_epochs=2 --max_start_epoch=30 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path=exp30 --data_path=. --num_eval 1

# cifar100 zca ipc 10
python distill.py --dataset=CIFAR100 --ipc=10 --syn_steps=20 --expert_epochs=2 --max_start_epoch=20 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path=exp30 --data_path=. --num_eval 1

# cifar100 zca ipc 5
# python distill.py --dataset=CIFAR100 --ipc=5 --syn_steps=20 --expert_epochs=2 --max_start_epoch=20 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path=exp30 --data_path=. --num_eval=1

# cifar100 zca ipc 1
# python distill.py --dataset=CIFAR100 --ipc=1 --syn_steps=20 --expert_epochs=3 --max_start_epoch=20 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path=exp30 --data_path=. --num_eval=1
