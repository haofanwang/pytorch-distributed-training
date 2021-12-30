'''
This code is mostly built on the top of https://github.com/Lisennlp/distributed_train_pytorch
'''

import os
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler



class ToyModel(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ToyModel, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.linear(x)
        out = F.softmax(out, dim=-1)
        return out


def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size


def build_fake_data(size=1000):
    x1 = [(random.uniform(0, 0.5), 0) for i in range(size // 2)]
    x2 = [(random.uniform(0.5, 1), 1) for i in range(size // 2)]
    return x1 + x2


def evaluate(valid_loader):
    model.eval()
    with torch.no_grad():
        cnt = 0
        total = 0
        for inputs, labels in valid_loader:
            inputs, labels = inputs.unsqueeze(1).float().cuda(), labels.long().cuda()
            output = model(inputs)
            predict = torch.argmax(output, dim=1)
            cnt += (predict == labels).sum().item()
            total += len(labels)
            # print(f'right = {(predict == labels).sum()}')
        cnt = torch.Tensor([cnt]).to(inputs.device)
        total = torch.Tensor([total]).to(inputs.device)
        reduced_param = torch.cat((cnt.view(1), total.view(1)))
        cnt = reduced_param[0].item()
        total = reduced_param[1].item()
    return cnt, total


def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help="local gpu id")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learn rate")
    parser.add_argument('--epochs', type=int, default=5, help="train epoch")
    parser.add_argument('--seed', type=int, default=40, help="train epoch")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = get_args()
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    set_random_seed(args.seed)
    
    # initilization
    # world_size = node_number * gpu_number_per_node
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    global_rank = dist.get_rank()
    print(f'global_rank = {global_rank} local_rank = {args.local_rank} world_size = {args.world_size}')
    
    # build a model
    model = ToyModel(1, 2).cuda()
    
    # DDP setting
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # construct fake datasets
    trainset = build_fake_data(size=10000)
    validset = build_fake_data(size=10000)
    
    # DDP samplers
    train_sampler = DistributedSampler(trainset)
    valid_sampler = DistributedSampler(validset)
    
    # build dataloaders
    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              pin_memory=True,
                              sampler=train_sampler)

    valid_loader = DataLoader(validset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              pin_memory=True,
                              sampler=valid_sampler)
    
    # optmizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # training mode
    model.train()
    
    # main process
    for e in range(int(args.epochs)):
        for idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.unsqueeze(1).float().cuda()
            labels = labels.long().cuda()
            output = model(inputs)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            reduce_loss(loss, global_rank, args.world_size)
        cnt, total = evaluate(valid_loader)
        if global_rank == 0:
            print(f'epoch {e} || eval accuracy: {cnt / total}')
