import os
import time
import json

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from DataProcess import SourceDataset, linear_collate
from models.optimizer import RAdam
from models.DPCNN import DPCNN

global_label_cnt = 4
global_vocab_size = 5414 + 2
global_embed_size = 128
global_hidden_size = 128
global_num_layers = 5
global_batch_size = 128
global_num_workers = 10
global_learning_rate = 1e-3
global_epoch_count = 300


# Important Parameters
def train(dataset, model, device, rank=0):
    torch.cuda.set_device(device)
    dataset_train, dataset_eval_in, dataset_eval_out = dataset
    dataset_train = SourceDataset(dataset_train)
    dataset_eval_in = SourceDataset(dataset_eval_in)
    dataset_eval_out = SourceDataset(dataset_eval_out)

    model = DPCNN(global_label_cnt, global_vocab_size, global_embed_size, global_hidden_size, global_num_layers)
    criterion = nn.CrossEntropyLoss()
    model, criterion = model.to(device), criterion.to(device)

    dataloader_train = DataLoader(dataset_train, shuffle=True, pin_memory=True, num_workers=global_num_workers,
                                  batch_size=global_batch_size, drop_last=True, collate_fn=linear_collate)
    dataloader_eval_in = DataLoader(dataset_eval_in, shuffle=False, num_workers=global_num_workers, batch_size=1, collate_fn=linear_collate)
    dataloader_eval_out = DataLoader(dataset_eval_out, shuffle=False, num_workers=global_num_workers, batch_size=1, collate_fn=linear_collate)
    optimizer = RAdam(model.parameters(), lr=global_learning_rate)
    t1 = time.time()
    best_in_eval = 0.0
    best_out_eval = 0.0
    for epoch in range(global_epoch_count):
        total_loss = []
        total_loss_eval_in = []
        total_loss_eval_out = []
        coarse_group_correct = [0, 0]
        deep_group_correct = [0, 0]
        total_group = [0, 0]
        bar = tqdm(desc='(0/3)Train #{:02d}'.format(epoch), total=len(dataloader_train), leave=False)

        model.train()
        for data, label in dataloader_train:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            bar.update()
        bar.close()
        bar2 = tqdm(desc='(1/3)Eval In #{:02d}'.format(epoch), total=len(dataset_eval_in), leave=False)
        bar3 = tqdm(desc='(2/3)Eval In #{:02d}'.format(epoch), total=len(dataset_eval_out), leave=False)
        model.eval()
        with torch.no_grad():
            for data, label in dataset_eval_in:
                data, label = data.to(device), label.to(device)
                output = model(data)
                loss = criterion(output, label)
                total_loss_eval_in.append(loss.item())
                if output >= 2 and label >= 2:
                    coarse_group_correct[0] = coarse_group_correct[0] + 1
                elif output < 2 and label < 2:
                    coarse_group_correct[0] = coarse_group_correct[0] + 1
                total_group[0] = total_group + 1
                if output == label:
                    deep_group_correct[0] = deep_group_correct[0] + 1
                bar2.update()
            bar2.close()

            for data, label in dataset_eval_out:
                data, label = data.to(device), label.to(device)
                output = model(data)
                loss = criterion(output, label)
                total_loss_eval_out.append(loss.item())
                if output >= 2 and label >= 2:
                    coarse_group_correct[1] = coarse_group_correct[1] + 1
                elif output < 2 and label < 2:
                    coarse_group_correct[1] = coarse_group_correct[1] + 1
                total_group[1] = total_group + 1
                if output == label:
                    deep_group_correct[1] = deep_group_correct[1] + 1
                bar3.update()
            bar3.close()
        epoch_train_loss = np.mean(total_loss)
        epoch_eval_in_loss = np.mean(total_loss_eval_in)
        epoch_eval_out_loss = np.mean(total_loss_eval_out)
        print('(3/3)Report: Epoch #{:02d}: Training loss: {:.06F}, Eval Loss: (I {:.06F})(O {:.06F}), Eval Accuracy: (I {:.03F}%)(O {:.03F}%), {:.03F}%, lr: {:.06F}')
