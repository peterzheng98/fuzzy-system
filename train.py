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
import sys

global_label_cnt = 2
global_vocab_size = 5414 + 2
global_embed_size = 128
global_hidden_size = 128
global_num_layers = 5
global_batch_size = 128
global_num_workers = 10
global_learning_rate = 1e-3
global_epoch_count = 300
global_t_max = 64
global_eta_min = 1e-9


# Important Parameters
def train(dataset, model_file, device, rank=0):
    torch.cuda.set_device(device)
    dataset_train, dataset_eval_in, dataset_eval_out = dataset
    dataset_train = SourceDataset(dataset_train, global_vocab_size, 25)
    dataset_eval_in = SourceDataset(dataset_eval_in, global_vocab_size, 25)
    dataset_eval_out = SourceDataset(dataset_eval_out, global_vocab_size, 25)

    model = DPCNN(global_label_cnt, global_vocab_size, global_embed_size, global_hidden_size, global_num_layers)
    criterion = nn.CrossEntropyLoss()
    model, criterion = model.to(device), criterion.to(device)

    dataloader_train = DataLoader(dataset_train, shuffle=True, pin_memory=True, num_workers=global_num_workers,
                                  batch_size=global_batch_size, drop_last=True, collate_fn=linear_collate)
    dataloader_eval_in = DataLoader(dataset_eval_in, shuffle=False, num_workers=global_num_workers, batch_size=global_batch_size, collate_fn=linear_collate)
    dataloader_eval_out = DataLoader(dataset_eval_out, shuffle=False, num_workers=global_num_workers, batch_size=global_batch_size, collate_fn=linear_collate)
    optimizer = RAdam(model.parameters(), lr=global_learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=global_t_max, eta_min=global_eta_min)
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
        bar = tqdm(desc='Train #{:02d}'.format(epoch), total=len(dataloader_train), leave=False)

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
        # bar2 = tqdm(desc='(1/3)Eval In #{:02d}'.format(epoch), total=len(dataset_eval_in), leave=False)
        # bar3 = tqdm(desc='(2/3)Eval In #{:02d}'.format(epoch), total=len(dataset_eval_out), leave=False)
        model.eval()
        with torch.no_grad():
            for data, label in dataloader_eval_in:
                data, label = data.to(device), label.to(device)
                output_raw = model(data)
                loss = criterion(output_raw, label)
                _, output = torch.max(output_raw, 1)
                total_loss_eval_in.append(loss.item())
                assert output.shape == label.shape
                for i in range(len(output)):
                    if output[i] >= 2 and label[i] >= 2:
                        coarse_group_correct[0] = coarse_group_correct[0] + 1
                    elif output[i] < 2 and label[i] < 2:
                        coarse_group_correct[0] = coarse_group_correct[0] + 1
                    total_group[0] = total_group[0] + 1
                    if output[i] == label[i]:
                        deep_group_correct[0] = deep_group_correct[0] + 1
                # bar2.update()
            # bar2.close()

            for data, label in dataloader_eval_out:
                data, label = data.to(device), label.to(device)
                output_raw = model(data)
                loss = criterion(output_raw, label)
                _, output = torch.max(output_raw, 1)
                total_loss_eval_out.append(loss.item())
                assert output.shape == label.shape
                for i in range(len(output)):
                    if output[i] >= 2 and label[i] >= 2:
                        coarse_group_correct[1] = coarse_group_correct[1] + 1
                    elif output[i] < 2 and label[i] < 2:
                        coarse_group_correct[1] = coarse_group_correct[1] + 1
                    total_group[1] = total_group[1] + 1
                    if output[i] == label[i]:
                        deep_group_correct[1] = deep_group_correct[1] + 1
                # bar3.update()
            # bar3.close()
        epoch_train_loss = np.mean(total_loss)
        epoch_eval_in_loss = np.mean(total_loss_eval_in)
        epoch_eval_out_loss = np.mean(total_loss_eval_out)
        print('Report: Epoch #{:02d}: Training loss: {:.06F}, Eval Loss: (I {:.06F})(O {:.06F}), Eval Accuracy: (I {:.03F}%-{:.03F}%)(O {:.03F}%-{:.03F}%), {:.03F}%, lr: {:.06F}'.format(
            epoch, epoch_train_loss, epoch_eval_in_loss, epoch_eval_out_loss, 100.0 * coarse_group_correct[0] / total_group[0], 100.0 * deep_group_correct[0] / total_group[0],
            100.0 * coarse_group_correct[1] / total_group[1], 100.0 * deep_group_correct[1] / total_group[1], 100.0 * (coarse_group_correct[0] + coarse_group_correct[1]) / (total_group[0] + total_group[1]),
            scheduler.get_lr()[0]
        ))
        state_dict = model.state_dict()
        torch.save(state_dict, '{}.{:03d}'.format(model_file, epoch))
        scheduler.step()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    dataset = ('datasets/tokenized/in_domain_train.reformed.csv', 'datasets/tokenized/in_domain_dev.reformed.csv', 'datasets/tokenized/out_of_domain_dev.reformed.csv')
    train(dataset, sys.argv[1], 0)
