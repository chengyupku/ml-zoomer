#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on 20 Oct, 2021

@author: yucheng
"""

import os
import time
import argparse
import pickle
import numpy as np
import random
from tqdm import tqdm
from os.path import join
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
# from torch.backends import cudnn
from sklearn.metrics import roc_auc_score

from utils import collate_fn
from zoomer import Zoomer
from dataloader import GRDataset
torch.set_printoptions(profile="full")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='./datalist/', help='dataset directory path')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
parser.add_argument('--embed_dim', type=int, default=64, help='the dimension of embedding')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=30, help='the number of steps after which the learning rate decay')
parser.add_argument('--sp', type=float, default=0.8, help='the proportion of the training data to the total data')
args = parser.parse_args()
print(args)

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

def main():
    print('Loading data...', flush=True)
    user_count = 162541
    item_count = 119571
    query_count = 1128
    genre_count = 20
    with open(args.dataset_path+'new_taglist.pkl', 'rb') as f:
        data_set = pickle.load(f)
        random.shuffle(data_set)
        data_set_len = len(data_set)
        train_set_len = int(data_set_len * args.sp)
        valid_set_len = data_set_len-train_set_len
        train_set = data_set[:train_set_len]
        valid_set = data_set[train_set_len:]
        # valid_set = pickle.load(f)
        # test_set = pickle.load(f)

    with open(args.dataset_path+'m-q.pkl', 'rb') as f:
        m-q_list = pickle.load(f)
    with open(args.dataset_path+'m-u.pkl', 'rb') as f:
        m-u_list = pickle.load(f)
    with open(args.dataset_path+'q-m.pkl', 'rb') as f:
        q-m_list = pickle.load(f)
    with open(args.dataset_path+'u-m.pkl', 'rb') as f:
        u-m_list = pickle.load(f)
    with open(args.dataset_path+'movie_genre.pkl', 'rb') as f:
        movie_genre_list = pickle.load(f)
    
    train_data = GRDataset(train_set, m-q_list, m-u_list, q-m_list, u-m_list, movie_genre_list)
    valid_data = GRDataset(valid_set, m-q_list, m-u_list, q-m_list, u-m_list, movie_genre_list)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
    valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)

    model = Zoomer(user_count+1, item_count*5, genre_count+1, query_count+1, args.embed_dim).to(device)

    # if args.test:
    #     print('Load checkpoint and testing...')
    #     ckpt = torch.load('best_checkpoint.pth.tar')
    #     model.load_state_dict(ckpt['state_dict'])
    #     mae, rmse = validate(test_loader, model)
    #     print("Test: MAE: {:.4f}, RMSE: {:.4f}".format(mae, rmse))
    #     return

    optimizer = optim.RMSprop(model.parameters(), args.lr)
    criterion = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)

    for epoch in range(args.epoch):
        # train for one epoch
        scheduler.step(epoch = epoch)
        trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr = 100)

        mae, rmse, auc = validate(valid_loader, model)

        # store best loss and save a model checkpoint
        # ckpt_dict = {
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'optimizer': optimizer.state_dict()
        # }

        # torch.save(ckpt_dict, 'latest_checkpoint.pth.tar')

        # if epoch == 0:
        #     best_mae = mae
        # elif mae < best_mae:
        #     best_mae = mae
        #     torch.save(ckpt_dict, 'best_checkpoint.pth.tar')

        print('Epoch {} validation: MAE: {:.4f}, RMSE: {:.4f}, AUC: {:.4f}'.format(epoch, mae, rmse, auc))
        # print('Epoch {} validation: MAE: {:.4f}, RMSE: {:.4f}, Best MAE: {:.4f}'.format(epoch, mae, rmse, best_mae))

def get_roc_samples(scores, labels, TP, FP, TN, FN, bucket_num=200):
    gaps = 1/bucket_num
    ones = torch.ones(scores.shape[-1], dtype=torch.int32)
    zeros = torch.zeros(scores.shape[-1], dtype=torch.int32)
    for i in range(bucket_num, 0, -1):
        threshold = i*gaps
        TP[i] += (torch.where(scores>=threshold, ones, zeros) * torch.where(labels==1, ones, zeros)).sum()
        FP[i] += (torch.where(scores>=threshold, ones, zeros) * torch.where(labels==0, ones, zeros)).sum()
        TN[i] += (torch.where(scores<threshold, ones, zeros) * torch.where(labels==0, ones, zeros)).sum()
        FN[i] += (torch.where(scores<threshold, ones, zeros) * torch.where(labels==1, ones, zeros)).sum()
    return 

def auc_calculate(TP, FP, TN, FN, bucket_num=100):
    gaps = 1/bucket_num
    lastFPR = 0
    area = 0
    for i in range(bucket_num, 0, -1):
        TPR = TP[i] / (TP[i]+FN[i])
        FPR = FP[i] / (FP[i]+TN[i])
        # print(TP)
        # print(FP)
        # print(TN)
        # print(FN)
        # print(i, TPR, FPR)
        # print("TPR", TPR)
        # print("FPR", FPR)
        area += TPR*(FPR-lastFPR)
        lastFPR = FPR
    return area

def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    # for i, (uids, iids, labels, u_items, u_users, u_users_items, i_users) in tqdm(enumerate(train_loader), total=len(train_loader)):
    for i, (uids, mids, qids, labels, m_querys, m_users, q_movies, u_movies, m_genre, \
             u_mgenre, q_mgenre) in enumerate(train_loader):
        
        optimizer.zero_grad()
        outputs = model(uids, mids, qids, m_genre, m_querys, m_users, u_movies, q_movies, u_mgenre, q_mgenre)
        
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step() 

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                  len(uids) / (time.time() - start)))

        start = time.time()
        if i % 100==0:
            break

def validate(valid_loader, model):
    model.eval()
    errors = []
    bucket_num = 200
    TP = np.zeros((bucket_num+1,), dtype=np.int)
    FP = np.zeros((bucket_num+1,), dtype=np.int)
    TN = np.zeros((bucket_num+1,), dtype=np.int)
    FN = np.zeros((bucket_num+1,), dtype=np.int)
    with torch.no_grad():
        for uids, mids, qids, labels, m_querys, m_users, q_movies, u_movies, m_genre, \
             u_mgenre, q_mgenre in tqdm(valid_loader):

            preds = model(uids, mids, qids, m_genre, m_querys, m_users, u_movies, q_movies, u_mgenre, q_mgenre)
            error = torch.abs(preds.squeeze(1) - labels)
            errors.extend(error.data.cpu().numpy().tolist())
            true_value = labels.cpu().numpy()
            pred_value = preds.cpu().numpy()
            # print(true_value)
            # print(pred_value)
            get_roc_samples(preds.cpu(), labels.cpu(), TP, FP, TN, FN, bucket_num=bucket_num)
    
    # auc = auc_calculate(TP, FP, TN, FN, bucket_num=bucket_num)
    auc = roc_auc_score(true_value, pred_value)
    print(auc)

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.power(errors, 2)))
    return mae, rmse, auc


if __name__ == '__main__':
    main()
