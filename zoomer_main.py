#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on 30 Sep, 2019

@author: wangshuo
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
from zoomer_2 import Zoomer
from dataloader import GRDataset

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_path', default='datasets/Ciao/', help='dataset directory path: datasets/Ciao/Epinions')
# parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
# parser.add_argument('--embed_dim', type=int, default=64, help='the dimension of embedding')
# parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
# parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
# parser.add_argument('--lr_dc_step', type=int, default=30, help='the number of steps after which the learning rate decay')
# parser.add_argument('--test', action='store_true', help='test')
# args = parser.parse_args()
# print(args)

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

def main():
    print('Loading data...', flush=True)
    user_count = 162541
    item_count = 119571
    query_count = 1128
    embed_dim = 64
    genre_count = 20
    sp = 0.8
    with open('./taglist.pkl', 'rb') as f:
        data_set = pickle.load(f)
        random.shuffle(data_set)
        data_set_len = len(data_set)
        train_set_len = int(data_set_len*sp)
        valid_set_len = data_set_len-train_set_len
        train_set = data_set[:train_set_len]
        valid_set = data_set[train_set_len:]
        # valid_set = pickle.load(f)
        # test_set = pickle.load(f)

    with open('./datalist/mq_list.pkl', 'rb') as f:
        m_query_list = pickle.load(f)
    with open('./datalist/mu_list.pkl', 'rb') as f:
        m_user_list = pickle.load(f)
    with open('./datalist/qm_list.pkl', 'rb') as f:
        q_movie_list = pickle.load(f)
    with open('./datalist/um_list.pkl', 'rb') as f:
        u_movie_list = pickle.load(f)
    with open('./datalist/movie_genre_list.pkl', 'rb') as f:
        movie_genre_list = pickle.load(f)
    # with open(args.dataset_path + 'list.pkl', 'rb') as f:
    #     u_users_list = pickle.load(f)
    #     u_users_items_list = pickle.load(f)
    #     i_users_list = pickle.load(f)
    #     (user_count, item_count, rate_count) = pickle.load(f)
    
    train_data = GRDataset(train_set, m_query_list, m_user_list, q_movie_list, u_movie_list, movie_genre_list)
    valid_data = GRDataset(valid_set, m_query_list, m_user_list, q_movie_list, u_movie_list, movie_genre_list)
    # test_data = GRDataset(test_set, u_items_list, u_users_list, u_users_items_list, i_users_list)
    train_loader = DataLoader(train_data, batch_size = 512, shuffle = True, collate_fn = collate_fn)
    valid_loader = DataLoader(valid_data, batch_size = 512, shuffle = True, collate_fn = collate_fn)
    # test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)

    model = Zoomer(user_count+1, item_count*5, genre_count+1, query_count+1, embed_dim).to(device)

    # if args.test:
    #     print('Load checkpoint and testing...')
    #     ckpt = torch.load('best_checkpoint.pth.tar')
    #     model.load_state_dict(ckpt['state_dict'])
    #     mae, rmse = validate(test_loader, model)
    #     print("Test: MAE: {:.4f}, RMSE: {:.4f}".format(mae, rmse))
    #     return

    lr = 0.001
    lr_dc = 0.1
    lr_dc_step = 30
    optimizer = optim.RMSprop(model.parameters(), lr)
    criterion = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size = lr_dc_step, gamma = lr_dc)

    epochs = 100
    for epoch in range(epochs):
        # train for one epoch
        scheduler.step(epoch = epoch)
        trainForEpoch(train_loader, model, optimizer, epoch, epochs, criterion, log_aggr = 100)

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
    for i, (uids, mids, qids, labels, m_querys, m_users, q_movies, u_movies, movie_genre, mg_offset) in enumerate(train_loader):
        uids = uids.to(device)
        mids = mids.to(device)
        qids = qids.to(device)
        labels = labels.to(device)
        m_querys = m_querys.to(device)
        m_users = m_users.to(device)
        q_movies = q_movies.to(device)
        u_movies = u_movies.to(device)
        movie_genre = movie_genre.to(device)
        mg_offset = mg_offset.to(device)
        
        optimizer.zero_grad()
        outputs = model(uids, mids, qids, m_querys, m_users, q_movies, u_movies, movie_genre, mg_offset)
        
        # outputs_np = outputs.cpu().detach().numpy()
        # labels_np = labels.cpu().detach().numpy()
        # auc = auc_calculate(labels_np, outputs_np)
        # print('AUC: ', auc)
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
        for uids, mids, qids, labels, m_querys, m_users, q_movies, u_movies, movie_genre, mg_offset in tqdm(valid_loader):
            uids = uids.to(device)
            mids = mids.to(device)
            qids = qids.to(device)
            labels = labels.to(device)
            m_querys = m_querys.to(device)
            m_users = m_users.to(device)
            q_movies = q_movies.to(device)
            u_movies = u_movies.to(device)
            movie_genre = movie_genre.to(device)
            mg_offset = mg_offset.to(device)
            preds = model(uids, mids, qids, m_querys, m_users, q_movies, u_movies, movie_genre, mg_offset)
            error = torch.abs(preds.squeeze(1) - labels)
            errors.extend(error.data.cpu().numpy().tolist())
            # true_value = labels.cpu().numpy()
            # pred_value = preds.cpu().numpy()
            # print(true_value)
            # print(pred_value)
            get_roc_samples(preds.cpu(), labels.cpu(), TP, FP, TN, FN, bucket_num=bucket_num)
    
    auc = auc_calculate(TP, FP, TN, FN, bucket_num=bucket_num)
    print(auc)

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.power(errors, 2)))
    return mae, rmse, auc


if __name__ == '__main__':
    main()
