from zoomer import zoomer
from dataset import MovieRankDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
# --------------- hyper-parameters------------------
user_max_dict={
    'uid':6041,  # 6040 users
    'gender':2,
    'age':7,
    'job':21
}

movie_max_dict={
    'mid':3953,  # 3952 movies
    'mtype':19,
    'mword':5215   # 5215 words
}

convParams={
    'kernel_sizes':[2,3,4,5]
}


def train(model,num_epochs=5, lr=0.0001):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)

    datasets = MovieRankDataset(pkl_file='data.p')
    dataloader = DataLoader(datasets,batch_size=64,shuffle=True)

    losses=[]
    for epoch in range(num_epochs):
        loss_all = 0
        num_correct_all = 0
        for i_batch,sample_batch in enumerate(dataloader):

            user_inputs = sample_batch['user_inputs']
            movie_inputs = sample_batch['movie_inputs']
            # target = sample_batch['target'].to(device)
            target = sample_batch['target'].to(device)
            model.zero_grad()

            tag_rank, pred_rating = model(user_inputs, movie_inputs)

            loss = loss_function(tag_rank, target)
            num_correct = torch.eq(pred_rating, target).sum()
            num_correct_all += num_correct
            if i_batch%1 ==0:
                # print(pred_rating)
                print("loss: ", loss.item())
                print("acc: ", num_correct.item() / 256)
            loss_all += loss
            loss.backward(retain_graph=True)
            optimizer.step()
        print('Epoch {}:\t loss:{}'.format(epoch,loss_all))



if __name__=='__main__':
    all_user = pd.read_pickle('user.pickle')
    all_movie = pd.read_pickle('movie.pickle')
    adj_mat = torch.tensor(np.load('adj.npy')).to(torch.float32)
    # adj_mat = torch.eye(6041+3953)
    model = zoomer(user_max_dict=user_max_dict, movie_max_dict=movie_max_dict, 
                all_user=all_user, all_movie=all_movie, adj_mat=adj_mat, convParams=convParams)
    model=model.to(device)
    # model = nn.DataParallel(model)
    # print(device)
    # model=model.to(device)
    # for name, param in model.named_parameters():
    #     if param.requires_grad and param.is_leaf:
    #         print(name)
    # train model
    train(model=model,num_epochs=10)

