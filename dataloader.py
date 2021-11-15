import numpy as np
import random
import torch
from torch.utils.data import Dataset

class GRDataset(Dataset):
    def __init__(self, data, m_query_list, m_user_list, q_movie_list, u_movie_list, movie_genre_list):
        
        self.data = data
        self.m_query_list = m_query_list
        self.m_user_list = m_user_list
        self.q_movie_list = q_movie_list
        self.u_movie_list = u_movie_list
        self.movie_genre_list = movie_genre_list

    def __getitem__(self, index):
        # print(self.data[1][0])
        uid = self.data[index][0]
        # print(uid)
        mid = self.data[index][1]
        # print(mid)
        qid = self.data[index][2]
        label = self.data[index][3]
        m_query = self.m_query_list[mid]
        m_user = self.m_user_list[mid]
        q_movie = self.q_movie_list[qid]
        u_movie = self.u_movie_list[uid]
        mg = self.movie_genre_list[mid]

        

        # u_items = self.u_items_list[uid]
        # u_users = self.u_users_list[uid]
        # u_users_items = self.u_users_items_list[uid]
        # i_users = self.i_users_list[iid]

        return (uid, mid, qid, label), m_query, m_user, q_movie, u_movie, mg
        # return (uid, iid, label), u_items, u_users, u_users_items, i_users

    def __len__(self):
        return len(self.data)
