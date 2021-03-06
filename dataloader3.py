import numpy as np
import random
import torch
from torch.utils.data import Dataset

class GRDataset(Dataset):
    def __init__(self, data, m_query_list, m_user_list, q_movie_list, u_movie_list, movie_genre_list):
        
        self.data = data
        self.m_query_list = m_query_list
        self.m_user_list = m_user_list
        # print("m_user_list", len(m_user_list), flush=True)
        self.q_movie_list = q_movie_list
        # print("q_movie_list", len(q_movie_list), flush=True)
        self.u_movie_list = u_movie_list
        # print("u_movie_list", len(u_movie_list), flush=True)
        self.movie_genre_list = movie_genre_list
        # print("movie_genre_list", len(movie_genre_list), flush=True)

    def __getitem__(self, index):
        print(index, flush=True)
        # print(self.data[1][0])
        uid = self.data[index][0]
        # print(uid)
        mid = self.data[index][1]
        mgenre = self.movie_genre_list[mid]
        # print(mid)
        qid = self.data[index][2]
        label = self.data[index][3]

        mq = self.m_query_list[mid]
        mu = self.m_user_list[mid]
        qm = self.q_movie_list[qid]
        um = self.u_movie_list[uid]
        qmgenre = []
        umgenre = []
        # print(qm)
        for i in qm:
            qmgenre.append(self.movie_genre_list[i[0]])
        for i in um:
            umgenre.append(self.movie_genre_list[i[0]])

        mqm = []
        mum = []
        qmq = []
        qmu = []
        umq = []
        umu = []
        for i in mq:
            mqm.append(self.q_movie_list[i[0]]) # [[], [], [],...]
        for i in mu:
            mum.append(self.u_movie_list[i[0]]) # [[], [], [],...]
        for i in qm:
            qmq.append(self.m_query_list[i[0]]) # [[], [], [],...]
            qmu.append(self.m_user_list[i[0]])
        for i in um:
            umq.append(self.m_query_list[i[0]]) # [[], [], [],...]
            umu.append(self.m_user_list[i[0]])
        mqmgenre = []
        mumgenre = []
        for j in mqm:
            tmp = []
            for i in j:
                tmp.append(self.movie_genre_list[i[0]])
            mqmgenre.append(tmp)    #[[[], [], []]] 3???list

        for j in mum:
            tmp = []
            for i in j:
                tmp.append(self.movie_genre_list[i[0]])
            mumgenre.append(tmp)

        return (uid, mid, mgenre, qid, label), (mq, mu, qm, qmgenre, um, umgenre), (mqm, mqmgenre, mum, mumgenre, qmq, qmu, umq, umu)
        # return (uid, iid, label), u_items, u_users, u_users_items, i_users

    def __len__(self):
        return len(self.data)
