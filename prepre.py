# -*- coding: utf-8 -*-
"""
create on Sep 24, 2019

@author: wangshuo
"""

import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.utils import shuffle

random.seed(1234)
top_k = 5


# user_cnt = 0
# movie_cnt = 0
# for ln in tqdm(range(len(rt))):
#     line = rt.iloc[ln]
#     uid = int(line[0])
#     mid = int(line[1])
#     if uid > user_cnt:
#         user_cnt = uid
#     if mid > movie_cnt:
#         movie_cnt = mid
# u_movie_list = []
# m_user_list = []
# m_query_list = []
# q_movie_list = []
# for uid in tqdm(range(user_cnt+1)):
#     hist = rt[rt['userId'] == uid]
#     u_movies = hist['movieId'].tolist()
#     u_ratings = hist['rating'].tolist()
#     if u_movies == []:
#         u_movie_list.append([0,0])
#     else:  
#         u_movie_list.append([(m, r) for m, r in zip(u_movies, u_ratings)])

# for mid in tqdm(range(movie_cnt+1)):
#     hist = rt[rt['movieId'] == mid]
#     m_users = hist['userId'].tolist()
#     m_ratings = hist['rating'].tolist()
#     if m_users == []:
#         m_user_list.append([0,0])
#     else:  
#         m_user_list.append([(u, r) for u, r in zip(m_users, m_ratings)])

# rel = pd.read_csv('ml-25m/genome-scores.csv')
# rel_rank = rel[rel.groupby(['movieId'])['relevance'].rank(method="first", ascending=False)<=top_k]
# for mid in tqdm(range(movie_cnt+1)):
#     hist = rel_rank[rel_rank['movieId'] == mid]
#     m_query = hist['tagId'].tolist()
#     m_rel = hist['relevance'].tolist()
#     if m_query == []:
#         m_query_list.append([0,0])
#     else:  
#         m_query_list.append([(q, r) for q, r in zip(m_query, m_rel)])

# for qid in tqdm(range(query_cnt+1)):
#     hist = rel_rank[rel_rank['tagId'] == qid]
#     q_movie = hist['movieId'].tolist()
#     q_rel = hist['relevance'].tolist()
#     if q_movie == []:
#         q_movie_list.append([0,0])
#     else:  
#         q_movie_list.append([(m, r) for m, r in zip(q_movie, q_rel)])

# maxlen = 5
# tagmap = {}
# gtags = pd.read_csv('ml-25m/genome-tags.csv')
# for idx in tqdm(range(len(gtags))):
#     line = gtags.iloc[idx]
#     tagid = line[0]
#     tagname = line[1]
#     tagmap[tagname] = tagid

# tag_list = []
# tags = pd.read_csv('ml-25m/tags.csv')
# for uid in tqdm(range(user_cnt+1)):
#     hist = tags[tags['userId'] == uid]
#     padlen = maxlen - len(hist)
#     u_movie = hist['movieId'].tolist()
#     u_tag = hist['tag'].tolist()
#     u_label = [1]*len(hist)
#     u_tagid = []
#     for u_tagname in u_tag:
#         if u_tagname in tagmap:
#             u_tagid.append(tagmap[u_tagname])
#         else:
#             u_tagid.append(0)
#     if padlen > 0:
#         u_movie = u_movie + [0]*padlen
#         u_tagid = u_tagid + [0]*padlen
#         u_label = u_label + [0]*padlen
#     tag_list.append([(uid, u_m, u_t, u_l) for u_m, u_t, u_l in zip(u_movie, u_tagid, u_label)])
    


# ret = []
# padlen = 209172 - 162542 + 1
with open('./datalist/mu_list.pkl', 'rb') as f:
    a = pickle.load(f)
    # ret = []
    padlen = 209172 - len(a)
    for i in tqdm(range(padlen)):
        a.append([(0,0)])
    # for i in tqdm(range(len(a))):
    #     # a.append([0,0])
    #     if not isinstance(a[i][0], tuple):
    #         a[i] = [(0,0)]

with open('./datalist/mu_list.pkl', 'wb') as f:
	# pickle.dump(u_movie_list, f, pickle.HIGHEST_PROTOCOL)
	# pickle.dump(m_user_list, f, pickle.HIGHEST_PROTOCOL)
	# pickle.dump(m_query_list, f, pickle.HIGHEST_PROTOCOL)
	# pickle.dump(q_movie_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(a, f, pickle.HIGHEST_PROTOCOL)
	# pickle.dump((user_count, item_count, rate_count), f, pickle.HIGHEST_PROTOCOL)


