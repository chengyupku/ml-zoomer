import torch
import random
from itertools import chain

truncate_len = 30

"""
movielens-25m dataset info:
Avg number of items rated per user: 38.3
Avg number of users interacted per user: 2.7
Avg number of users connected per item: 16.4
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

node_feature_name_config = {
    "m":["mid", "mgenre"],
    "q":["qid"],
    "u":["uid"],
}
nb_config = {
    "m":["q", "u"],
    "q":["m"],
    "u":["m"],
}

# def collate_fn(batch_data):
#     """This function will be used to pad the graph to max length in the batch
#        It will be used in the Dataloader
#     """
#     uids, mids, qids, labels = [], [], [], []
#     m_query, m_user, q_movie, u_movie, movie_genre = [], [], [], [], []
#     m_query_len, m_user_len, q_movie_len, u_movie_len, mg_len = [], [], [], [], []
#     # u_items, u_users, u_users_items, i_users = [], [], [], []
#     # u_items_len, u_users_len, i_users_len = [], [], []

#     for data, mq, mu, qm, um, mg in batch_data:
#         (uid, mid, qid, label) = data
#         uids.append(uid)
#         mids.append(mid)
#         qids.append(qid)
#         labels.append(label)
#         movie_genre.append(mg)
#         mg_len.append(len(mg))

#         # user-items    
#         if len(mq) <= truncate_len:
#             m_query.append(mq)
#         else:
#             m_query.append(random.sample(mq, truncate_len))
#         m_query_len.append(min(len(mq), truncate_len))
        
#         if len(mu) <= truncate_len:
#             m_user.append(mu)
#         else:
#             m_user.append(random.sample(mu, truncate_len))
#         m_user_len.append(min(len(mu), truncate_len))

#         if len(qm) <= truncate_len:
#             q_movie.append(qm)
#         else:
#             q_movie.append(random.sample(qm, truncate_len))
#         q_movie_len.append(min(len(qm), truncate_len))

#         if len(um) <= truncate_len:
#             u_movie.append(um)
#         else:
#             u_movie.append(random.sample(um, truncate_len))
#         u_movie_len.append(min(len(um), truncate_len))

#         # u_users_len.append(min(len(u_users_u), truncate_len))	

#         # # item-users
#         # if len(i_users_i) <= truncate_len:
#         #     i_users.append(i_users_i)
#         # else:
#         #     i_users.append(random.sample(i_users_i, truncate_len))
#         # i_users_len.append(min(len(i_users_i), truncate_len))

#     batch_size = len(batch_data)

#     # padding
#     m_query_maxlen = max(m_query_len)
#     m_user_maxlen = max(m_user_len)
#     q_movie_maxlen = max(q_movie_len)
#     u_movie_maxlen = max(u_movie_len)

#     # u_items_maxlen = max(u_items_len)
#     # u_users_maxlen = max(u_users_len)
#     # i_users_maxlen = max(i_users_len)
    

#     m_query_pad = torch.zeros([batch_size, m_query_maxlen, 2], dtype=torch.long) 
#     for i, mq in enumerate(m_query):
#         m_query_pad[i, :len(mq),:] = torch.LongTensor(mq)

#     m_user_pad = torch.zeros([batch_size, m_user_maxlen, 2], dtype=torch.long) 
#     for i, mu in enumerate(m_user):
#         m_user_pad[i, :len(mu),:] = torch.LongTensor(mu)

#     q_movie_pad = torch.zeros([batch_size, q_movie_maxlen, 2], dtype=torch.long) 
#     for i, qm in enumerate(q_movie):
#         q_movie_pad[i, :len(qm),:] = torch.LongTensor(qm)
    
#     u_movie_pad = torch.zeros([batch_size, u_movie_maxlen, 2], dtype=torch.long) 
#     for i, um in enumerate(u_movie):
#         u_movie_pad[i, :len(um),:] = torch.LongTensor(um)

#     movie_genre = list(chain.from_iterable(movie_genre))
#     mg_offset = [0]
#     for i in mg_len[:-1]:
#         mg_offset.append(mg_offset[-1]+i)
#     movie_genre = torch.LongTensor(movie_genre)
#     mg_offset = torch.LongTensor(mg_offset)

#     # u_item_pad = torch.zeros([batch_size, u_items_maxlen, 2], dtype=torch.long)
#     # for i, ui in enumerate(u_items):
#     #     u_item_pad[i, :len(ui), :] = torch.LongTensor(ui)
    
#     # u_user_pad = torch.zeros([batch_size, u_users_maxlen], dtype=torch.long)
#     # for i, uu in enumerate(u_users):
#     #     u_user_pad[i, :len(uu)] = torch.LongTensor(uu)
    
#     # u_user_item_pad = torch.zeros([batch_size, u_users_maxlen, u_items_maxlen, 2], dtype=torch.long)
#     # for i, uu_items in enumerate(u_users_items):
#     #     for j, ui in enumerate(uu_items):
#     #         u_user_item_pad[i, j, :len(ui), :] = torch.LongTensor(ui)

#     # i_user_pad = torch.zeros([batch_size, i_users_maxlen, 2], dtype=torch.long)
#     # for i, iu in enumerate(i_users):
#     #     i_user_pad[i, :len(iu), :] = torch.LongTensor(iu)

#     return torch.LongTensor(uids), torch.LongTensor(mids), torch.LongTensor(qids), torch.FloatTensor(labels), \
#             m_query_pad, m_user_pad, q_movie_pad, u_movie_pad, movie_genre, mg_offset


def collate_fn(batch_data):
    """This function will be used to pad the graph to max length in the batch
       It will be used in the Dataloader
    """
    uids, mids, qids, labels = [], [], [], []
    m_query, m_user, q_movie, u_movie, movie_genre, u_mgenre, q_mgenre = [], [], [], [], [], [], []
    m_query_len, m_user_len, q_movie_len, u_movie_len, mg_len, q_mgenre_len, u_mgenre_len = [], [], [], [], [], [], []
    # u_items, u_users, u_users_items, i_users = [], [], [], []
    # u_items_len, u_users_len, i_users_len = [], [], []

    for data, mq, mu, qm, um, mg, u_mg, q_mg in batch_data:
        (uid, mid, qid, label) = data
        uids.append(uid)
        mids.append(mid)
        qids.append(qid)
        labels.append(label)
        movie_genre.append(mg)
        mg_len.append(len(mg))

        # user-items    
        if len(mq) <= truncate_len:
            m_query.append(mq)
        else:
            m_query.append(random.sample(mq, truncate_len))
        m_query_len.append(min(len(mq), truncate_len))
        
        if len(mu) <= truncate_len:
            m_user.append(mu)
        else:
            m_user.append(random.sample(mu, truncate_len))
        m_user_len.append(min(len(mu), truncate_len))

        if len(qm) <= truncate_len:
            q_movie.append(qm)
            q_mgenre.append(q_mg)
        else:
            q_movie.append(random.sample(qm, truncate_len))
            q_mgenre.append(random.sample(q_mg, truncate_len))
        q_movie_len.append(min(len(qm), truncate_len))
        q_mgenre_len.append(min(len(qm), truncate_len))

        if len(um) <= truncate_len:
            u_movie.append(um)
            u_mgenre.append(u_mg)
        else:
            u_movie.append(random.sample(um, truncate_len))
            u_mgenre.append(random.sample(u_mg, truncate_len))
        u_movie_len.append(min(len(um), truncate_len))
        u_mgenre_len.append(min(len(um), truncate_len))

        # u_users_len.append(min(len(u_users_u), truncate_len))	

        # # item-users
        # if len(i_users_i) <= truncate_len:
        #     i_users.append(i_users_i)
        # else:
        #     i_users.append(random.sample(i_users_i, truncate_len))
        # i_users_len.append(min(len(i_users_i), truncate_len))

    batch_size = len(batch_data)

    # padding
    m_query_maxlen = max(m_query_len)
    m_user_maxlen = max(m_user_len)
    q_movie_maxlen = max(q_movie_len)
    u_movie_maxlen = max(u_movie_len)

    # u_items_maxlen = max(u_items_len)
    # u_users_maxlen = max(u_users_len)
    # i_users_maxlen = max(i_users_len)
    

    m_query_pad = torch.zeros([batch_size, m_query_maxlen, 2], dtype=torch.long) 
    for i, mq in enumerate(m_query):
        m_query_pad[i, :len(mq),:] = torch.LongTensor(mq)

    m_user_pad = torch.zeros([batch_size, m_user_maxlen, 2], dtype=torch.long) 
    for i, mu in enumerate(m_user):
        m_user_pad[i, :len(mu),:] = torch.LongTensor(mu)

    q_movie_pad = torch.zeros([batch_size, q_movie_maxlen, 2], dtype=torch.long) 
    for i, qm in enumerate(q_movie):
        q_movie_pad[i, :len(qm),:] = torch.LongTensor(qm)
    
    u_movie_pad = torch.zeros([batch_size, u_movie_maxlen, 2], dtype=torch.long) 
    for i, um in enumerate(u_movie):
        u_movie_pad[i, :len(um),:] = torch.LongTensor(um)

    movie_genre = list(chain.from_iterable(movie_genre))
    mg_offset = [0]
    for i in mg_len[:-1]:
        mg_offset.append(mg_offset[-1]+i)
    movie_genre = torch.LongTensor(movie_genre)
    mg_offset = torch.LongTensor(mg_offset)

    x = []
    x_offset = [0]
    u_mgenre_offset = []
    for i in range(len(u_mgenre)):
        x_offset = [x_offset[-1]]
        x.append(list(chain.from_iterable(u_mgenre[i])))
        for j in u_mgenre[i]:
            x_offset.append(x_offset[-1]+len(j))  # 有len+1个数
        if u_mgenre_len[i]==u_movie_maxlen:
            x_offset = x_offset[:-1]
        elif u_mgenre_len[i]<u_movie_maxlen:
            for _ in range(u_movie_maxlen-u_mgenre_len[i]-1):
                x_offset.append(x_offset[-1])
        u_mgenre_offset.append(x_offset)
    u_mgenre = list(chain.from_iterable(x))
    u_mgenre = torch.LongTensor(u_mgenre)
    u_mgenre_offset = list(chain.from_iterable(u_mgenre_offset))
    u_mgenre_offset = torch.LongTensor(u_mgenre_offset)


    x = []
    x_offset = [0]
    q_mgenre_offset = []
    for i in range(len(q_mgenre)):
        x_offset = [x_offset[-1]]
        x.append(list(chain.from_iterable(q_mgenre[i])))
        for j in q_mgenre[i]:
            x_offset.append(x_offset[-1]+len(j))  # 有len+1个数
        if q_mgenre_len[i] == q_movie_maxlen:
            x_offset = x_offset[:-1]
        elif q_mgenre_len[i] < q_movie_maxlen:
            for _ in range(q_movie_maxlen - q_mgenre_len[i]-1):
                x_offset.append(x_offset[-1])
        q_mgenre_offset.append(x_offset)
    q_mgenre = list(chain.from_iterable(x))
    q_mgenre = torch.LongTensor(q_mgenre)
    q_mgenre_offset = list(chain.from_iterable(q_mgenre_offset))
    q_mgenre_offset = torch.LongTensor(q_mgenre_offset)
    # u_item_pad = torch.zeros([batch_size, u_items_maxlen, 2], dtype=torch.long)
    # for i, ui in enumerate(u_items):
    #     u_item_pad[i, :len(ui), :] = torch.LongTensor(ui)
    
    # u_user_pad = torch.zeros([batch_size, u_users_maxlen], dtype=torch.long)
    # for i, uu in enumerate(u_users):
    #     u_user_pad[i, :len(uu)] = torch.LongTensor(uu)
    
    # u_user_item_pad = torch.zeros([batch_size, u_users_maxlen, u_items_maxlen, 2], dtype=torch.long)
    # for i, uu_items in enumerate(u_users_items):
    #     for j, ui in enumerate(uu_items):
    #         u_user_item_pad[i, j, :len(ui), :] = torch.LongTensor(ui)

    # i_user_pad = torch.zeros([batch_size, i_users_maxlen, 2], dtype=torch.long)
    # for i, iu in enumerate(i_users):
    #     i_user_pad[i, :len(iu), :] = torch.LongTensor(iu)

    return torch.LongTensor(uids).to(device), torch.LongTensor(mids).to(device), torch.LongTensor(qids).to(device), torch.FloatTensor(labels).to(device), \
            m_query_pad.to(device), m_user_pad.to(device), q_movie_pad.to(device), u_movie_pad.to(device), \
            (movie_genre.to(device), mg_offset.to(device)),     \
            (u_mgenre.to(device), u_mgenre_offset.to(device)),  \
            (q_mgenre.to(device), q_mgenre_offset.to(device))   \
    
    #  return torch.LongTensor(uids), torch.LongTensor(mids), torch.LongTensor(qids), torch.FloatTensor(labels), \
    #         m_query_pad, m_user_pad, q_movie_pad, u_movie_pad, movie_genre, mg_offset, u_mgenre, u_mgenre_offset, q_mgenre, q_mgenre_offset