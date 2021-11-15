import torch
import random
from itertools import chain

truncate_len = 30

"""
Ciao dataset info:
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

def collate_fn(batch_data):
    """This function will be used to pad the graph to max length in the batch
       It will be used in the Dataloader
    """
    uids, mids, qids, labels = [], [], [], []
    m_query, m_user, q_movie, u_movie, movie_genre = [], [], [], [], []
    m_query_len, m_user_len, q_movie_len, u_movie_len, mg_len = [], [], [], [], []
    # u_items, u_users, u_users_items, i_users = [], [], [], []
    # u_items_len, u_users_len, i_users_len = [], [], []

    for data, mq, mu, qm, um, mg in batch_data:
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
        else:
            q_movie.append(random.sample(qm, truncate_len))
        q_movie_len.append(min(len(qm), truncate_len))

        if len(um) <= truncate_len:
            u_movie.append(um)
        else:
            u_movie.append(random.sample(um, truncate_len))
        u_movie_len.append(min(len(um), truncate_len))

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

    return torch.LongTensor(uids), torch.LongTensor(mids), torch.LongTensor(qids), torch.FloatTensor(labels), \
            m_query_pad, m_user_pad, q_movie_pad, u_movie_pad, movie_genre, mg_offset


def new_collate_fn(batch_data):
    cnt1 = 5
    cnt2 = 5
    bubble = torch.zeros(512*cnt1, dtype=torch.int32).to(device)
    bubble_2 = torch.zeros(512*cnt1*cnt2, dtype=torch.int32).to(device)
    uids = []
    mids = []
    qids = []
    labels = []
    mgenres = []
    mgenres_len = []
    m_query, m_user, q_movie, u_movie, q_movie_genre, u_movie_genre = [], [], [], [], [], []
    m_query_len, m_user_len, q_movie_len, u_movie_len, q_movie_genre_len, u_movie_genre_len = [], [], [], [], [], []
    batch_size = len(batch_data)
    for node_data, nb_data, nb2_data in batch_data:
        (uid, mid, mgenre, qid, label) = data
        (mq, mu, qm, qmgenre, um, umgenre) = nb_data
        (mqm, mqmgenre, mum, mumgenre, qmq, qmu, umq, umu) = nb2_data
        uids.append(uid)
        mids.append(mid)
        qids.append(qid)
        mgenres.append(mgenre)
        mgenres_len.append(len(mgenre))
        labels.append(label)

        if len(mq) <= cnt1:
            m_query.append(mq)
        else:
            m_query.append(mq[:cnt1])
        m_query_len.append(min(len(mq), cnt1))
        
        if len(mu) <= cnt1:
            m_user.append(mu)
        else:
            m_user.append(mu[:cnt1])
        m_user_len.append(min(len(mu), cnt1))

        if len(qm) <= cnt1:
            q_movie.append(qm)
        else:
            q_movie.append(qm[:cnt1])
        q_movie_len.append(min(len(qm), cnt1))

        if len(um) <= cnt1:
            u_movie.append(um)
        else:
            u_movie.append(um[:cnt1])
        u_movie_len.append(min(len(um), cnt1))

        if len(qmgenre>cnt1):
            qmgenre = qmgenre[:cnt1]
        q_movie_genre.append(qmgenre[:])
        q_movie_genre_len.append([len(x) for x in qmgenre])
        if len(umgenre>cnt1):
            umgenre = umgenre[:cnt1]
        u_movie_genre.append(umgenre)
        u_movie_genre_len.append([len(x) for x in umgenre])



        for i in range(len(mqm)):
            if len(mqm[i]) > cnt2:
                mqm[i] = mqm[i][:cnt2]
            else:
                mqm[i].extend((0,0) for _ in range(cnt2-len(mqm[i])))
        m_q_movie.append(mqm)

        for i in range(len(mum)):
            if len(mum[i]) > cnt2:
                mum[i] = mum[i][:cnt2]
            else:
                mum[i].extend((0,0) for _ in range(cnt2-len(mum[i])))
        m_u_movie.append(mum)

        for i in range(len(qmq)):
            if len(qmq[i]) > cnt2:
                qmq[i] = qmq[i][:cnt2]
            else:
                qmq[i].extend((0,0) for _ in range(cnt2-len(qmq[i])))
        q_m_query.append(qmq)

        for i in range(len(qmu)):
            if len(qmu[i]) > cnt2:
                qmu[i] = qmu[i][:cnt2]
            else:
                qmu[i].extend((0,0) for _ in range(cnt2-len(qmu[i])))
        q_m_user.append(qmu)

        for i in range(len(umq)):
            if len(umq[i]) > cnt2:
                umq[i] = umq[i][:cnt2]
            else:
                umq[i].extend((0,0) for _ in range(cnt2-len(umq[i])))
        u_m_query.append(umq)

        for i in range(len(umu)):
            if len(umu[i]) > cnt2:
                umu[i] = umu[i][:cnt2]
            else:
                umu[i].extend((0,0) for _ in range(cnt2-len(umu[i])))
        u_m_user.append(umu)

        u_movie_genre

        # [[], [], []]
        m_query_pad = torch.zeros([batch_size, cnt1, 2], dtype=torch.long) 
        m_user_pad = torch.zeros([batch_size, cnt1, 2], dtype=torch.long) 
        q_movie_pad = torch.zeros([batch_size, cnt1, 2], dtype=torch.long) 
        u_movie_pad = torch.zeros([batch_size, cnt1, 2], dtype=torch.long) 
        m_q_movie_pad = torch.zeros([batch_size, cnt1, cnt2, 2], dtype=torch.long) 
        m_u_movie_pad = torch.zeros([batch_size, cnt1, cnt2, 2], dtype=torch.long) 
        u_m_query_pad = torch.zeros([batch_size, cnt1, cnt2, 2], dtype=torch.long) 
        u_m_user_pad = torch.zeros([batch_size, cnt1, cnt2, 2], dtype=torch.long) 
        q_m_query_pad = torch.zeros([batch_size, cnt1, cnt2, 2], dtype=torch.long)
        q_m_user_pad = torch.zeros([batch_size, cnt1, cnt2, 2], dtype=torch.long) 

        
        for idx in batch_size:
            m_query_pad[idx, :min(cnt1, len(mq)), :] = torch.LongTensor(mq)
            m_user_pad[idx, :min(cnt1, len(mu)), :] = torch.LongTensor(mu)
            q_movie_pad[idx, :min(cnt1, len(qm)), :] = torch.LongTensor(qm)
            u_movie_pad[idx, :min(cnt1, len(um)), :] = torch.LongTensor(um)
            m_q_movie_pad[idx, :min(cnt1, len(mqm)), :, :] = torch.LongTensor(mqm)
            m_u_movie_pad[idx, :min(cnt1, len(mum)), :, :] = torch.LongTensor(mum)
            u_m_query_pad[idx, :min(cnt1, len(umq)), :, :] = torch.LongTensor(umq)
            u_m_user_pad[idx, :min(cnt1, len(umu)), :, :] = torch.LongTensor(umu)
            q_m_query_pad[idx, :min(cnt1, len(qmq)), :, :] = torch.LongTensor(qmq)
            q_m_user_pad[idx, :min(cnt1, len(qmu)), :, :] = torch.LongTensor(qmu)

    nb = {
            "m":                    
                {                   
                    "q":{"qid":m_query_pad},
                    "u":{"uid":m_user_pad},
                },                  
            "u":                   
                {                   
                    "m":{"mid":u_movie_pad, "mgenre":bubble},    
                },                  
            "q":                    
                {                  
                    "m":{"mid":q_movie_pad, "mgenre":bubble},   
                },                  
        }

    nb2 = {
            "m":                    
                {                   
                    "q":
                    {
                        "m":{"mid":m_q_movie_pad, "mgenre":bubble_2},
                    },
                    "u":
                    {
                        "m":{"mid":m_u_movie_pad, "mgenre":bubble_2},
                    },              
                },                 
            "u":                    
                {                   
                    "m":
                    {
                        "q":{"qid":u_m_query_pad},
                        "u":{"uid":u_m_user_pad},     
                    },    
                },                  
            "q":                    
                {                   
                    "m":
                    {
                        "q":{"qid":q_m_query_pad},
                        "u":{"uid":q_m_user_pad},
                    },    
                },                  
            }

    return torch.LongTensor(uids), torch.LongTensor(mids), torch.LongTensor(qids), torch.FloatTensor(labels), \
            nb, nb2