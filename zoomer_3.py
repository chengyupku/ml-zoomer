import torch
from torch import nn
import torch.nn.functional as F
import math
from utils import nb_config

class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)

class _GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, bias=True, activation=F.relu):
        super(_GraphAttentionLayer, self).__init__()
        self.attnlayer = _AttentionLayer(input_dim, bias, activation)

    def forward(self, q, k, v):
        y = self.attnlayer(q, k, v)
        return y

class _AttentionLayer(nn.Module):
    def __init__(self, input_dim, bias=True, activation=F.relu):
        super(_AttentionLayer, self).__init__()
        self.linear_q = nn.Linear(input_dim, input_dim, bias)
        self.linear_k = nn.Linear(input_dim, input_dim, bias)
        self.linear_v = nn.Linear(input_dim, input_dim, bias)
        self.linear_o = nn.Linear(input_dim, input_dim, bias)
        self.activation = activation

    def forward(self, q, k, v):
        # q = q.unsqueeze(1)
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        y = ScaledDotProductAttention()(q, k, v)
        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y



class _MultiLayerPercep(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_MultiLayerPercep, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=True),
            nn.ReLU(),            
            nn.Linear(input_dim // 2, output_dim, bias=True),
        )

    def forward(self, x):
        return self.mlp(x)


class _Aggregation(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_Aggregation, self).__init__()
        self.aggre = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.aggre(x)


class _UserModel(nn.Module):
    ''' User modeling to learn user latent factors.
    User modeling leverages two types aggregation: item aggregation and social aggregation
    '''
    def __init__(self, emb_dim, user_emb, item_id_emb, item_genre_emb, query_emb, semantic_level_attn=True):
        super(_UserModel, self).__init__()
        self.user_emb = user_emb
        self.item_id_emb = item_id_emb
        self.item_genre_emb = item_genre_emb
        self.query_emb = query_emb
        self.emb_dim = emb_dim
        self.semantic_level_attn = semantic_level_attn

        self.g_v = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)

        self.user_items_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_items = _Aggregation(self.emb_dim, self.emb_dim)

        self.user_users_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_neigbors = _Aggregation(self.emb_dim, self.emb_dim)
    
        self.gat = _GraphAttentionLayer(input_dim=emb_dim)
        self.semantic_attn_layer = _AttentionLayer(input_dim=emb_dim)
        self.outlayer = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # used for preventing zero div error when calculating softmax score
        self.eps = 1e-10

    def forward(self, uids, qids, u_movies):
        # item aggregation
        node_embedding = self.user_emb(uids)
        ex_node_embedding = self.query_emb(qids)
        # ex_node_embedding = node_embedding
        nb_movie_embedding = self.item_id_emb(u_movies[:,:,0])
        # print("node_embedding", node_embedding.shape, flush=True)
        # print("nb_movie_embedding", nb_movie_embedding.shape, flush=True)
        # print("nb_movie_embedding", nb_movie_embedding.shape, flush=True)
        nb_aggregated_embedding = self.gat(node_embedding.unsqueeze(1), nb_movie_embedding, nb_movie_embedding).squeeze(1)
        # print("nb_aggregated_embedding", nb_aggregated_embedding.shape,flush=True)
        query_embedding = torch.add(node_embedding, ex_node_embedding)
        if self.semantic_level_attn:
            key_embedding = torch.concat([node_embedding.unsqueeze(1), \
                                    nb_aggregated_embedding.unsqueeze(1)], axis=1) # [512, 2, 64]
            outputs = self.semantic_attn_layer(node_embedding.unsqueeze(1), key_embedding, key_embedding).squeeze(1)
        else:
            y = torch.concat([node_embedding, nb_aggregated_embedding], axis=1)
            outputs = self.outlayer(y)
        # print("outputs", outputs.shape,flush=True)
        return outputs

class _ItemModel(nn.Module):
    '''Item modeling to learn item latent factors.
    '''
    def __init__(self, emb_dim, user_emb, item_id_emb, item_genre_emb, query_emb, semantic_level_attn=True):
        super(_ItemModel, self).__init__()
        self.emb_dim = emb_dim
        self.user_emb = user_emb
        self.item_id_emb = item_id_emb
        self.item_genre_emb = item_genre_emb
        self.query_emb = query_emb
        self.semantic_level_attn = semantic_level_attn

        self.g_u = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        
        self.item_users_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_users = _Aggregation(self.emb_dim, self.emb_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gat = _GraphAttentionLayer(input_dim=emb_dim)
        self.semantic_attn_layer = _AttentionLayer(input_dim=emb_dim)
        self.outlayer = _MultiLayerPercep(3 * self.emb_dim, self.emb_dim)
        # used for preventing zero div error when calculating softmax score
        self.eps = 1e-10

    def forward(self, mids, m_querys, m_users):
        # user aggregation
        node_embedding = self.item_id_emb(mids)    # [512, 64]
        # print("m_querys ", m_querys, flush=True)
        # print("m_querys ", m_querys.shape, flush=True)
        # print("m_users ", m_users, flush=True)
        # print("m_users ", m_users.shape, flush=True)
        # print("m_querys ", m_querys[:,:,0], flush=True)
        # print("m_querys ", m_querys[:,:,0].shape, flush=True)
        # print("m_users ", m_users[:,:,0], flush=True)
        # print("m_users ", m_users[:,:,0].shape, flush=True)

        nb_query_embedding = self.query_emb(m_querys[:,:,0])
        nb_user_embedding = self.user_emb(m_users[:,:,0])
        # print("node_embedding ", node_embedding.shape, flush=True)
        # print("nb_query_embedding ", nb_query_embedding.shape, flush=True)
        # print("nb_user_embedding ", nb_user_embedding.shape, flush=True)
        nb_query_aggregated_embedding = self.gat(node_embedding.unsqueeze(1), nb_query_embedding, nb_query_embedding).squeeze(1) # [512, 64]
        nb_user_aggregated_embedding = self.gat(node_embedding.unsqueeze(1), nb_user_embedding, nb_user_embedding).squeeze(1)    # [512, 64]
        # print("node_embedding", node_embedding.shape,flush=True)
        # print("nb_query_aggregated_embedding", nb_query_aggregated_embedding.shape,flush=True)
        # print("nb_user_aggregated_embedding", nb_user_aggregated_embedding.shape,flush=True)
        key_embedding = torch.concat([node_embedding.unsqueeze(1), \
                                    nb_query_aggregated_embedding.unsqueeze(1), \
                                    nb_user_aggregated_embedding.unsqueeze(1)], axis=1) # [512, 3, 64]
        if self.semantic_level_attn:
            outputs = self.semantic_attn_layer(node_embedding.unsqueeze(1), key_embedding, key_embedding).squeeze(1)

            # [512, 1, 64], [512, 3, 64]
        else:
            y = torch.concat([node_embedding, nb_query_aggregated_embedding, nb_user_aggregated_embedding], axis=1)
            outputs = self.outlayer(y)
        return outputs

        # p_t = self.user_emb(i_user_pad[:,:,0])
        # mask_i = torch.where(i_user_pad[:,:,0] > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))
        # i_user_er = self.rate_emb(i_user_pad[:,:,1])
        
        # f_jt = self.g_u(torch.cat([p_t, i_user_er], dim = 2).view(-1, 2 * self.emb_dim)).view(p_t.size())
         
        # # calculate attention scores in user aggregation
        # q_j = mask_i.unsqueeze(2).expand_as(f_jt) * self.item_emb(iids).unsqueeze(1).expand_as(f_jt)
        
        # miu = self.item_users_att(torch.cat([f_jt, q_j], dim = 2).view(-1, 2 * self.emb_dim)).view(mask_i.size())
        # miu = torch.exp(miu) * mask_i
        # miu = miu / (torch.sum(miu, 1).unsqueeze(1).expand_as(miu) + self.eps)
        
        # z_j = self.aggre_users(torch.sum(miu.unsqueeze(2).expand_as(f_jt) * f_jt, 1))

        # return agg_emb


class _QueryModel(nn.Module):
    '''Item modeling to learn item latent factors.
    '''
    def __init__(self, emb_dim, user_emb, item_id_emb, item_genre_emb, query_emb, semantic_level_attn=True):
        super(_QueryModel, self).__init__()
        self.emb_dim = emb_dim
        self.user_emb = user_emb
        self.item_id_emb = item_id_emb
        self.item_genre_emb = item_genre_emb
        self.query_emb = query_emb
        self.semantic_level_attn = semantic_level_attn

        self.g_u = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        
        self.item_users_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_users = _Aggregation(self.emb_dim, self.emb_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gat = _GraphAttentionLayer(input_dim=emb_dim)
        self.semantic_attn_layer = _AttentionLayer(input_dim=emb_dim)
        self.outlayer = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        # used for preventing zero div error when calculating softmax score
        self.eps = 1e-10

    def forward(self, qids, uids, q_movies):
        # user aggregation
        # print(qids.max(), flush=True)
        node_embedding = self.query_emb(qids)
        # ex_node_embedding = node_embedding
        ex_node_embedding = self.user_emb(uids)
        nb_movie_embedding = self.item_id_emb(q_movies[:,:,0])
        nb_aggregated_embedding = self.gat(node_embedding.unsqueeze(1), nb_movie_embedding, nb_movie_embedding).squeeze(1)
        # print("node_embedding", node_embedding.shape,flush=True)
        # print("nb_aggregated_embedding", nb_aggregated_embedding.shape,flush=True)
        query_embedding = torch.add(node_embedding, ex_node_embedding)
        key_embedding = torch.concat([node_embedding.unsqueeze(1), \
                                    nb_aggregated_embedding.unsqueeze(1)], axis=1) # [512, 2, 64]
        if self.semantic_level_attn:
            outputs = self.semantic_attn_layer(node_embedding.unsqueeze(1), key_embedding, key_embedding).squeeze(1)
        else:
            y = torch.concat([node_embedding, nb_aggregated_embedding], axis=1)
            outputs = self.outlayer(y)
        return outputs

class Zoomer(nn.Module):
    '''GraphRec model proposed in the paper Graph neural network for social recommendation 

    Args:
        number_users: the number of users in the dataset.
        number_items: the number of items in the dataset.
        num_rate_levels: the number of rate levels in the dataset.
        emb_dim: the dimension of user and item embedding (default = 64).

    '''
    def __init__(self, num_users, num_items, num_item_genres, num_querys, emb_dim = 64, \
                    use_feature_level_attn=True, semantic_level_attn=True, ):
        super(Zoomer, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_item_genres = num_item_genres
        self.num_querys = num_querys
        self.emb_dim = emb_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.user_emb = nn.Embedding(self.num_users, self.emb_dim, padding_idx = 0)
        self.item_id_emb = nn.Embedding(self.num_items, self.emb_dim, padding_idx = 0)
        self.item_genre_emb = nn.EmbeddingBag(self.num_item_genres, self.emb_dim, padding_idx = 0)
        self.query_emb = nn.Embedding(self.num_querys, self.emb_dim, padding_idx = 0)

        self.use_feature_level_attn = use_feature_level_attn
        self.semantic_level_attn = semantic_level_attn
        self.feature_level_attn = _AttentionLayer(input_dim=self.emb_dim)
        self.item_dense = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        self.user_model = _UserModel(self.emb_dim, self.user_emb, \
                            self.item_id_emb, self.item_genre_emb, self.query_emb, self.semantic_level_attn)
        self.item_model = _ItemModel(self.emb_dim, self.user_emb, \
                            self.item_id_emb, self.item_genre_emb, self.query_emb, self.semantic_level_attn)
        self.query_model = _QueryModel(self.emb_dim, self.user_emb, \
                            self.item_id_emb, self.item_genre_emb, self.query_emb, self.semantic_level_attn)
        self.semantic_attn_layer =  _AttentionLayer(input_dim=emb_dim)
        self.feature_input_layer = {}
        self.feature_input_layer["mid"] = self.item_id_emb
        self.feature_input_layer["mgenre"] = self.item_genre_emb
        self.feature_input_layer["qid"] = self.query_emb
        self.feature_input_layer["uid"] = self.user_emb
        self.feature_dense = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        self.sl = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        # self.DSSM = nn.Sequential(
        #     nn.Linear(2 * self.emb_dim, self.emb_dim, bias = True),
        #     nn.ReLU(),
        #     nn.Linear(self.emb_dim, self.emb_dim, bias = True),
        #     nn.ReLU(),
        #     nn.Linear(self.emb_dim, 1),
        # )
        self.DSSM = _MultiLayerPercep(3 * self.emb_dim, 1)
        self.sg = nn.Sigmoid()
        # self.rate_pred = nn.Sequential(
        #     nn.Linear(2 * self.emb_dim, self.emb_dim, bias = True),
        #     nn.ReLU(),
        #     nn.Linear(self.emb_dim, self.emb_dim, bias = True),
        #     nn.ReLU(),
        #     nn.Linear(self.emb_dim, 1),
        # )

    def graph_aggregate(self, node_type, nb, nb2_dict, cnt1, cnt2):
        # nb:[B*cnt1, E], nb2_dict:{"m":[B*cnt1*cnt2, E],"q":[B*cnt1*cnt2, E], "u":[B*cnt1*cnt2, E]}
        x_emb_dim = nb.size()[-1]
        x = nb.view(-1, cnt1, 1, x_emb_dim) 
        z = []
        for nb2_type in nb_config[node_type]:
            y = nb2_dict[nb2_type].view(-1, cnt1, cnt2, x_emb_dim)
            y = self.semantic_attn_layer(x, y, y)  # [B, cnt1, 1, E]
            y = y.view(-1, x_emb_dim)   #  [B*cnt1, E]
            z.append(y)
        if len(z)>1:
            z = self.sl(torch.concat(z, axis=1)) # [B*cnt1, E]
        else:
            z = z[0]
        return z
        


    def forward(self, uids, mids, qids, nb, nb_2):
        '''
        Args:
            uids: the user id sequences.
            iids: the item id sequences.
            u_item_pad: the padded user-item graph.
            u_user_pad: the padded user-user graph.
            u_user_item_pad: the padded user-user-item graph.
            i_user_pad: the padded item-user graph.

        Shapes:
            uids: (B).
            iids: (B).
            u_item_pad: (B, ItemSeqMaxLen, 2).
            u_user_pad: (B, UserSeqMaxLen).
            u_user_item_pad: (B, UserSeqMaxLen, ItemSeqMaxLen, 2).
            i_user_pad: (B, UserSeqMaxLen, 2).

        Returns:
            the predicted rate scores of the user to the item.
        '''
        # nb 是一个dict{"m":{"m":{"mid":[B*cnt1], "mgenre":[B*cnt1]}, "u":{"uid":[B*cnt1]}}}
        # nb_2是一个dict, {"m":{"m":{"m":{"mid":[B*cnt1*cnt2], "mgenre":[B*cnt1*cnt2]}, "u":{"uid":[B*cnt1*cnt2]}}}}
        cnt1 = 5
        cnt2 = 5
        u_embedding = self.user_emb(uids)
        m_embedding = self.item_id_emb(mids)
        q_embedding = self.query_emb(qids)
        node_emb = {}
        node_emb["m"] = m_embedding
        node_emb["u"] = u_embedding
        node_emb["q"] = q_embedding

        nb_feature_dict = {}
        for node_type in ["m", "q", "u"]:
            nb_feature_dict[node_type] = {}
            for nb_type in nb_config[node_type]:
                nb_dict = nb[node_type][nb_type]  # {"m":{"mid":[B*cnt1*cnt2], "mgenre":[B*cnt1*cnt2]}, "u":{"uid":[B*cnt1*cnt2]}}
                tmp_feature = []
                for feature_name in nb_dict:
                    if feature_name=="mgenre":
                        offset = [i for i in range(nb_dict[feature_name].size()[0])]
                        offset = torch.tensor(offset, dtype=torch.int32).to(self.device)
                        tmp_feature.append(self.feature_input_layer[feature_name](nb_dict[feature_name], offset))
                    else:   
                        tmp_feature.append(self.feature_input_layer[feature_name](nb_dict[feature_name]))
                if len(tmp_feature)>1:
                    tmp_feature_proc = self.feature_dense(torch.concat(tmp_feature, 1))
                else:
                    tmp_feature_proc = tmp_feature[0]
                nb_feature_dict[node_type][nb_type] = tmp_feature_proc
                # nb_feature_dict是一个dict {"m":{"m":[B*cnt1, E], "u":[B*cnt1, E], "q":[B*cnt1, E]}}

        nb_2_feature_dict = {}
        for node_type in ["m", "q", "u"]:
            nb_2_feature_dict[node_type] = {}
            for nb_type in nb_config[node_type]:
                nb_dict = nb_2[node_type][nb_type]  # {"m":{"mid":[B*cnt1*cnt2], "mgenre":[B*cnt1*cnt2]}, "u":{"uid":[B*cnt1*cnt2]}}
                nb_2_feature_dict[node_type][nb_type] = {}
                for nb_2_type in nb_config[nb_type]:
                    nb_2_dict = nb_dict[nb_2_type]  # {"mid":[B*cnt1*cnt2], "mgenre":[B*cnt1*cnt2]}, "u":{"uid":[B*cnt1*cnt2]}
                    tmp_feature = []
                    for feature_name in nb_2_dict:
                        if feature_name=="mgenre":
                            offset = [i for i in range(nb_2_dict[feature_name].size()[0])]
                            offset = torch.tensor(offset, dtype=torch.int32).to(self.device)
                            tmp_feature.append(self.feature_input_layer[feature_name](nb_2_dict[feature_name], offset))
                        else:   
                            tmp_feature.append(self.feature_input_layer[feature_name](nb_2_dict[feature_name]))
                    if len(tmp_feature)>1:
                        tmp_feature_proc = self.feature_dense(torch.concat(tmp_feature, 1))
                    else:
                        tmp_feature_proc = tmp_feature[0]
                    nb_2_feature_dict[node_type][nb_type][nb_2_type] = tmp_feature_proc

        # nb_2_feature_dict是一个dict {"m":{"m":{"m":[B*cnt1*cnt2, E], "u":[B*cnt1*cnt2, E]}}}
        nb_agg_emb = {}
        for node_type in ["m", "q", "u"]:
            nb_agg_emb[node_type] = {}
            for nb_type in nb_config[node_type]:
                nb_agg_emb[node_type][nb_type] = self.graph_aggregate(nb_type, 
                                                            nb_feature_dict[node_type][nb_type], \
                                                            nb_2_feature_dict[node_type][nb_type], \
                                                            cnt1, cnt2)   
            # nb_agg_emb是一个dict, {"m":{"m":[B*cnt1, E], "u":[B*cnt1, E]}}
        node_agg_emb = {}
        for node_type in ["m", "q", "u"]:
            # print(nb_agg_emb[node_type], flush=True)
            node_agg_emb[node_type] = self.graph_aggregate(node_type, 
                                                        node_emb[node_type], \
                                                        nb_agg_emb[node_type], \
                                                        1, cnt1) # [B, E], {"m":[B*cnt1, E], "u":[B*cnt1, E]}
            # node_agg_emb是一个dict, {"m":[B, E], "u":[B, E], "q":[B, E]}

        movie_embedding = node_agg_emb["m"]
        user_embedding = node_agg_emb["u"]
        query_embedding = node_agg_emb["q"]
        qu_embedding = torch.cat([user_embedding, movie_embedding], 1)
        
        
        # # print(qids, q_movies)
        # # print("mids", mids.shape, "m_querys", m_querys.shape, "m_users", m_users.shape)
        # user_emb = self.user_model(uids, qids, u_movies)
        # # print(uids, mids, qids, m_querys, m_users, q_movies, u_movies)
        # mid_embedding = self.item_id_emb(mids)
        # movie_genre_id = movie_genre
        # movie_genre_offset = mg_offset
        # movie_genre_embedding = self.item_genre_emb(movie_genre_id, movie_genre_offset)
        # if self.use_feature_level_attn:
        #     context_embedding = user_emb   # [512, 64], [512, 64], [512, 64]
        #     feature_embedding = torch.concat([mid_embedding.unsqueeze(1), movie_genre_embedding.unsqueeze(1)], 1)
        #     item_embedding = self.feature_level_attn(context_embedding, feature_embedding, feature_embedding)
        # else:
        #     item_embedding = torch.concat([mid_embedding, movie_genre_embedding])
        #     item_embedding = self.item_dense(item_embedding)
        # item_emb = self.item_model(mids, m_querys, m_users)
        
        # # # query_emb = self.query_model(qids, q_movies)
        # # user_emb = item_emb
        # # query_emb = item_emb

        # query_emb = self.query_model(qids, uids, q_movies)
        # # user_emb = query_emb
        # # item_emb = query_emb

        # qu_emb = torch.cat([user_emb, item_emb], 1)
        # print("item_emb", item_emb.shape, flush=True)
        # print("query_emb", query_emb.shape, flush=True)
        # print("user_emb", user_emb.shape, flush=True)

        # make prediction
        r_ij = self.DSSM(torch.cat([qu_embedding, movie_embedding], 1))
        pred = self.sg(r_ij)

        return pred
