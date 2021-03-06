import torch
from torch import nn
import torch.nn.functional as F
import math

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_users = num_users
        self.num_items = num_items
        self.num_item_genres = num_item_genres
        self.num_querys = num_querys
        self.emb_dim = emb_dim
        self.user_emb = nn.Embedding(self.num_users, self.emb_dim, padding_idx = 0)
        self.item_id_emb = nn.Embedding(self.num_items, self.emb_dim, padding_idx = 0)
        self.item_genre_emb = nn.EmbeddingBag(self.num_item_genres, self.emb_dim, padding_idx = 0)
        self.query_emb = nn.Embedding(self.num_querys, self.emb_dim, padding_idx = 0)

        self.use_roi = True 
        self.use_feature_level_attn = use_feature_level_attn
        self.semantic_level_attn = semantic_level_attn
        self.feature_level_attn = _AttentionLayer(input_dim=self.emb_dim)
        self.item_dense = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        # self.user_model = _UserModel(self.emb_dim, self.user_emb, \
        #                     self.item_id_emb, self.item_genre_emb, self.query_emb, self.semantic_level_attn)
        # self.item_model = _ItemModel(self.emb_dim, self.user_emb, \
        #                     self.item_id_emb, self.item_genre_emb, self.query_emb, self.semantic_level_attn)
        # self.query_model = _QueryModel(self.emb_dim, self.user_emb, \
        #                     self.item_id_emb, self.item_genre_emb, self.query_emb, self.semantic_level_attn)
        
        self.feature_map_layer = {}
        self.feature_map_layer['u'] = _MultiLayerPercep(self.emb_dim, self.emb_dim).to(self.device)
        self.feature_map_layer['q'] = _MultiLayerPercep(self.emb_dim, self.emb_dim).to(self.device)
        self.feature_map_layer['i'] = _MultiLayerPercep(2*self.emb_dim, self.emb_dim).to(self.device)
        # self.DSSM = nn.Sequential(
        #     nn.Linear(2 * self.emb_dim, self.emb_dim, bias = True),
        #     nn.ReLU(),
        #     nn.Linear(self.emb_dim, self.emb_dim, bias = True),
        #     nn.ReLU(),
        #     nn.Linear(self.emb_dim, 1),
        # )
        self.gat = _GraphAttentionLayer(input_dim=emb_dim)
        self.semantic_attn_layer = _AttentionLayer(input_dim=self.emb_dim)
        self.outlayer = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        self.qu_layer = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        self.DSSM = _MultiLayerPercep(3 * self.emb_dim, 1)
        self.sg = nn.Sigmoid()
        
        # self.rate_pred = nn.Sequential(
        #     nn.Linear(2 * self.emb_dim, self.emb_dim, bias = True),
        #     nn.ReLU(),
        #     nn.Linear(self.emb_dim, self.emb_dim, bias = True),
        #     nn.ReLU(),
        #     nn.Linear(self.emb_dim, 1),
        # )

    def user_graph_aggregate(self, user_feature, query_feature, u_movie_feature): # [B, E], [B, E], [B, cnt1, E]
        node_embedding = user_feature
        node_roi_embedding = query_feature
        nb_item_embedding = torch.concat([u_movie_feature, user_feature.unsqueeze(1)], 1)   # ???B, cnt1+1, E???
        if self.use_roi:
            query_embedding = torch.add(node_embedding, node_roi_embedding)
        else:
            query_embedding = node_embedding
        nb_aggregated_embedding = self.gat(query_embedding.unsqueeze(1), nb_item_embedding, nb_item_embedding).squeeze(1)
        # print("nb_aggregated_embedding", nb_aggregated_embedding.shape,flush=True)
        
        y = torch.concat([node_embedding, nb_aggregated_embedding], axis=1)
        outputs = self.outlayer(y)
        # print("outputs", outputs.shape,flush=True)
        return outputs

    def query_graph_aggregate(self, query_feature, user_feature, q_movie_feature): # [B, E], [B, E], [B, cnt1, E]
        node_embedding = query_feature
        node_roi_embedding = user_feature
        nb_item_embedding = torch.concat([q_movie_feature, query_feature.unsqueeze(1)], 1)
        if self.use_roi:
            query_embedding = torch.add(node_embedding, node_roi_embedding)
        else:
            query_embedding = node_embedding
        nb_aggregated_embedding = self.gat(query_embedding.unsqueeze(1), nb_item_embedding, nb_item_embedding).squeeze(1)
        # print("nb_aggregated_embedding", nb_aggregated_embedding.shape,flush=True)
        
        y = torch.concat([node_embedding, nb_aggregated_embedding], axis=1)
        outputs = self.outlayer(y)
        # print("outputs", outputs.shape,flush=True)
        return outputs

    def item_graph_aggregate(self, item_feature, user_feature, m_user_feature, m_query_feature): # [B, E], [B, E], [B, 30, E], [B, 5, E]
        node_embedding = item_feature
        node_roi_embedding = user_feature
        nb_embedding = {}
        nb_aggregated_embedding = {}
        # print("m_user_feature", m_user_feature.size(), flush=True)
        # print("item_feature", item_feature.size(), flush=True)
        # print("node_embedding", node_embedding.size(), flush=True)
        # print("node_roi_embedding", node_roi_embedding.size(), flush=True)
        nb_embedding['u'] = torch.concat([m_user_feature, item_feature.unsqueeze(1)], 1)
        nb_embedding['q'] = torch.concat([m_query_feature, item_feature.unsqueeze(1)], 1)
        if self.use_roi:
            query_embedding = torch.add(node_embedding, node_roi_embedding).unsqueeze(1)
        else:
            query_embedding = node_embedding.unsqueeze(1)
        # print("query_embedding", query_embedding.size(), flush=True)
        # print("nb_embedding[u]", nb_embedding['u'].size(), flush=True)
        # print("nb_embedding[q]", nb_embedding['q'].size(), flush=True)
        nb_aggregated_embedding['u'] = self.gat(query_embedding, nb_embedding['u'], nb_embedding['u'])
        nb_aggregated_embedding['q'] = self.gat(query_embedding, nb_embedding['q'], nb_embedding['q'])
        # print("nb_aggregated_embedding[u]", nb_aggregated_embedding['u'].size(), flush=True)
        # print("nb_aggregated_embedding[q]", nb_aggregated_embedding['q'].size(), flush=True)
        # [B, 1, E]
        # print("nb_aggregated_embedding", nb_aggregated_embedding.shape,flush=True)
        
        if self.semantic_level_attn:
            KV_embedding = torch.concat([nb_aggregated_embedding['u'], nb_aggregated_embedding['q']], 1)
            outputs = self.semantic_attn_layer(query_embedding, KV_embedding, KV_embedding).squeeze(1)
        else:
            y = torch.concat([node_embedding, \
                    nb_aggregated_embedding['u'].squeeze(1), \
                    nb_aggregated_embedding['q'].squeeze(1)], \
                    axis=1)
            outputs = self.outlayer(y)
        # print("outputs", outputs.shape,flush=True)
        return outputs

    def forward(self, uids, mids, qids, movie_genre, mg_offset,
                m_qids, m_uids, u_mids, q_mids,
                u_mgenre, u_mg_offset, q_mgenre, q_mg_offset
                ):
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
        # print(qids, q_movies)
        # print("mids", mids.shape, "m_querys", m_querys.shape, "m_users", m_users.shape)
        # node features
        # print("uids", uids.size(), flush=True)
        # print("mids", mids.size(), flush=True)
        # print("qids", qids.size(), flush=True)
        # print("movie_genre", movie_genre.size(), flush=True)
        # print("mg_offset", mg_offset.size(), flush=True)
        # print("m_qids", m_qids.size(), flush=True)
        # print("m_uids", m_uids.size(), flush=True)
        # print("u_mids", u_mids.size(), flush=True)
        # print("q_mids", q_mids.size(), flush=True)
        # print("u_mgenre", u_mgenre.size(), flush=True)
        # print("u_mg_offset", u_mg_offset.size(), flush=True)
        # print("q_mgenre", q_mgenre.size(), flush=True)
        # print("q_mg_offset", q_mg_offset.size(), flush=True)
        batch_size = uids.size()[0]
        mqcnt = m_qids.size()[1]
        mucnt = m_uids.size()[1]
        umcnt = u_mids.size()[1]
        qmcnt = q_mids.size()[1]
        m_qids = m_qids[:,:,0]
        m_uids = m_uids[:,:,0]
        u_mids = u_mids[:,:,0]
        q_mids = q_mids[:,:,0]
        movie_genre_feature = self.item_genre_emb(movie_genre, mg_offset)   # [B, E]
        user_feature = self.feature_map_layer['u'](self.user_emb(uids))     # [B, E]
        query_feature = self.feature_map_layer['q'](self.query_emb(qids))   # [B, E]
        item_id_features = self.item_id_emb(mids)
        # nb features
        m_query_feature = self.feature_map_layer['q'](self.query_emb(m_qids))   # [B, cnt1, E]
        m_user_feature = self.feature_map_layer['u'](self.user_emb(m_uids))     # [B, cnt1, E]
        # print("m_query_feature", m_query_feature.size(), flush=True)
        # print("m_user_feature", m_user_feature.size(), flush=True)
        
        
        # q_item_features = torch.concat([self.item_id_emb(q_mids), q_movies_genre_feature], 1)   # [B, cnt1, 2*E]
        q_item_id_features = self.item_id_emb(q_mids)
        q_movie_genre_feature = self.item_genre_emb(q_mgenre, q_mg_offset).view(-1, qmcnt, self.emb_dim)     # [B, cnt1, E]
        # print("q_item_id_features", q_item_id_features.size(), flush=True)
        # print("q_movie_genre_feature", q_movie_genre_feature.size(), flush=True)
        if self.use_feature_level_attn:
            context_embedding = query_feature
            item_features = torch.concat([q_item_id_features.unsqueeze(2), q_movie_genre_feature.unsqueeze(2)], 2)
            q_item_features = self.feature_level_attn(context_embedding.unsqueeze(1).unsqueeze(1), item_features, item_features).squeeze(1).squeeze(2)
        else:
            q_item_features = self.feature_map_layer['i'](torch.concat([q_item_id_features, q_movie_genre_feature], 2)).squeeze(1)

        # q_movie_genre_feature = self.item_genre_emb(q_mgenre, q_mg_offset)
        
        
        # print("u_mids", u_mids.size(), flush=True)
        # print("u_item_id_features", u_item_id_features.size(), flush=True)
        
        u_item_id_features = self.item_id_emb(u_mids)
        u_movie_genre_feature = self.item_genre_emb(u_mgenre, u_mg_offset).view(-1, umcnt, self.emb_dim)
        if self.use_feature_level_attn:
            context_embedding = user_feature
            item_features = torch.concat([u_item_id_features.unsqueeze(2), u_movie_genre_feature.unsqueeze(2)], 2)
            u_item_features = self.feature_level_attn(context_embedding.unsqueeze(1).unsqueeze(1), item_features, item_features).squeeze(1).squeeze(2)
        else:
            u_item_features = self.feature_map_layer['i'](torch.concat([u_item_id_features, u_movie_genre_feature], 2)).squeeze(1)
        # print("u_movie_genre_feature", u_movie_genre_feature.size(), flush=True)
        # print("u_item_id_features", u_item_id_features.size(), flush=True)
        # q_item_features = torch.zeros([batch_size, qmcnt, 2*self.emb_dim], dtype=torch.float32).to(self.device)
        

        # u_movies_genre_feature = self.item_genre_emb(u_mgenre, u_mg_offset)     # [B, cnt1, E]
        # u_item_features = torch.concat([self.item_id_emb(u_mids), u_movies_genre_feature], 1)   # [B, cnt1, 2*E]
        # u_item_features = torch.zeros([batch_size, umcnt, 2*self.emb_dim], dtype=torch.float32).to(self.device)

        # q_item_features = self.feature_map_layer['i'](q_item_features)  # [B, cnt1, E]
        # u_item_features = self.feature_map_layer['i'](u_item_features)  # [B, cnt1, E]
        
        if self.use_feature_level_attn:
            context_embedding = user_feature
            item_features = torch.concat([item_id_features.unsqueeze(1), movie_genre_feature.unsqueeze(1)], 1)
            # print("context_embedding", context_embedding.size(), flush=True)
            # print("item_features", item_features.size(), flush=True)
            item_feature = self.feature_level_attn(context_embedding.unsqueeze(1), item_features, item_features).squeeze(1)
        else:
            item_feature = self.feature_map_layer['i'](torch.concat([item_id_features, movie_genre_feature], 1)).squeeze(1)

        # print("item_feature", item_feature.size(), flush=True)
        # print("user_feature", user_feature.size(), flush=True)
        # print("m_user_feature", m_user_feature.size(), flush=True)
        # print("q_item_features", q_item_features.size(), flush=True)

        user_embedding = self.user_graph_aggregate(user_feature, query_feature, u_item_features)
        query_embedding = self.query_graph_aggregate(query_feature, user_feature, q_item_features)
        item_embedding = self.item_graph_aggregate(item_feature, user_feature, m_user_feature, m_query_feature)
        
        qu_embedding = torch.concat([user_embedding, query_embedding], 1)

        # print("qu_embedding", qu_embedding.size(), flush=True)
        # print("item_embedding", item_embedding.size(), flush=True)
        r_ij = self.DSSM(torch.cat([qu_embedding, item_embedding], 1))
        pred = self.sg(r_ij)


        return pred

