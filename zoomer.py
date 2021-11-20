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



class Zoomer(nn.Module):
    '''Zoomer model proposed in the paper 
       Zoomer: Improving and Accelerating Recommendation on Web-Scale Graphs via Regions of Interests

    Args:
        num_users: the number of users in the dataset.
        num_items: the number of items in the dataset.
        num_item_genres: the number of genres of item in the dataset.
        num_querys: the number of queries in the dataset.
        emb_dim: the dimension of user and item embedding (default = 64).
        use_feature_level_attn: whether to use feature level attention (using ROI).
        use_semantic_level_attn: whether to use semantic level attention (using ROI).
    '''
    def __init__(self, num_users, num_items, num_item_genres, num_querys, emb_dim = 64, \
                    use_feature_level_attn=True, use_semantic_level_attn=True, ):
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
        self.use_semantic_level_attn = use_semantic_level_attn
        self.feature_level_attn = _AttentionLayer(input_dim=self.emb_dim)
        self.semantic_attn_layer = _AttentionLayer(input_dim=self.emb_dim)
        self.item_dense = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        
        self.feature_map_layer = {}
        self.feature_map_layer['u'] = _MultiLayerPercep(self.emb_dim, self.emb_dim).to(self.device)
        self.feature_map_layer['q'] = _MultiLayerPercep(self.emb_dim, self.emb_dim).to(self.device)
        self.feature_map_layer['i'] = _MultiLayerPercep(2*self.emb_dim, self.emb_dim).to(self.device)
        
        self.DSSM = nn.Sequential(
            nn.Linear(3 * self.emb_dim, 2 * self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Linear(2 * self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, 1),
        )
        self.gat = _GraphAttentionLayer(input_dim=emb_dim)
        
        self.outlayer = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        self.qu_layer = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        # self.DSSM = _MultiLayerPercep(3 * self.emb_dim, 1)
        self.sg = nn.Sigmoid()
        
    def user_graph_aggregate(self, user_feature, query_feature, u_movie_feature): # [B, E], [B, E], [B, cnt1, E]
        node_embedding = user_feature
        node_roi_embedding = query_feature
        nb_item_embedding = torch.concat([u_movie_feature, user_feature.unsqueeze(1)], 1)   # 【B, cnt1+1, E】
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
        
        if self.use_semantic_level_attn:
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

    def forward(self, uids, mids, qids, m_genre,
                m_qids, m_uids, u_mids, q_mids,
                u_mgenre, q_mgenre
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
            the predicted rate scores of the user to the item under query.
        '''
        # print(qids, q_movies)
        # print("mids", mids.shape, "m_querys", m_querys.shape, "m_users", m_users.shape)
        # node features
        # print("uids", uids.size(), flush=True)
        # print("mids", mids.size(), flush=True)
        # print("qids", qids.size(), flush=True)
        # print("m_genre", m_genre.size(), flush=True)
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
        
        m_genre_feature = self.item_genre_emb(m_genre[0], m_genre[1])   # [B, E]
        user_feature = self.feature_map_layer['u'](self.user_emb(uids))     # [B, E]
        query_feature = self.feature_map_layer['q'](self.query_emb(qids))   # [B, E]
        item_id_features = self.item_id_emb(mids)
        # nb features
        m_query_feature = self.feature_map_layer['q'](self.query_emb(m_qids))   # [B, cnt1, E]
        m_user_feature = self.feature_map_layer['u'](self.user_emb(m_uids))     # [B, cnt1, E]
        
        q_item_id_features = self.item_id_emb(q_mids)
        q_m_genre_feature = self.item_genre_emb(q_mgenre[0], q_mgenre[1]).view(-1, qmcnt, self.emb_dim)     # [B, cnt1, E]
        if self.use_feature_level_attn:
            context_embedding = query_feature
            item_features = torch.concat([q_item_id_features.unsqueeze(2), q_m_genre_feature.unsqueeze(2)], 2)
            q_item_features = self.feature_level_attn(context_embedding.unsqueeze(1).unsqueeze(1), item_features, item_features).squeeze(1).squeeze(2)
        else:
            q_item_features = self.feature_map_layer['i'](torch.concat([q_item_id_features, q_m_genre_feature], 2)).squeeze(1)
        
        u_item_id_features = self.item_id_emb(u_mids)
        u_m_genre_feature = self.item_genre_emb(u_mgenre[0], u_mgenre[1]).view(-1, umcnt, self.emb_dim)
        if self.use_feature_level_attn:
            context_embedding = user_feature
            item_features = torch.concat([u_item_id_features.unsqueeze(2), u_m_genre_feature.unsqueeze(2)], 2)
            u_item_features = self.feature_level_attn(context_embedding.unsqueeze(1).unsqueeze(1), item_features, item_features).squeeze(1).squeeze(2)
        else:
            u_item_features = self.feature_map_layer['i'](torch.concat([u_item_id_features, u_m_genre_feature], 2)).squeeze(1)

        if self.use_feature_level_attn:
            context_embedding = user_feature
            item_features = torch.concat([item_id_features.unsqueeze(1), m_genre_feature.unsqueeze(1)], 1)
            item_feature = self.feature_level_attn(context_embedding.unsqueeze(1), item_features, item_features).squeeze(1)
        else:
            item_feature = self.feature_map_layer['i'](torch.concat([item_id_features, m_genre_feature], 1)).squeeze(1)

        user_embedding = self.user_graph_aggregate(user_feature, query_feature, u_item_features)
        query_embedding = self.query_graph_aggregate(query_feature, user_feature, q_item_features)
        item_embedding = self.item_graph_aggregate(item_feature, user_feature, m_user_feature, m_query_feature)
        
        qu_embedding = torch.concat([user_embedding, query_embedding], 1)

        r_ij = self.DSSM(torch.cat([qu_embedding, item_embedding], 1))
        pred = self.sg(r_ij)


        return pred

