import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.Parameter as Parameter
import torch.optim
from gat import GAT

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
class zoomer(nn.Module):

    def __init__(self, user_max_dict, movie_max_dict, convParams, all_user, all_movie, 
                    adj_mat, embed_dim=32, fc_size=200):
        '''

        Args:
            user_max_dict: the max value of each user attribute. {'uid': xx, 'gender': xx, 'age':xx, 'job':xx}
            user_embeds: size of embedding_layers.
            movie_max_dict: {'mid':xx, 'mtype':18, 'mword':15}
            fc_sizes: fully connect layer sizes. normally 2
        '''

        super(zoomer, self).__init__()

        # --------------------------------- user channel ----------------------------------------------------------------
        # user embeddings
        self.user_num = user_max_dict['uid']
        self.movie_num = movie_max_dict['mid']
        # k = torch.Tensor([user_max_dict['uid'], embed_dim])
        # print(k)
        # self.embedding_uid = nn.Embedding.from_pretrained(k, freeze=False)
        # k = torch.tensor(nn.Embedding(user_max_dict['uid'], embed_dim))
        k = torch.randn(user_max_dict['uid'], embed_dim)
        self.embedding_uid = nn.Embedding.from_pretrained(k, freeze=False)
        self.embedding_gender = nn.Embedding(user_max_dict['gender'], embed_dim // 2)
        self.embedding_age = nn.Embedding(user_max_dict['age'], embed_dim // 2)
        self.embedding_job = nn.Embedding(user_max_dict['job'], embed_dim // 2)

        self.all_user = all_user
        self.all_movie = all_movie
        self.adj_mat = adj_mat.to(device)

        # user embedding to fc: the first dense layer
        self.fc_uid = nn.Linear(embed_dim, embed_dim)
        self.fc_gender = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_age = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_job = nn.Linear(embed_dim // 2, embed_dim)

        # concat embeddings to fc: the second dense layer
        self.fc_user_combine = nn.Linear(4 * embed_dim, fc_size)

        # --------------------------------- movie channel -----------------------------------------------------------------
        # movie embeddings
        self.embedding_mid = nn.Embedding(movie_max_dict['mid'], embed_dim)  # normally 32
        self.embedding_mtype_sum = nn.EmbeddingBag(movie_max_dict['mtype'], embed_dim, mode='sum')

        self.fc_mid = nn.Linear(embed_dim, embed_dim)
        self.fc_mtype = nn.Linear(embed_dim, embed_dim)

        # movie embedding to fc
        self.fc_mid_mtype = nn.Linear(embed_dim * 2, fc_size)

        # text convolutional part
        # wordlist to embedding matrix B x L x D  L=15 15 words
        self.embedding_mwords = nn.Embedding(movie_max_dict['mword'], embed_dim)

        # input word vector matrix is B x 15 x 32
        # load text_CNN params
        kernel_sizes = convParams['kernel_sizes']
        # 8 kernel, stride=1,padding=0, kernel_sizes=[2x32, 3x32, 4x32, 5x32]
        # self.Convs_text = [nn.Sequential(
        #     nn.Conv2d(1, 8, kernel_size=(k, embed_dim)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(15 - k + 1, 1), stride=(1, 1))
        # ).to(device) for k in kernel_sizes]

        self.Convs_text = [nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(k, embed_dim)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(15 - k + 1, 1), stride=(1, 1))
        ).to(device) for k in kernel_sizes]

        # movie channel concat
        self.fc_movie_combine = nn.Linear(embed_dim * 2 + 8 * len(kernel_sizes), fc_size)  # tanh

        # BatchNorm layer
        self.BN_uid = nn.BatchNorm2d(1)
        self.BN_gender = nn.BatchNorm2d(1)
        self.BN_age = nn.BatchNorm2d(1)
        self.BN_job = nn.BatchNorm2d(1)
        
        self.BN_mid = nn.BatchNorm2d(1)
        self.BN_mtype = nn.BatchNorm2d(1)

        self.mapped_embedding_dim = 32
        self.output_embedding_dim = 16

        self.user_map = nn.Linear(5*embed_dim // 2, self.mapped_embedding_dim)
        self.movie_map = nn.Linear(2*embed_dim, self.mapped_embedding_dim)

        self.GCN_layer = nn.Linear(self.mapped_embedding_dim, self.output_embedding_dim)
        self.pred_layer = nn.Linear(self.output_embedding_dim*2, 1)

        self.gat = GAT(nfeat=self.mapped_embedding_dim, 
                nhid=8, 
                nclass=self.output_embedding_dim,
                dropout=0.6, 
                nheads=2,
                alpha=0.2).to(device)

        self.all_uid = []
        self.all_gender = []
        self.all_age = []
        self.all_job = []
        self.all_mid = []
        self.all_mtype = []
        self.all_mtite = []

        for idx in range(user_max_dict['uid']):
            line = all_user.iloc[idx]
            self.all_uid.append(line[0])
            self.all_gender.append(line[1])
            self.all_age.append(line[2])
            self.all_job.append(line[3])
        
        for idx in range(movie_max_dict['mid']):
            line = all_movie.iloc[idx]
            self.all_mid.append(line[0])
            self.all_mtype.append(line[2])
            self.all_mtite.append(line[1])          

        self.all_uid = torch.tensor(self.all_uid).to(device)
        self.all_gender = torch.tensor(self.all_gender).to(device)
        self.all_age = torch.tensor(self.all_age).to(device)
        self.all_job = torch.tensor(self.all_job).to(device)
        self.all_mid = torch.tensor(self.all_mid).to(device)
        self.all_mtype = torch.tensor(self.all_mtype).to(device)
        self.all_mtite = torch.tensor(self.all_mtite).to(device)

        
        
        # self.all_movie_embedding = self.all_mid_embedding
        self.sg = nn.Sigmoid()
        
    def GCN(self, embedding, adj_mat):
        adj_mat = adj_mat.to(device)
        outputs = torch.mm(adj_mat, embedding)
        outputs = self.GCN_layer(outputs)
        return outputs

    # def forward(self, user_input, movie_input):
    #     # pack train_data
    #     uid = user_input['uid']
    #     gender = user_input['gender']
    #     age = user_input['age']
    #     job = user_input['job']

    #     mid = movie_input['mid']
    #     mtype = movie_input['mtype']
    #     mtext = movie_input['mtext']
    #     if torch.cuda.is_available():
    #         uid, gender, age, job,mid,mtype,mtext = \
    #         uid.to(device), gender.to(device), age.to(device), job.to(device), mid.to(device), mtype.to(device), mtext.to(device)
    #     # user channel
    #     feature_uid = self.BN_uid(F.relu(self.fc_uid(self.embedding_uid(uid))))
    #     feature_gender = self.BN_gender(F.relu(self.fc_gender(self.embedding_gender(gender))))
    #     feature_age =  self.BN_age(F.relu(self.fc_age(self.embedding_age(age))))
    #     feature_job = self.BN_job(F.relu(self.fc_job(self.embedding_job(job))))

    #     # feature_user B x 1 x 200
    #     feature_user = F.tanh(self.fc_user_combine(
    #         torch.cat([feature_uid, feature_gender, feature_age, feature_job], 3)
    #     )).view(-1,1,200)

    #     # movie channel
        feature_mid = self.BN_mid(F.relu(self.fc_mid(self.embedding_mid(mid))))
        feature_mtype = self.BN_mtype(F.relu(self.fc_mtype(self.embedding_mtype_sum(mtype)).view(-1,1,1,32)))

    #     # feature_mid_mtype = torch.cat([feature_mid, feature_mtype], 2)

    #     # text cnn part
    #     feature_img = self.embedding_mwords(mtext)  # to matrix B x 15 x 32
    #     flattern_tensors = []
    #     for conv in self.Convs_text:
    #         flattern_tensors.append(conv(feature_img.view(-1,1,15,32)).view(-1,1, 8))  # each tensor: B x 8 x1 x 1 to B x 8

    #     feature_flattern_dropout = F.dropout(torch.cat(flattern_tensors,2), p=0.5)  # to B x 32

    #     # feature_movie B x 1 x 200
    #     feature_movie = F.tanh(self.fc_movie_combine(
    #         torch.cat([feature_mid.view(-1,1,32), feature_mtype.view(-1,1,32), feature_flattern_dropout], 2)
    #     ))

    #     output = torch.sum(feature_user * feature_movie, 2)  # B x rank
    #     return output, feature_user, feature_movie



    # def build_feature_level_attn(self):
    #     for ntype in ['user', ,'query', 'item']:
    #     pass

    def get_embedding_from_graph(self, gcn_embedding, idx, mode='user'):
        if mode=='user':
            return gcn_embedding[idx]
        elif mode=='movie':
            return gcn_embedding[idx+self.user_num]

    def forward(self, user_input, movie_input):
        uid = user_input['uid']
        gender = user_input['gender']
        age = user_input['age']
        job = user_input['job']

        mid = movie_input['mid']
        mtype = movie_input['mtype']
        mtext = movie_input['mtext']

        # print(torch.cuda.is_available())
        # if torch.cuda.is_available():
        #     self.all_uid, self.all_gender, self.all_age, self.all_job, self.all_mid, self.all_mtype, self.all_mtite = \
        #         self.all_uid.to(device), self.all_gender.to(device), self.all_age.to(device), self.all_job.to(device), \
        #         self.all_mid.to(device), self.all_mtype.to(device), self.all_mtite.to(device)

        self.all_uid_embedding = self.embedding_uid(self.all_uid)
        self.all_gender_embedding = self.embedding_gender(self.all_gender)
        self.all_age_embedding =  self.embedding_age(self.all_age)
        self.all_job_embedding = self.embedding_job(self.all_job)
        self.all_user_embedding = torch.cat([self.all_uid_embedding, self.all_gender_embedding, 
                                                self.all_age_embedding, self.all_job_embedding], 1)
        
        self.all_mid_embedding = self.embedding_mid(self.all_mid)
        self.all_mtype_embedding = self.embedding_mtype_sum(self.all_mtype)

        feature_img = self.embedding_mwords(self.all_mtite)
        flattern_tensors = []
        for conv in self.Convs_text:
            flattern_tensors.append(conv(feature_img.view(-1,1,15,32)).view(-1,1, 8))  # each tensor: B x 8 x1 x 1 to B x 8

        # self.all_mtixtle = F.dropout(torch.cat(flattern_tensors,2), p=0.5).view(-1, 32)  # to B x 32
        self.all_movie_embedding = torch.cat([self.all_mid_embedding, self.all_mtype_embedding], 1)

        # print(self.all_user_embedding)
        user_mapped_embedding = self.user_map(self.all_user_embedding)
        movie_mapped_embedding = self.movie_map(self.all_movie_embedding)
        all_embedding = torch.cat([user_mapped_embedding, movie_mapped_embedding], 0)

        # if torch.cuda.is_available():
        #     uid, gender, age, job,mid,mtype,mtext = \
        #     uid.to(device), gender.to(device), age.to(device), job.to(device), mid.to(device), mtype.to(device), mtext.to(device)
       
        gcn_embedding = self.gat(all_embedding, self.adj_mat)
        user_embedding = self.get_embedding_from_graph(gcn_embedding, uid.view(-1), 'user')
        movie_embedding = self.get_embedding_from_graph(gcn_embedding, mid.view(-1), 'movie')

        um_tensor = torch.cat([user_embedding, movie_embedding], 1)
        # print(um_tensor)
        # print(um_tensor.shape)
        pred = self.pred_layer(um_tensor)
        pred = self.sg(pred)*5
        pred_rating = pred.ceil()
        return pred, pred_rating
        # uid = user_input['uid']
        # gender = user_input['gender']
        # age = user_input['age']
        # job = user_input['job']

        # mid = movie_input['mid']
        # mtype = movie_input['mtype']
        # mtext = movie_input['mtext']
        # if torch.cuda.is_available():
        #     uid, gender, age, job,mid,mtype,mtext = \
        #     uid.to(device), gender.to(device), age.to(device), job.to(device), mid.to(device), mtype.to(device), mtext.to(device)
        # # user channel
        # feature_uid = self.BN_uid(F.relu(self.fc_uid(self.embedding_uid(uid))))
        # feature_gender = self.BN_gender(F.relu(self.fc_gender(self.embedding_gender(gender))))
        # feature_age =  self.BN_age(F.relu(self.fc_age(self.embedding_age(age))))
        # feature_job = self.BN_job(F.relu(self.fc_job(self.embedding_job(job))))

        # # feature_user B x 1 x 200
        # feature_user = torch.tanh(self.fc_user_combine(
        #     torch.cat([feature_uid, feature_gender, feature_age, feature_job], 3)
        # )).view(-1,1,200)

        # # movie channel
        # feature_mid = self.BN_mid(F.relu(self.fc_mid(self.embedding_mid(mid))))
        # feature_mtype = self.BN_mtype(F.relu(self.fc_mtype(self.embedding_mtype_sum(mtype)).view(-1,1,1,32)))

        # # feature_mid_mtype = torch.cat([feature_mid, feature_mtype], 2)

        # # text cnn part
        # feature_img = self.embedding_mwords(mtext)  # to matrix B x 15 x 32
        # flattern_tensors = []
        # for conv in self.Convs_text:
        #     flattern_tensors.append(conv(feature_img.view(-1,1,15,32)).view(-1,1, 8))  # each tensor: B x 8 x1 x 1 to B x 8

        # feature_flattern_dropout = F.dropout(torch.cat(flattern_tensors,2), p=0.5)  # to B x 32

        # # feature_movie B x 1 x 200
        # feature_movie = torch.tanh(self.fc_movie_combine(
        #     torch.cat([feature_mid.view(-1,1,32), feature_mtype.view(-1,1,32), feature_flattern_dropout], 2)
        # ))

        # output = torch.sum(feature_user * feature_movie, 2)  # B x rank
        # output = self.sg(output)*5
        # pred_rating = output.ceil()
        # return output, pred_rating