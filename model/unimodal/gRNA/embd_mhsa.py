import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.torch_util import PositionalEncoding, Flattening


class EMBD_MHSA(nn.Module):
    def __init__(self, drop_prob: float, len: int):
        super(EMBD_MHSA, self).__init__()

        self.seq_len = len
        self.multi_head_num = 8##
        self.single_head_size = 16##
        self.multi_head_size = 128##
        
        self.RNN_hidden = 24#

        self.embedding_dim = 128
        self.dropout_rate = drop_prob
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.embedding_layer = nn.Embedding(
            num_embeddings=4, embedding_dim=self.embedding_dim, max_norm=True
        )
        self.position_encoding = PositionalEncoding(
            dim=self.embedding_dim, max_len=self.seq_len, dropout=0.1
        )

        self.gru = nn.GRU(
            48, 24, num_layers=2, bidirectional=True
        )

        self.Q = nn.ModuleList([
                nn.Linear(
                    in_features=self.embedding_dim, out_features=self.single_head_size
                )
                for i in range(0, self.multi_head_num)
        ])
    
        self.K = nn.ModuleList([
                nn.Linear(
                    in_features=self.embedding_dim, out_features=self.single_head_size
                )
                for i in range(0, self.multi_head_num)
        ])

        self.V = nn.ModuleList([        
                nn.Linear(
                    in_features=self.embedding_dim, out_features=self.single_head_size
                )
                for i in range(0, self.multi_head_num)
        ])

        self.relu = nn.ModuleList([nn.ReLU() for i in range(0, self.multi_head_num)])
        self.MultiHeadLinear = nn.Sequential(
            nn.LayerNorm(self.single_head_size * self.multi_head_num),
            nn.Linear(
                in_features=self.single_head_size * self.multi_head_num,
                out_features=self.multi_head_size,
            ),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv1d(self.embedding_dim, 64, kernel_size = 3, padding = "same", stride = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(64, 32, kernel_size=3, padding="same", stride = 1),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(32, 16, kernel_size=3, padding="same", stride = 1),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.ReLU()
        )

        self.Conv5 = nn.Sequential(
            nn.Conv1d(self.embedding_dim, 64, kernel_size = 5, padding = "same", stride = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(64, 32, kernel_size=5, padding="same", stride = 1),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(32, 16, kernel_size=5, padding="same", stride = 1),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.ReLU()
        )

        self.Conv7 = nn.Sequential(
            nn.Conv1d(self.embedding_dim, 64, kernel_size = 7, padding = "same", stride = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(64, 32, kernel_size=7, padding="same", stride = 1),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(32, 16, kernel_size=7, padding="same", stride = 1),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.ReLU()
        )

        self.flattening = Flattening()

        self.fclayer = nn.Sequential(
            nn.Linear(in_features=336, out_features=32),
            #nn.Linear(in_features=112, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features=32, out_features=1),
        )

    def attention(self, query, key, value, mask=None, dropout=0.0):
        # based on: https://nlp.seas.harvard.edu/2018/04/03/attention.html
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.matmul(p_attn, value), p_attn
    
    def forward(self, inputs):

        if isinstance(inputs, dict):
            inputs = inputs['Xg'].to(self.device)

        #embedding layer
        embd = self.embedding_layer(inputs)# * math.sqrt(self.embedding_dim)
        #embd = self.position_encoding(embd) #512, 33, 128

        #Attention layer
        # pAttn_concat = torch.Tensor([]).to(self.device)
        # attn_concat = torch.Tensor([]).to(self.device)
        # for i in range(0, self.multi_head_num):
        #     query = self.Q[i](embd)
        #     key = self.K[i](embd)
        #     value = self.V[i](embd)
        #     attnOut, p_attn = self.attention(query, key, value, dropout=0.0)
        #     attnOut = self.relu[i](attnOut)
        #     attn_concat = torch.cat((attn_concat, attnOut), dim=2)

        # attn_out = self.MultiHeadLinear(attn_concat) #512, 33, 128

        #bi-gru layer
        # embd, _ = self.gru(embd)
        # F_RNN = embd[:, :, : self.RNN_hidden]
        # R_RNN = embd[:, :, self.RNN_hidden :]
        # embd = torch.cat((F_RNN, R_RNN), 2)
        #print(embd.shape) #[512, 33, 128]
        
        attn_out = embd.transpose(1, 2)
        attn_out5 = attn_out.clone()
        attn_out7 = attn_out.clone()

        attn_out = self.Conv3(attn_out) #[batch, 16, 7]
        attn_out5 = self.Conv5(attn_out5)
        attn_out7 = self.Conv7(attn_out7)
        mc = torch.cat([attn_out, attn_out5, attn_out7], dim=1)
        mc = mc.transpose(1, 2)

        mc, _ = self.gru(mc)
        F_RNN = mc[:, :, : self.RNN_hidden]
        R_RNN = mc[:, :, self.RNN_hidden :]
        mc = torch.cat((F_RNN, R_RNN), 2)
        #print(mc.shape) #[512, 33, 128]
        
  
        #mc = torch.cat([attn_out, attn_out5, attn_out7], dim=1)
        out = self.flattening(mc) #[batch, 48, 7]
        out = self.fclayer(out)
        return out.squeeze()