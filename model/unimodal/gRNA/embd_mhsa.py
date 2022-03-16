import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.torch_util import PositionalEncoding, Flattening


class ATTN_CNN(nn.Module):
    def __init__(self, drop_prob: float, len: int):
        super(ATTN_CNN, self).__init__()

        self.seq_len = len
        self.embedding_dim = 128
        self.dropout_rate = drop_prob

        self.embedding_layer = nn.Embedding(
            num_embeddings=4, embedding_dim=self.embedding_dim, max_norm=True
        )
        self.position_encoding = PositionalEncoding(
            dim=self.embedding_dim, max_len=self.seq_len, dropout=0.1
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

        self.flattening = Flattening()

        self.predictor = nn.Sequential(
            nn.Linear(in_features=336, out_features=32),
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

    def mh_attention(self, inputs, in_features):
        single_head_size = 16
        multi_head_num = 4
        multi_head_size = 64  ###

        Q = nn.ModuleList(
            [
                nn.Linear(
                    in_features=in_features,
                    out_features=single_head_size,
                )
                for i in range(0, multi_head_num)
            ]
        ).to(inputs.device)

        K = nn.ModuleList(
            [
                nn.Linear(
                    in_features=in_features,
                    out_features=single_head_size,
                )
                for i in range(0, multi_head_num)
            ]
        ).to(inputs.device)

        V = nn.ModuleList(
            [
                nn.Linear(
                    in_features=in_features,
                    out_features=single_head_size,
                )
                for i in range(0, multi_head_num)
            ]
        ).to(inputs.device)

        MultiHeadLinear = nn.Sequential(
            nn.LayerNorm(single_head_size * multi_head_num),
            nn.Linear(
                in_features=single_head_size * multi_head_num,
                out_features=multi_head_size,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        ).to(inputs.device)

        pAttn_concat = torch.Tensor([]).to(inputs.device)
        attn_concat = torch.Tensor([]).to(inputs.device)
        relu_list = nn.ModuleList([nn.ReLU() for i in range(0, multi_head_num)])

        for i in range(0, multi_head_num):
            query = Q[i](inputs)
            key = K[i](inputs)
            value = V[i](inputs)
            attnOut, p_attn = self.attention(
                query, key, value, dropout=self.dropout_rate
            )
            attnOut = relu_list[i](attnOut)
            attn_concat = torch.cat((attn_concat, attnOut), dim=2)
        attn_out = MultiHeadLinear(attn_concat)

        return attn_out

    def forward(self, inputs):

        embd = self.embedding_layer(inputs) * math.sqrt(self.embedding_dim) #[batch, len, embd_dim]
        embd = self.position_encoding(embd)

        # embd = embd.transpose(1, 2) #[batch, 128, 33]
        # embd5 = embd.clone()
        # embd7 = embd.clone()

        # embd = self.Conv3(embd) #[batch, 16, 7]
        # embd5 = self.Conv5(embd5)
        # embd7 = self.Conv7(embd7)

        # mc = torch.cat([embd, embd5, embd7], dim=1)
        out = self.flattening(mc) #[batch, 48, 7]
        return out.squeeze()