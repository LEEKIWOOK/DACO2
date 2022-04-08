#Spearman correlation : 0.84

import math
import torch
import torch.nn as nn
from utils.torch_util import PositionalEncoding, Flattening

class EMBD_MK_CNN(nn.Module):
    def __init__(self, dropprob, seqlen, device):
        super(EMBD_MK_CNN, self).__init__()

        self.seqlen = seqlen
        self.indim = 288 if self.seqlen == 30 else 240

        self.embedding_dim = 128
        self.dropout_rate = dropprob
        self.device = device

        self.embedding_layer = nn.Embedding(
            num_embeddings=4, embedding_dim=self.embedding_dim, max_norm=True
        )
        # self.position_encoding = PositionalEncoding(
        #     dim=self.embedding_dim, max_len=self.seqlen, dropout=0.1
        # )

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
            nn.Linear(in_features=self.indim, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features=32, out_features=1),
        )

    def forward(self, inputs):

        if isinstance(inputs, dict):
            inputs = inputs['X'].to(self.device)

        embd = self.embedding_layer(inputs) #* math.sqrt(self.embedding_dim) #[batch, len, embd_dim]
        #embd = self.position_encoding(embd)

        embd = embd.transpose(1, 2) #[batch, 128, 33]
        embd5 = embd.clone()
        embd7 = embd.clone()

        embd = self.Conv3(embd) #[batch, 16, 7]
        embd5 = self.Conv5(embd5)
        embd7 = self.Conv7(embd7)

        mc = torch.cat([embd, embd5, embd7], dim=1)
        out = self.flattening(mc) #[batch, 48, 7]
        out = self.fclayer(out)
        return out.squeeze()