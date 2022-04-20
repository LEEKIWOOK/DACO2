#Spearman correlation : 0.84
import math
import torch
import torch.nn as nn
from utils.torch_util import Flattening

def dim(k):
    return int(math.floor((k-3)/2)+1)

class EMBD_MK_CNN(nn.Module):
    def __init__(self, dropprob, cfg):
        super(EMBD_MK_CNN, self).__init__()

        self.embd_mod = cfg['embd']
        if self.embd_mod == 0:
            self.embedding_dim = 4
            self.fcdim = cfg['seqlen'] * 32
        elif self.embd_mod == 1:
            self.embedding_dim = 128
            self.fcdim = dim(dim(cfg['seqlen'])) * 16 * 3
        elif self.embd_mod == 2:
            self.embedding_dim = 100
            kmerlen = int(math.floor((cfg['seqlen'] - cfg['kmer'])/cfg['stride']) + 1)
            self.fcdim = dim(dim(kmerlen)) * 16 * 3
        
        self.dropout_rate = dropprob
        self.device = cfg['device']
        #self.mod = mod

        self.embedding_layer = nn.Embedding(
            num_embeddings=4, embedding_dim=self.embedding_dim, max_norm=True
        )

        self.Conv_one = nn.Sequential(
            nn.Conv1d(self.embedding_dim, 16, kernel_size=3, padding = "same", stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(16, 32, kernel_size=3, padding = "same", stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
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
            nn.Linear(in_features=self.fcdim, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features=32, out_features=1),
        )

    def forward(self, input):
        #inputs = input[0]

        if isinstance(input, dict):
            input = input['X'].to(self.device)

        if self.embd_mod == 0: #one_hot encoding
            out = self.Conv_one(input)
        else:
            if self.embd_mod == 1:
                input = self.embedding_layer(input)
                
            input = input.transpose(1, 2) #[batch, 128, 33]
            embd = input.clone()
            embd5 = input.clone()
            embd7 = input.clone()
            
            embd = self.Conv3(embd) #[batch, 16, 7]
            embd5 = self.Conv5(embd5)
            embd7 = self.Conv7(embd7)
            out = torch.cat([embd, embd5, embd7], dim=1)

        out = self.flattening(out) #[batch, 48, 7]
        out = self.fclayer(out)
        return out.squeeze()