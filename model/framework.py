import math
import torch
import torch.nn as nn
from model.mcnn import BasicBlock, MCNN
from utils.torch_util import Flattening, PositionalEncoding

class Framework(nn.Module):
    def __init__(self, param):
        super(Framework, self).__init__()
        self.indim = 128
        self.dropout = param['dropout']
        DNA = MCNN(drop_rate=self.dropout, indim=self.indim)

        self.dna_prelayer = DNA.prelayer(indim = 100)
        self.dna_layer1 = DNA.make_layer(BasicBlock, self.indim * 2, blocks = 1, stride = 2)
        self.dna_layer2 = DNA.make_layer(BasicBlock, self.indim * 4, blocks = 1, stride = 2)
        #self.dna_pooling = nn.AvgPool1d(kernel_size=3, stride=2)

        self.embedding_rna = nn.Embedding(
            num_embeddings=14, embedding_dim = self.indim, max_norm=True
        )
        self.position_rna = PositionalEncoding(
            dim = self.indim, max_len = 102, dropout=0.1
        )

        RNA = MCNN(drop_rate=self.dropout, indim=self.indim)
        self.rna_layer1 = RNA.make_layer(BasicBlock, self.indim * 4, blocks = 1, stride = 2)
        #self.rna_layer2 = RNA.make_layer(BasicBlock, self.indim * 4, blocks = 1, stride = 2)
        self.rna_pooling = nn.AvgPool1d(kernel_size=3, stride=2)

        self.rnam_prelayer = RNA.prelayer(indim = 102)
        self.rnam_layer1 = RNA.make_layer(BasicBlock, self.indim * 2, blocks = 1, stride = 2)
        self.rnam_layer2 = RNA.make_layer(BasicBlock, self.indim * 4, blocks = 1, stride = 2)
        
        self.flattening = Flattening()
        self.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, dnaseq, rnass, rnamat):
        
        x = self.dna_prelayer(dnaseq)
        x = self.dna_layer1(x)
        x = self.dna_layer2(x)
        #x = self.dna_pooling(x)
        
        # r1 = self.embedding_rna(rnass[:,:,0]) * math.sqrt(self.indim)
        # r1 = r1.transpose(1, 2)
        # r1 = self.rna_layer1(r1)
        
        # r2 = self.rnam_prelayer(rnamat)
        # r2 = self.rnam_layer1(r2)
        # r2 = self.rnam_layer2(r2)
        out = self.flattening(x)
        out = self.fc(out)
        return out.squeeze()
