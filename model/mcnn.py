import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride, padding=1, bias=False)

def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        
        self.conv_bn3 = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            conv3x3(planes, planes),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv_bn3(x)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class MCNN(nn.Module):
    def __init__(self, drop_rate, indim):
        super(MCNN, self).__init__()
        self.drop_rate = drop_rate
        self.inplanes = indim

    def prelayer(self, indim):

        layer = nn.Sequential(
            nn.Conv1d(indim, 128, kernel_size=7, stride = 2, padding = 3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.inplanes = 128
        return layer
    
    def make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
                nn.ReLU(),
                nn.Dropout(self.drop_rate),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate = self.drop_rate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, drop_rate = self.drop_rate))

        return nn.Sequential(*layers)
