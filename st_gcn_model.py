import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A, bias=True):
        super(GraphConv, self).__init__()
        self.A = torch.tensor(A, dtype=torch.float32, requires_grad=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        # x: (N, C, T, V)
        if self.A.device != x.device:
            A = self.A.to(x.device)
        else:
            A = self.A
        x = torch.einsum('nctv,vw->nctw', x, A)  # graph conv
        x = self.conv(x)
        return x

class STGCN(nn.Module):
    def __init__(self, in_channels, num_class, A, num_layers=3):
        super(STGCN, self).__init__()
        self.layers = nn.ModuleList()
        channels = [64, 128, 256]
        last_c = in_channels
        for c in channels[:num_layers]:
            self.layers.append(GraphConv(last_c, c, A))
            last_c = c
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channels[num_layers-1], num_class)
    
    def forward(self, x):
        # x: (N, C, T, V, M=1)
        x = x.squeeze(-1)  # remove M dim
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.pool(x)  # (N, C, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
