import torch.nn as nn
import torch
import torch.nn.functional as F

class Relation_Net(nn.Module):
    def __init__(self, feature_size):
        super(Relation_Net, self).__init__()
        
        self.bn = nn.Sequential(
            nn.BatchNorm1d(200),
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(200 ** 2, 5),
        )
        
    def forward(self, x1, x2):
        x1, x2 = torch.relu(x1), torch.relu(x2)
        x = torch.matmul(x1.unsqueeze(2), x2.unsqueeze(1))
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
      
        return x
