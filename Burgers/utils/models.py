import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,30)
        self.fc2 = nn.Linear(30,30)
        self.fc3 = nn.Linear(30,30)
        self.out = nn.Linear(30,1)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.out(x)

class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models
    def forward(self, x):
        shape = [x.shape[0],1] if len(x.shape)==2 else [1]
        preds = torch.zeros(shape)
        for model in self.models:
            preds += model(x)
        return preds / len(self.models)
