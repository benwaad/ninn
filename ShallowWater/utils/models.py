import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    def __init__(self, inp_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(inp_dim,30)
        self.fc2 = nn.Linear(30,30)
        self.fc3 = nn.Linear(30,30)
        self.out = nn.Linear(30,1)
        self.activation = torch.tanh
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
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

