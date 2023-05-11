import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    def __init__(self, inp_dim=2, periodic=False):
        super().__init__()
        self.periodic = periodic
        # Split x dimension into sin(x), cos(x)
        if periodic:
            inp_dim += 1
        self.fc1 = nn.Linear(inp_dim,30)
        self.fc2 = nn.Linear(30,30)
        self.fc3 = nn.Linear(30,30)
        self.out = nn.Linear(30,1)
    def forward(self, x):
        batch_dim = len(x.shape) == 2
        if not batch_dim:
            x = x.reshape((1,-1))
        if self.periodic:
            shape = list(x.shape)
            shape[-1] += 1
            tmp = torch.zeros(shape)
            tmp[:,[0, -1]] = x[:,[0,-1]]
            tmp[:,[1,2]] = torch.column_stack((torch.sin(x[:,1]), torch.cos(x[:,1])))
            x = tmp
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        out = self.out(x)
        return out if batch_dim else out.squeeze()

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

