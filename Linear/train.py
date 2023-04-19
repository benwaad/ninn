import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from utils import flux, models
from utils.tools import apply_along_axis
import math

class Config:
    def __init__(self, init, source, mesh: flux.Mesh, scheme: flux.Scheme):
        self.init = init
        self.source = source
        self.mesh = mesh
        self.scheme = scheme

def get_dataset(config: Config):
    M, N = config.mesh.M, config.mesh.N
    dt = config.scheme.dt
    V = torch.zeros((M,N), requires_grad=False)
    V[:,0] = config.init(config.mesh.centroids)
    for n in range(N-1):
        V[:,n+1] = config.scheme.vectorstep(V[:,n]) + dt*config.source(n*dt,config.mesh.centroids)
    return V
def make_pointwise_dataset(dataset, config: Config):
    M, N = config.mesh.M, config.mesh.N
    tgrid = torch.linspace(0, config.mesh.T, N)
    xgrid = config.mesh.centroids
    coords = torch.cartesian_prod(tgrid, xgrid)
    # Need west, centre, east for LW.step, and ground truth
    target = torch.zeros((coords.shape[0], 4))
    for n, t in enumerate(tgrid[:-1]):
        for i, x in enumerate(xgrid):
            coord_ind = i + M*n
            west = M-1 if i==0 else i-1   # Ensures periodic BCs (positive a)
            east = 0 if i==M-1 else i+1
            # (w,c,e,target)
            target_row = [dataset[west,n], dataset[i,n], dataset[east,n], dataset[i,n+1]]
            target[coord_ind,:] = torch.tensor(target_row)
    return coords, target


def train(model: nn.Module, dataset, config: Config):
    M, N = config.mesh.M, config.mesh.N
    dt = config.scheme.dt
    coords, target = make_pointwise_dataset(dataset, config)

    criterion = nn.MSELoss()
    def loss_fn(preds, target):
        west, centre, east, true = torch.unbind(target, dim=1)
        nosource = config.scheme.step(west, centre, east)
        return criterion(nosource+dt*preds.squeeze(), true)
    optim = torch.optim.Adam(model.parameters(), lr=0.005)

    BATCH_SIZE = 32
    EPOCHS = 8
    log_freq = 2
    dataset = TensorDataset(coords, target)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    history = []
    running_loss = 0.
    loss_updates = 0
    for epoch in range(EPOCHS):
        for (inputs, targets) in loader:
            optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optim.step()
            history.append(loss.item())
            running_loss += loss.item()
            loss_updates += 1
        if epoch%log_freq==(log_freq-1):
            print(f'Epoch = {epoch+1}  avg loss = {running_loss/loss_updates:5.2e}')
            running_loss = 0.0
            loss_updates = 0
    return history


