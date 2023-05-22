import torch
from utils import flux, models
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math
from typing import Type

class Bathymetry:
    @staticmethod
    def call(x):
        raise NotImplementedError()
    @staticmethod
    def diff(x):
        raise NotImplementedError()

class ExpSlope(Bathymetry):
    @staticmethod
    def call(x):
        return torch.exp(-(x-1)**2)
    @staticmethod
    def diff(x):
        return -2.*(x-1)*ExpSlope.call(x)


class Config:
    def __init__(self, init, bath_x, mesh: flux.Mesh, scheme: flux.Scheme):
        self.init = init
        self.bath_x = bath_x
        self.mesh = mesh
        self.scheme = scheme
        self.g = 9.81
    def source(self,t,x,Q):
        ret = torch.zeros([len(x), 2])
        ret[:,1] = -self.g*Q[:,0]*self.bath_x(x)
        return ret

def get_dataset(config: Config):
    '''V.shape = [N, M, 2]'''
    M, N = config.mesh.M, config.mesh.N
    dt = config.scheme.dt
    V = torch.zeros((N,M,2), requires_grad=False)
    V[0,:,:] = config.init(config.mesh.centroids)
    for n in range(N-1):
        V[n+1,:,:] = config.scheme.vectorstep(V[n,:,:]) + dt*config.source(n*dt, config.mesh.centroids, V[n,:,:])
    return V

def make_pointwise(dataset, config: Config):
    M, N = config.mesh.M, config.mesh.N
    tgrid = torch.linspace(0, config.mesh.T, N)
    xgrid = config.mesh.centroids
    coords = torch.cartesian_prod(tgrid[:-1], xgrid)
    qgrid = torch.zeros(M*(N-1), 2)
    # coords = torch.cartesian_prod(tgrid, xgrid)
    # Need west, centre, east for LW.step, and ground truth
    target = torch.zeros((M*(N-1), 4, 2))
    for n, t in enumerate(tgrid[:-1]):
        for i, x in enumerate(xgrid):
            coord_ind = i + M*n
            west = None if i==0 else dataset[n,i-1,:]   # Ensures periodic BCs (positive a)
            east = None if i==M-1 else dataset[n,i+1,:]
            if west is None:
                west = dataset[n,i,:]
                # west[1] = -west[1]
                west[1] = 0.
            if east is None:
                east = dataset[n,i,:]
                # east[1] = -east[1]
                east[1] = 0.
            qgrid[coord_ind,:] = dataset[n,i,:]
            # (w,c,e,target)
            target_row = [west, dataset[n,i,:], east, dataset[n+1,i,:]]
            target[coord_ind,:,:] = torch.vstack(target_row)
    # coords = torch.cat([coords, ugrid.reshape((-1,1))], dim=1)
    return coords, qgrid, target

def train(model: nn.Module, dataset, config: Config, epochs: int):
    # First iteration: No explicit differentiation
    M, N = config.mesh.M, config.mesh.N
    dt = config.scheme.dt
    coords, qgrid, target = make_pointwise(dataset, config)
    # coords.require_grad = True
    criterion = nn.MSELoss()
    def compute_loss(model, coord_batch, Q_batch, target_batch):
        west, centre, east, true = torch.unbind(target_batch, dim=1)
        nosource = config.scheme.step(west, centre, east)

        bath_diff_preds = model(coord_batch[:,1].reshape((-1,1))).squeeze()
        # sourcepreds = torch.zeros_like(Q_batch)
        sourcepreds = -config.g*Q_batch[:,0]*bath_diff_preds
        return criterion(nosource[:,1]+dt*sourcepreds, true[:,1])
    
    optim = torch.optim.Adam(model.parameters(), lr=0.005)
    BATCH_SIZE = 32
    EPOCHS = epochs
    log_freq = max(math.floor(epochs/4), 1) # TODO: 2
    dataset = TensorDataset(coords, qgrid, target)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    history = []
    running_loss = 0.
    loss_updates = 0
    for epoch in range(EPOCHS):
        batch = 0
        for (coord_batch, Q_batch, target_batch) in loader:
            if torch.any(Q_batch.isnan()):
                raise ValueError(f'ERROR: Encountered NaN in input at epoch {epoch}, batch {batch}')
            optim.zero_grad()
            loss = compute_loss(model, coord_batch, Q_batch, target_batch)
            
            # outputs = model(inputs)
            # loss = loss_fn(outputs, targets)
            loss.backward()
            optim.step()
            history.append(loss.item())
            running_loss += loss.item()
            loss_updates += 1
            batch += 1
        if epoch%log_freq==(log_freq-1):
            print(f'Epoch = {epoch+1}  avg loss = {running_loss/loss_updates:5.2e}')
            running_loss = 0.0
            loss_updates = 0
    return history
    
