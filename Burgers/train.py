import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from utils import flux, models
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
        V[:,n+1] = config.scheme.vectorstep(V[:,n]) + dt*config.source(n*dt, config.mesh.centroids)
    return V
def make_pointwise(dataset, config: Config):
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
def get_wavefront_dataset(config: Config):
    V = get_dataset(config)
    threshold = .05
    frontfinder = lambda v: torch.argwhere(v<threshold)[0,0]
    V = torch.column_stack([frontfinder(v) for v in torch.stack(torch.unbind(V,dim=1))])



def train(model: nn.Module, dataset, config: Config, epochs=8):
    M, N = config.mesh.M, config.mesh.N
    dt = config.scheme.dt
    coords, target = make_pointwise(dataset, config)
    criterion = nn.MSELoss()
    def loss_fn(preds, target):
        west, centre, east, true = torch.unbind(target, dim=1)
        nosource = config.scheme.step(west, centre, east)
        return criterion(nosource+dt*preds.squeeze(), true)
    optim = torch.optim.Adam(model.parameters(), lr=0.005)
    BATCH_SIZE = 32
    EPOCHS = epochs # TODO: 200
    log_freq = math.floor(epochs/4) # TODO: 40
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

def get_trained_ensemble(config: Config, n_models=5, epochs=8):
    dataset = get_dataset(config)
    histories = []
    model_list = [models.DenseNet() for _ in range(n_models)]
    for i, model in enumerate(model_list,1):
        print(f'Training model {i}')
        hist = train(model, dataset, config, epochs)
        histories.append(torch.tensor(hist,requires_grad=False))
    ensemble = models.Ensemble(model_list)
    histories = torch.stack(histories)
    return ensemble, torch.mean(histories,0)

#############################################################################
######################### Solution dependent source #########################
#############################################################################

# This can be made better, but the following is basically a
# copy paste of the above code adapted to take source functions
# taking three input arguments (t,x,u). It also modifies
# make_pointwise to supplies u in the training set.

def get_dataset_soldep(config: Config):
    M, N = config.mesh.M, config.mesh.N
    dt = config.scheme.dt
    V = torch.zeros((M,N), requires_grad=False)
    V[:,0] = config.init(config.mesh.centroids)
    for n in range(N-1):
        V[:,n+1] = config.scheme.vectorstep(V[:,n]) + dt*config.source(n*dt, config.mesh.centroids, V[:,n])
    return V
def make_pointwise_soldep(dataset, config: Config):
    M, N = config.mesh.M, config.mesh.N
    tgrid = torch.linspace(0, config.mesh.T, N)
    xgrid = config.mesh.centroids
    coords = torch.cartesian_prod(tgrid, xgrid)
    ugrid = torch.zeros(M*N)
    # coords = torch.cartesian_prod(tgrid, xgrid)
    # Need west, centre, east for LW.step, and ground truth
    target = torch.zeros((M*N, 4))
    for n, t in enumerate(tgrid[:-1]):
        for i, x in enumerate(xgrid):
            coord_ind = i + M*n
            west = M-1 if i==0 else i-1   # Ensures periodic BCs (positive a)
            east = 0 if i==M-1 else i+1
            ugrid[coord_ind] = dataset[i,n]
            # (w,c,e,target)
            target_row = [dataset[west,n], dataset[i,n], dataset[east,n], dataset[i,n+1]]
            target[coord_ind,:] = torch.tensor(target_row)
    coords = torch.cat([coords, ugrid.reshape((-1,1))], dim=1)
    return coords, target

def train_soldep(model: nn.Module, dataset, config: Config):
    M, N = config.mesh.M, config.mesh.N
    dt = config.scheme.dt
    coords, target = make_pointwise_soldep(dataset, config)
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
        batch = 0
        for (inputs, targets) in loader:
            if torch.any(inputs.isnan()):
                raise ValueError(f'ERROR: Encountered NaN in input at epoch {epoch}, batch {batch}')
            optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
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

def get_trained_ensemble_soldep(config: Config, n_models=5):
    dataset = get_dataset_soldep(config)
    histories = []
    model_list = [models.DenseNet(inp_dim=3) for _ in range(n_models)]
    for i, model in enumerate(model_list,1):
        print(f'Training model {i}')
        hist = train_soldep(model, dataset, config)
        histories.append(torch.tensor(hist,requires_grad=False))
    ensemble = models.Ensemble(model_list)
    histories = torch.stack(histories)
    return ensemble, torch.mean(histories,0)


if __name__ == '__main__':
    # amax = 1.5
    # T = 1
    # cfl = .7
    # M = 100
    # N = int(amax*T*M/(2*cfl))
    # dt = T / (N-1)
    # faces = torch.linspace(-1,1,M, requires_grad=False)

    # init = lambda x: torch.where(x<0,-1.,1.)
    # source = lambda t,x: torch.sin(2*torch.pi*t)*torch.sin(2*torch.pi*x)
    # mesh = flux.Mesh(faces, T, N)
    # scheme = flux.Godunov(lambda u:u**2/2,dt,mesh.dx)
    # config = Config(init, source, mesh, scheme)
    
    # dataset = get_wavefront_dataset(config)
    # model = models.DenseNet()
    # # Possible pretraining here
    # model = train(model, dataset, config)
    # Save model, do tests, etc.
    pass


