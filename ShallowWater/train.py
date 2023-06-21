import torch
import numpy as np
from utils import flux, models
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math
from typing import Type

# class Bathymetry:
#     @staticmethod
#     def call(x):
#         raise NotImplementedError()
#     @staticmethod
#     def diff(x):
#         raise NotImplementedError()

# class ExpSlope(Bathymetry):
#     @staticmethod
#     def call(x):
#         return torch.exp(-(x-1)**2)
#     @staticmethod
#     def diff(x):
#         return -2.*(x-1)*ExpSlope.call(x)


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
                west = dataset[n,i,:]   # Velocity equal
                # west[1] = -west[1]    # Velocity reflected
                # west[1] = 0.          # Velocity 0
            if east is None:
                east = dataset[n,i,:]   # Velocity equal
                # east[1] = -east[1]    # Velocity reflected
                # east[1] = 0.          # Velocity 0
            qgrid[coord_ind,:] = dataset[n,i,:]
            # (w,c,e,target)
            target_row = [west, dataset[n,i,:], east, dataset[n+1,i,:]]
            target[coord_ind,:,:] = torch.vstack(target_row)
    # coords = torch.cat([coords, ugrid.reshape((-1,1))], dim=1)
    # coords.requires_grad = True
    return coords, qgrid, target

def train(model: nn.Module, dataset, config: Config, epochs: int, direct=False):
    # First iteration: No explicit differentiation
    M, N = config.mesh.M, config.mesh.N
    dt = config.scheme.dt
    coords, qgrid, target = make_pointwise(dataset, config)
    # coords.require_grad = True
    criterion = nn.MSELoss()
    def compute_loss(model, coord_batch, Q_batch, target_batch):
        west, centre, east, true = torch.unbind(target_batch, dim=1)
        nosource = config.scheme.step(west, centre, east)
        x = coord_batch[:,1].detach()
        x.requires_grad = True
        bath_preds = model(x.reshape((-1,1))).flatten()
        if direct:
            bath_diff_preds = bath_preds
        else:
            bath_diff_preds = torch.autograd.grad(bath_preds, x, grad_outputs=torch.ones_like(bath_preds), create_graph=True)[0]
        sourcepreds = -config.g*Q_batch[:,0]*bath_diff_preds
        return criterion(nosource[:,1]+dt*sourcepreds, true[:,1])
    
    optim = torch.optim.Adam(model.parameters(), lr=0.0005)
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
    
def get_trained_ensemble(config: Config, n_models=5, epochs=8, fit_direct=False):
    dataset = get_dataset(config)
    histories = []
    model_list = [models.DenseNet(inp_dim=1) for _ in range(n_models)]
    best_loss = 1.0
    best_model = None
    best_index = 0
    for i, model in enumerate(model_list,1):
        print(f'Training model {i}')
        hist = train(model, dataset, config, epochs, direct=fit_direct)
        if (loss:=np.mean(np.array(hist)[-1000:])) < best_loss:
            best_index = i-1
            best_loss = loss
            best_model = model
        histories.append(torch.tensor(hist,requires_grad=False))
    ensemble = models.Ensemble([best_model])
    # histories = torch.stack(histories)
    _histories = histories[best_index]
    print(f'\nINFO: Selected model {best_index+1}.')
    return ensemble, _histories#torch.mean(histories,0)




def vectorized_train(model: nn.Module, dataset, config: Config, epochs: int):
    """Dataset can be incomplete, we only need data on u for t>0."""
    M, N = config.mesh.M, config.mesh.N
    dt = config.scheme.dt
    xgrid = config.mesh.centroids
    xgrid.requires_grad_(True)
    criterion = nn.MSELoss()
    def step_to_timepoint(model, init, n_target):
        current_sol = init
        for n in range(n_target):
            bath_preds = model(xgrid.reshape((-1,1))).flatten()
            grads = torch.autograd.grad(bath_preds, xgrid, grad_outputs=torch.ones_like(bath_preds), create_graph=True)[0]
            source = torch.zeros_like(current_sol)
            source[:,1] = -config.g*current_sol[:,0]*grads
            current_sol = config.scheme.vectorstep(current_sol) + dt*source
        return current_sol
    def compute_loss(predicted, target):
        return criterion(predicted[:,1], target[:,1])
    
    init = dataset[0,:,:]
    optim = torch.optim.Adam(model.parameters(), lr=0.0005)
    calc_iters = lambda n: 15 if n<N/5 else 5 if n<3*N/5 else 1
    # iters_per_timestep = 5
    # first_timestep_boost = 3
    history = []
    log_freq = max(math.floor(epochs/4), 1) # Max 4 epochs are logged, and we log first and last timestep loss
    # running_loss = 0.
    # loss_updates = 0
    for epoch in range(epochs):
        # batch = 0
        printepoch = True if epoch%log_freq==(log_freq-1) else False
        # print(f'INFO: Starting epoch {epoch+1}')
        first_timestep_loss = 0.
        last_timestep_loss = 0.
        for n in range(N-1):
            # if printepoch and (n in [0, int((N-2)/3), int(2*(N-2)/3), N-2]):
            #     print(f'INFO: Targeting timestep {n+1} / {N-1}')
            target = dataset[n+1,:,:]
            iters = calc_iters(n)
            # iters *= first_timestep_boost if n==0 else 1
            for it in range(iters):
                optim.zero_grad()
                preds = step_to_timepoint(model, init, n+1)
                loss = compute_loss(preds, target)
                loss.backward()
                optim.step()
                history.append(loss.item())
                if n==0:
                    first_timestep_loss += loss.item()
                elif n==(N-2):
                    last_timestep_loss += loss.item()
        if printepoch:
            first_avg_loss = first_timestep_loss/(calc_iters(0))
            last_avg_loss = last_timestep_loss/calc_iters(N-2)
            print(f'Epoch = {epoch+1}  first_step_loss = {first_avg_loss:5.2e}  last_step_loss = {last_avg_loss:5.2e}')


        #     if torch.any(Q_batch.isnan()):
        #         raise ValueError(f'ERROR: Encountered NaN in input at epoch {epoch}, batch {batch}')
        #     optim.zero_grad()
        #     loss = compute_loss(model, coord_batch, Q_batch, target_batch)
            
        #     # outputs = model(inputs)
        #     # loss = loss_fn(outputs, targets)
        #     loss.backward()
        #     optim.step()
        #     history.append(loss.item())
        #     running_loss += loss.item()
        #     loss_updates += 1
        #     # batch += 1
        # if epoch%log_freq==(log_freq-1):
        #     print(f'Epoch = {epoch+1}  avg loss = {running_loss/loss_updates:5.2e}')
        #     running_loss = 0.0
        #     loss_updates = 0
    return history

    
    