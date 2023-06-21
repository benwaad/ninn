import os
from train import *
from utils.tools import animate
from utils.kjetilplot import savePlot

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import simpson
from dataclasses import dataclass
from datetime import datetime

import copy
import math
import time
import pickle

def main():
    g = 9.81
    amax = 3 + (g*3)**.5
    T = 1.0
    cfl = .8
    M = 100
    N = int(amax*T*M/(2*cfl))
    print(f'INFO: Using N = {N}.')
    dt = T / (N-1)
    faces = torch.linspace(-1,1,M, requires_grad=False)
    anim = True

    # Parameters defining the bathymetry
    mu = .25
    sig = 1/10
    def bathymetry(x):
        # ret = np.zeros_like(x)
        # freqs = np.random.uniform(5, 20, 10)
        # for freq in freqs:
        #     ret += np.sin(freq*x.numpy() + np.random.uniform(0,6)) / freq
        # ret = ret / len(freqs) + .3
        # return torch.from_numpy(ret)
        return torch.exp(-(x-mu)**2/sig**2) + .5*torch.exp(-(x-(mu-.25))**2/sig**2)
        # return 2.*x**2
    def bathym_x(x):
        return -2.*(x-mu)*torch.exp(-(x-mu)**2/sig**2)/sig**2 - .5*2.*(x-(mu-.25))*torch.exp(-(x-(mu-.25))**2/sig**2)/sig**2
        # return 4.*x
    
    def init(x):
        ret = torch.zeros([len(x),2])
        # freqs = np.random.uniform(5, 20, 10)
        # for freq in freqs:
        #     ret[:,0] += np.sin(freq*x + np.random.uniform(0,6)) / freq
        # ret[:,0] = ret[:,0] / len(freqs) + 1.5
        # ret[:,0] = .2*torch.sin(4*torch.pi*x) + .5
        # ret[:,0] = torch.where(x<-.7,2,1.5)
        ret[:,0] = torch.exp(-100*(x+.5)**2) + 1.5
        ret[:,0] = torch.max(ret[:,0] - bathymetry(x), torch.zeros_like(x))
        ret[:,1] = 0.
        return ret
    # def source(t,x,Q):
    #     ret = torch.zeros([len(x), 2])
    #     ret[:,1] = -g*Q[:,0]*bathym_x(x)
    #     return ret

    mesh = flux.Mesh(faces, T, N)
    def sw_flux(Q):
        ret = torch.zeros_like(Q)
        ret[:,0] = Q[:,1]
        # h (Q[:,0]) can be zero, which corresponds to the first term being zero
        # In order to avoid division by zero, we mask these values out with zero as default
        # mask = (Q[:,0] >= 1e-6)
        # hu2 = torch.zeros_like(Q[:,0])
        hu2= Q[:,1]**2/Q[:,0]
        ret[:,1] = hu2 + g*Q[:,0]**2/2
        return ret
    scheme = flux.CUW(sw_flux,dt,mesh.dx, periodic=False)
    config = Config(init, bathym_x, mesh, scheme)

    dataset = get_dataset(config)
    # ------- ANIMATE --------
    if anim:
        seafloor = torch.zeros_like(dataset[:,:,0].T)
        for i in range(len(seafloor[0])):
            seafloor[:,i] = bathymetry(config.mesh.centroids)
        relative_surface = dataset[:,:,0].T + seafloor
        velocity = (dataset[:,:,1]/dataset[:,:,0]).T
        # # ----------- ILLUSTRATION -----------
        # histfig = plt.figure("Illustration", figsize=(10,5))
        # # with plt.style.context('ggplot'):   # type: ignore
        # ax = histfig.add_subplot(111)
        # ax.fill_between(config.mesh.centroids, relative_surface[:,0].numpy(), seafloor[:,0].numpy(),alpha=.5, color="#000DFF")
        # ax.fill_between(config.mesh.centroids, seafloor[:,0].numpy(), alpha=.7, color='sienna')
        # ax.vlines([-1,1], 0, 2, color='black', linewidths=.8)
        # ax.set_ylim((0.0,2.0)) # type: ignore
        # # ax.set_xlabel('$x$')
        # ax.get_yaxis().set_visible(False)
        # for spine in ['left', 'top', 'right']:
        #     ax.spines[spine].set_visible(False)
        # plt.show()
        # return
        # # ------------------------------------
        print('Starting animation process...')
        animate('height.mp4', torch.linspace(0,T,N).detach(), config.mesh.centroids, relative_surface, seafloor=seafloor[:,0], total_time=T)
        animate('velocity.mp4', torch.linspace(0,T,N).detach(), config.mesh.centroids, velocity, seafloor=seafloor[:,0], total_time=T, ylims=(-2.,2.))
        print('Finished animation')
    # ------------------------

    model = models.DenseNet(inp_dim=1)
    start = time.time()
    # history = train(model, dataset, config, epochs=1)
    model, history = get_trained_ensemble(config, n_models=5, epochs=4, fit_direct=False)
    print(f'INFO: Training finished in {time.time()-start:.1f} s.')
    np.save('history.npy', history)

    histfig = plt.figure("history", figsize=(8,5))
    with plt.style.context('ggplot'):   # type: ignore
        ax = histfig.add_subplot(111)
        ax.plot(history)
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')

    # -------------- Plot true and predicted bath --------------
    xgrid = config.mesh.centroids.detach()
    predfig = plt.figure("preds", figsize=(8,2.6))
    with plt.style.context('ggplot'):   # type: ignore
        ax = predfig.add_subplot(111)
        ax.plot(xgrid, bathymetry(xgrid), label=r'$B(x)$')    # TODO
        prd = model(xgrid.reshape((-1,1))).squeeze().detach()
        ax.plot(xgrid, prd-prd[0], label=r'$\hat{B}(x)$')
        ax.set_xlabel('$x$')
        ax.legend()
    # ----------------------------------------------------------

    # -------------- Plot true and predicted bath_x ------------
    predfig = plt.figure("diff_preds", figsize=(8,2.6))
    with plt.style.context('ggplot'):   # type: ignore
        ax = predfig.add_subplot(111)
        ax.plot(xgrid, bathym_x(xgrid), label=r'$B_x(x)$')
        xgrid.requires_grad_(True)
        prd = model(xgrid.reshape((-1,1))).squeeze()
        d_prd = torch.autograd.grad(prd, xgrid, torch.ones_like(prd))[0]
        xgrid.detach_()
        ax.plot(xgrid, d_prd, label=r'$\hat{B}_x(x)$')
        ax.set_xlabel('$x$')
        ax.legend()
    # ----------------------------------------------------------
    
    # -------------- Plot true and predicted wave --------------
    times = [0, .1, .2]
    # bhat = lambda x: model(x.reshape((-1,1))).flatten()
    def bhat(x):
        x.requires_grad_(True)
        y = model(x.reshape((-1,1))).flatten()
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    predconfig = Config(config.init, bhat, config.mesh, config.scheme)
    preddata = get_dataset(predconfig).detach()
    seafloor = bathymetry(xgrid).numpy()
    wavefig = plt.figure("wave_preds", figsize=(8,8))
    with plt.style.context('ggplot'):   # type: ignore
        for i, t in enumerate(times, start=1):
            ax = wavefig.add_subplot(len(times), 1, i)
            idx = int(t/config.scheme.dt)
            ax.plot(xgrid, dataset[idx,:,0]+seafloor, '--',  color="#9B82D5", label='True')
            ax.plot(xgrid, preddata[idx,:,0]+seafloor, color="#6533DB", label='Predicted')
            ax.fill_between(xgrid, seafloor, alpha=.8, color='sienna')
            ax.set_ylim(0, 2.7)
            ax.legend(loc='upper right')
            # ax.set_title(f'{t:.2f} s')
            ax.annotate(f'$t={t:.2f}$ s', (-0.745,0.7))
        ax.set_xlabel('$x$')    # Only set x label on bottom axis
    # ----------------------------------------------------------

    
    plt.show()


def vectorized_main():
    g = 9.81
    amax = 3 + (g*3)**.5
    T = .2
    cfl = .8
    M = 100
    N = int(amax*T*M/(2*cfl))
    print(f'INFO: Using N = {N}.')
    dt = T / (N-1)
    faces = torch.linspace(-1,1,M, requires_grad=False)
    anim = False

    # Parameters defining the bathymetry
    mu = .25
    sig = 1/10
    def bathymetry(x):
        return torch.exp(-(x-mu)**2/sig**2) + .5*torch.exp(-(x-(mu-.25))**2/sig**2)
        # return torch.sin(torch.pi*torch.cos(torch.pi*x))
    def bathym_x(x):
        return -2.*(x-mu)*torch.exp(-(x-mu)**2/sig**2)/sig**2 - .5*2.*(x-(mu-.25))*torch.exp(-(x-(mu-.25))**2/sig**2)/sig**2
        # return -torch.pi**2*torch.sin(torch.pi*x)*torch.cos(torch.pi*torch.cos(torch.pi*x))
    
    def init(x):
        ret = torch.zeros([len(x),2])
        ret[:,0] = torch.exp(-100*(x+.5)**2) + 1.5
        ret[:,0] = torch.max(ret[:,0] - bathymetry(x), torch.zeros_like(x))
        ret[:,1] = 0.
        return ret

    mesh = flux.Mesh(faces, T, N)
    def sw_flux(Q):
        ret = torch.zeros_like(Q)
        ret[:,0] = Q[:,1]
        hu2= Q[:,1]**2/Q[:,0]
        ret[:,1] = hu2 + g*Q[:,0]**2/2
        return ret
    scheme = flux.CUW(sw_flux,dt,mesh.dx, periodic=False)
    config = Config(init, bathym_x, mesh, scheme)

    dataset = get_dataset(config)
    dataset[1:,:,0] = torch.nan     # Simulating unknown data
    # ------- ANIMATE --------
    if anim:
        seafloor = torch.zeros_like(dataset[:,:,0].T)
        for i in range(len(seafloor[0])):
            seafloor[:,i] = bathymetry(config.mesh.centroids)
        relative_surface = dataset[:,:,0].T + seafloor
        velocity = (dataset[:,:,1]/dataset[:,:,0]).T
        print('Starting animation process...')
        animate('height.mp4', torch.linspace(0,T,N).detach(), config.mesh.centroids, relative_surface, seafloor=seafloor[:,0], total_time=T)
        animate('velocity.mp4', torch.linspace(0,T,N).detach(), config.mesh.centroids, velocity, seafloor=seafloor[:,0], total_time=T, ylims=(-2.,2.))
        print('Finished animation')
    # ------------------------

    model = models.DenseNet(inp_dim=1)
    start = time.time()
    EPOCHS = 3
    history = vectorized_train(model, dataset, config, epochs=EPOCHS)
    # model, history = get_trained_ensemble(config, n_models=5, epochs=4, fit_direct=False)
    print(f'INFO: Training finished in {time.time()-start:.1f} s.')
    np.save('history.npy', history)

    histfig = plt.figure("history", figsize=(8,5))
    with plt.style.context('ggplot'):   # type: ignore
        ax = histfig.add_subplot(111)
        ax.plot(history)
        ax.vlines([k*len(history)//EPOCHS for k in range(1,EPOCHS)], ymin=np.min(history), ymax=np.max(history), linestyles='dashed', colors='black', linewidths=1) # type: ignore
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')

    # -------------- Plot true and predicted bath --------------
    xgrid = config.mesh.centroids.detach()
    predfig = plt.figure("preds", figsize=(8,2.6))
    with plt.style.context('ggplot'):   # type: ignore
        ax = predfig.add_subplot(111)
        ax.plot(xgrid, bathymetry(xgrid), label=r'$B(x)$')    # TODO
        prd = model(xgrid.reshape((-1,1))).squeeze().detach()
        ax.plot(xgrid, prd-prd[0], label=r'$\hat{B}(x)$')
        ax.set_xlabel('$x$')
        ax.legend()
    # ----------------------------------------------------------

    # -------------- Plot true and predicted bath_x ------------
    predfig = plt.figure("diff_preds", figsize=(8,2.6))
    with plt.style.context('ggplot'):   # type: ignore
        ax = predfig.add_subplot(111)
        ax.plot(xgrid, bathym_x(xgrid), label=r'$B_x(x)$')
        xgrid.requires_grad_(True)
        prd = model(xgrid.reshape((-1,1))).squeeze()
        d_prd = torch.autograd.grad(prd, xgrid, torch.ones_like(prd))[0]
        xgrid.detach_()
        ax.plot(xgrid, d_prd, label=r'$\hat{B}_x(x)$')
        ax.set_xlabel('$x$')
        ax.legend()
    # ----------------------------------------------------------
    
    # -------------- Plot true and predicted wave --------------
    times = [0, .1, .2]
    # bhat = lambda x: model(x.reshape((-1,1))).flatten()
    def bhat(x):
        x.requires_grad_(True)
        y = model(x.reshape((-1,1))).flatten()
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    predconfig = Config(config.init, bhat, config.mesh, config.scheme)
    preddata = get_dataset(predconfig).detach()
    seafloor = bathymetry(xgrid).numpy()
    wavefig = plt.figure("wave_preds", figsize=(8,8))
    with plt.style.context('ggplot'):   # type: ignore
        for i, t in enumerate(times, start=1):
            ax = wavefig.add_subplot(len(times), 1, i)
            idx = int(t/config.scheme.dt)
            ax.plot(xgrid, dataset[idx,:,0]+seafloor, '--',  color="#9B82D5", label='True')
            ax.plot(xgrid, preddata[idx,:,0]+seafloor, color="#6533DB", label='Predicted')
            ax.fill_between(xgrid, seafloor, alpha=.8, color='sienna')
            ax.set_ylim(0, 2.7)
            ax.legend(loc='upper right')
            # ax.set_title(f'{t:.2f} s')
            ax.annotate(f'$t={t:.2f}$ s', (-0.745,0.7))
        ax.set_xlabel('$x$')    # Only set x label on bottom axis
    # ----------------------------------------------------------

    
    plt.show()

@dataclass
class PredictionCollection:
    x: torch.Tensor
    bath: torch.Tensor
    diff: torch.Tensor
    sol: torch.Tensor
    runtime: float

def compute_preds(M, pointwise=True):
    g = 9.81
    amax = 3 + (g*3)**.5
    T = .2
    cfl = .8
    # N = int(amax*T*M/(2*cfl))
    N = 105
    print(f'INFO: Using N = {N}.')
    dt = T / (N-1)
    faces = torch.linspace(-1,1,M, requires_grad=False)

    # Parameters defining the bathymetry
    mu = .25
    sig = 1/10
    def bathymetry(x):
        return torch.exp(-(x-mu)**2/sig**2) + .5*torch.exp(-(x-(mu-.25))**2/sig**2)
        # return torch.sin(torch.pi*torch.cos(torch.pi*x)) + 1.0
    def bathym_x(x):
        return -2.*(x-mu)*torch.exp(-(x-mu)**2/sig**2)/sig**2 - .5*2.*(x-(mu-.25))*torch.exp(-(x-(mu-.25))**2/sig**2)/sig**2
        # return -torch.pi**2*torch.sin(torch.pi*x)*torch.cos(torch.pi*torch.cos(torch.pi*x))
    
    def init(x):
        ret = torch.zeros([len(x),2])
        ret[:,0] = torch.exp(-100*(x+.5)**2) + 1.5
        ret[:,0] = torch.max(ret[:,0] - bathymetry(x), torch.zeros_like(x))
        # ret[:,0] = ret[:,0] - bathymetry(x)
        if torch.any(ret[:,0]<0.):
            raise ValueError("Encountered negative height in initial condition")
        ret[:,1] = 0.
        return ret

    mesh = flux.Mesh(faces, T, N)
    def sw_flux(Q):
        ret = torch.zeros_like(Q)
        ret[:,0] = Q[:,1]
        hu2= Q[:,1]**2/Q[:,0]
        ret[:,1] = hu2 + g*Q[:,0]**2/2
        return ret
    scheme = flux.CUW(sw_flux,dt,mesh.dx, periodic=False)
    config = Config(init, bathym_x, mesh, scheme)

    dataset = get_dataset(config)
    if not pointwise:
        dataset[1:,:,0] = torch.nan     # Simulating unknown data

    model = models.DenseNet(inp_dim=1)
    start = time.time()

    if pointwise:
        EPOCHS = 20 if M>80 else 100
    else:
        EPOCHS = 1 if M>80 else 1

    # EPOCHS = 1 if M>20 else 3
    train_fnc = train if pointwise else vectorized_train
    history = train_fnc(model, dataset, config, epochs=EPOCHS)
    # model, history = get_trained_ensemble(config, n_models=5, epochs=4, fit_direct=False)
    elapsed = time.time()-start
    print(f'INFO: Training (M={M} {"pw" if pointwise else "full"}) finished in {elapsed:.1f} s.')
    # np.save('history.npy', history)

    # histfig = plt.figure("history", figsize=(8,5))
    # with plt.style.context('ggplot'):   # type: ignore
    #     ax = histfig.add_subplot(111)
    #     ax.plot(history)
    #     ax.vlines([k*len(history)//EPOCHS for k in range(1,EPOCHS)], ymin=np.min(history), ymax=np.max(history), linestyles='dashed', colors='black', linewidths=1) # type: ignore
    #     ax.set_yscale('log')
    #     ax.set_xlabel('Iteration')

    # Configuration for generating predictions
    predmesh = flux.Mesh(torch.linspace(-1,1,100, requires_grad=False),T,N)
    predscheme = flux.CUW(sw_flux, dt, predmesh.dx, periodic=False)
    highres_config = Config(init, bathym_x, predmesh, predscheme)
    
    xgrid = highres_config.mesh.centroids.detach()
    # -------------- Plot true and predicted bath --------------
    bathpreds = model(xgrid.reshape((-1,1))).squeeze().detach()
    bathpreds = bathpreds - bathpreds[0] + bathymetry(xgrid)[0]
    # -------------- Plot true and predicted bath_x ------------
    xgrid.requires_grad_(True)
    prd = model(xgrid.reshape((-1,1))).squeeze()
    diffpreds = torch.autograd.grad(prd, xgrid, torch.ones_like(prd))[0]
    xgrid.detach_()
    # predfig = plt.figure("diff_preds", figsize=(8,2.6))
    # with plt.style.context('ggplot'):   # type: ignore
    #     ax = predfig.add_subplot(111)
    #     ax.plot(xgrid, bathym_x(xgrid), label=r'$B_x(x)$')
    #     xgrid.requires_grad_(True)
    #     prd = model(xgrid.reshape((-1,1))).squeeze()
    #     d_prd = torch.autograd.grad(prd, xgrid, torch.ones_like(prd))[0]
    #     xgrid.detach_()
    #     ax.plot(xgrid, d_prd, label=r'$\hat{B}_x(x)$')
    #     ax.set_xlabel('$x$')
    #     ax.legend()
    # ----------------------------------------------------------
    
    # -------------- Plot true and predicted wave --------------
    # times = [0, .1, .2]
    # bhat = lambda x: model(x.reshape((-1,1))).flatten()
    def bhat(x):
        x.requires_grad_(True)
        y = model(x.reshape((-1,1))).flatten()
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=False)[0]
    predconfig = Config(init, bhat, predmesh, predscheme)
    highres_data = get_dataset(highres_config)[-1,:,:].detach()
    solpreds = get_dataset(predconfig)[-1,:,:].detach()
    seafloor = bathymetry(xgrid).numpy()
    
    predcollection = PredictionCollection(xgrid, bathpreds, diffpreds, solpreds, elapsed)
    truecollection = PredictionCollection(xgrid, bathymetry(xgrid), bathym_x(xgrid), highres_data, 0.0)
    return predcollection, truecollection

def get_L2_error(x: torch.Tensor, pred:torch.Tensor, true: torch.Tensor):
    err = (pred-true)**2
    den = true**2
    if len(pred.shape)>1:
        err = err.sum(dim=-1)
        den = den.sum(dim=-1)
    return simpson(err, x) / simpson(den, x)

def print_stats(pred: PredictionCollection, true: PredictionCollection):
    x = pred.x
    bath = get_L2_error(x, pred.bath, true.bath)
    diff = get_L2_error(x, pred.diff, true.diff)
    sol = get_L2_error(x, pred.sol, true.sol)
    print(f"'bath': {bath:.4e},   'diff': {diff:.4e},   'sol': {sol:.4e},   'runtime': {pred.runtime:.1f}\n")

def comparison_main():
    M_high = 100
    M_low = 20
    start = time.time()

    print(f'\nRunning M={M_high} pointwise')
    high_comp, true = compute_preds(M=M_high, pointwise=True)
    print(f'Finished M={M_high} pointwise. Errors:')
    print_stats(high_comp, true)

    print(f'\nRunning M={M_high} full')
    high_incomp, _ = compute_preds(M=M_high, pointwise=False)
    print(f'Finished M={M_high} full. Errors:')
    print_stats(high_incomp, true)

    print(f'\nRunning M={M_low} pointwise')
    low_comp, _ = compute_preds(M=M_low, pointwise=True)
    print(f'Finished M={M_low} pointwise. Errors:')
    print_stats(low_comp, true)

    print(f'\nRunning M={M_low} full')
    low_incomp, _ = compute_preds(M=M_low, pointwise=False)
    print(f'Finished M={M_low} full. Errors:')
    print_stats(low_incomp, true)

    print(f'\nFinished all in {time.time()-start:.1f} s.')

    timestr = datetime.now().strftime('%m-%d-%H%M')
    exprpath = f'./ShallowWater/experiments/{timestr}'
    os.makedirs(exprpath)
    with open(f'{exprpath}/true.pickle', 'wb') as truefile:
        pickle.dump(true, truefile)
    with open(f'{exprpath}/high_comp.pickle', 'wb') as file:
        pickle.dump(high_comp, file)
    with open(f'{exprpath}/high_imcomp.pickle', 'wb') as file:
        pickle.dump(high_incomp, file)
    with open(f'{exprpath}/low_comp.pickle', 'wb') as file:
        pickle.dump(low_comp, file)
    with open(f'{exprpath}/low_incomp.pickle', 'wb') as file:
        pickle.dump(low_incomp, file)
    
    print(f'Results saved to {exprpath}')

    

    xgrid = high_comp.x
    alpha = 1.
    # Bathymetry
    predfig = plt.figure("preds", figsize=(8,2.6))
    with plt.style.context('ggplot'):   # type: ignore
        ax = predfig.add_subplot(111)
        ax.plot(xgrid, true.bath, label='True')
        ax.plot(xgrid, high_comp.bath, alpha=alpha, label=f'$M={M_high}$ (pw)')
        ax.plot(xgrid, high_incomp.bath, alpha=alpha, label=f'$M={M_high}$ (full)')
        ax.plot(xgrid, low_comp.bath, alpha=alpha, label=f'$M={M_low}$ (pw)')
        ax.plot(xgrid, low_incomp.bath, alpha=alpha, label=f'$M={M_low}$ (full)')
        ax.set_xlabel('$x$')
        ax.legend(loc='upper left')
    
    # Bathymetry derivative
    diffig = plt.figure("diff", figsize=(8,2.6))
    with plt.style.context('ggplot'):   # type: ignore
        ax = diffig.add_subplot(111)
        ax.plot(xgrid, true.diff, label='True')
        ax.plot(xgrid, high_comp.diff, alpha=alpha, label=f'$M={M_high}$ (pw)')
        ax.plot(xgrid, high_incomp.diff, alpha=alpha, label=f'$M={M_high}$ (full)')
        ax.plot(xgrid, low_comp.diff, alpha=alpha, label=f'$M={M_low}$ (pw)')
        ax.plot(xgrid, low_incomp.diff, alpha=alpha, label=f'$M={M_low}$ (full)')
        ax.set_xlabel('$x$')
        ax.legend(loc='upper left')
    
    # Solution
    floor = true.bath
    solfig = plt.figure("wave", figsize=(8,2.6))
    with plt.style.context('ggplot'):   # type: ignore
        ax = solfig.add_subplot(111)
        ax.plot(xgrid, true.sol[:,0]+floor, label='True')
        ax.plot(xgrid, high_comp.sol[:,0]+floor, alpha=alpha, label=f'$M={M_high}$ (pw)')
        ax.plot(xgrid, high_incomp.sol[:,0]+floor, alpha=alpha, label=f'$M={M_high}$ (full)')
        ax.plot(xgrid, low_comp.sol[:,0]+floor, alpha=alpha, label=f'$M={M_low}$ (pw)')
        ax.plot(xgrid, low_incomp.sol[:,0]+floor, alpha=alpha, label=f'$M={M_low}$ (full)')
        ax.fill_between(xgrid, floor.numpy(), alpha=.8, color='sienna')
        ax.set_xlabel('$x$')
        ax.legend(loc='upper left')

    plt.show()





if __name__ == '__main__':
    comparison_main()

