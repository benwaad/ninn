from train import *
from utils.tools import animate
from utils.kjetilplot import savePlot

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import simpson

import copy
import math
import time

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
    anim = False

    # Parameters defining the bathymetry
    mu = .25
    sig = 1/100
    def bathymetry(x):
        # ret = np.zeros_like(x)
        # freqs = np.random.uniform(5, 20, 10)
        # for freq in freqs:
        #     ret += np.sin(freq*x.numpy() + np.random.uniform(0,6)) / freq
        # ret = ret / len(freqs) + .3
        # return torch.from_numpy(ret)
        return torch.exp(-(x-mu)**2/sig) + .5*torch.exp(-(x-(mu-.25))**2/sig)
        # return 2.*x**2
    def bathym_x(x):
        return -2.*(x-mu)*torch.exp(-(x-mu)**2/sig)/sig - .5*2.*(x-(mu-.25))*torch.exp(-(x-(mu-.25))**2/sig)/sig
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

if __name__ == '__main__':
    main()