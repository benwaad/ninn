from train import *
from utils.tools import animate
import matplotlib.pyplot as plt
import numpy as np
from utils.kjetilplot import savePlot
from pathlib import Path
import copy
from scipy.integrate import simpson
import math
import time

def main():
    g = 9.81
    amax = 3 + (g*3)**.5
    T = 1.0
    cfl = .8
    M = 500
    N = int(amax*T*M/(2*cfl))
    print(f'INFO: Using N = {N}.')
    dt = T / (N-1)
    faces = torch.linspace(-1,1,M, requires_grad=False)
    anim = True

    # Parameters defining the bathymetry
    mu = .5
    sig = 1/100
    def bathymetry(x):
        # return torch.exp(-(x-1)**2)
        ret = np.zeros_like(x)
        freqs = np.random.uniform(5, 20, 10)
        for freq in freqs:
            ret += np.sin(freq*x.numpy() + np.random.uniform(0,6)) / freq
        ret = ret / len(freqs) + .3
        return torch.from_numpy(ret)
        return torch.exp(-(x-mu)**2/sig)
    def bathym_x(x):
        return -2.*(x-mu)*bathymetry(x)/sig
    def init(x):
        ret = torch.zeros([len(x),2])
        # ret[:,0] = torch.exp(-100*(x+.5)**2) + 1.5
        freqs = np.random.uniform(5, 20, 10)
        for freq in freqs:
            ret[:,0] += np.sin(freq*x + np.random.uniform(0,6)) / freq
        ret[:,0] = ret[:,0] / len(freqs) + 1.5
        # ret[:,0] = .2*torch.sin(4*torch.pi*x) + .5
        # ret[:,0] = torch.where(x<-.7,2,1.5)
        ret[:,0] = ret[:,0] - bathymetry(x)
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
        # ----------- ILLUSTRATION -----------
        velocity = (dataset[:,:,1]/dataset[:,:,0]).T
        histfig = plt.figure("Illustration", figsize=(10,5))
        # with plt.style.context('ggplot'):   # type: ignore
        ax = histfig.add_subplot(111)
        ax.fill_between(config.mesh.centroids, relative_surface[:,0].numpy(), seafloor[:,0].numpy(),alpha=.5, color="#000DFF")
        ax.fill_between(config.mesh.centroids, seafloor[:,0].numpy(), alpha=.7, color='sienna')
        ax.vlines([-1,1], 0, 2, color='black', linewidths=.8)
        ax.set_ylim((0.0,2.0)) # type: ignore
        # ax.set_xlabel('$x$')
        ax.get_yaxis().set_visible(False)
        for spine in ['left', 'top', 'right']:
            ax.spines[spine].set_visible(False)
        plt.show()
        # ------------------------------------
        return
        print('Starting animation process...')
        animate('height.mp4', torch.linspace(0,T,N).detach(), config.mesh.centroids, relative_surface, seafloor=seafloor[:,0], total_time=T)
        animate('velocity.mp4', torch.linspace(0,T,N).detach(), config.mesh.centroids, velocity, seafloor=seafloor[:,0], total_time=T, ylims=(-2.,2.))
        print('Finished animation')
    # ------------------------

    model = models.DenseNet(inp_dim=1)
    start = time.time()
    history = train(model, dataset, config, epochs=1)
    print(f'INFO: Training finished in {time.time()-start:.1f} s.')
    np.save('history.npy', history)

    histfig = plt.figure("History", figsize=(8,5))
    with plt.style.context('ggplot'):   # type: ignore
        ax = histfig.add_subplot(111)
        ax.plot(history)
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')

    xgrid = config.mesh.centroids.detach()
    predfig = plt.figure("Predictions")
    with plt.style.context('ggplot'):   # type: ignore
        ax = predfig.add_subplot(111)
        ax.plot(xgrid, bathym_x(xgrid), label='$B_x$')
        ax.plot(xgrid, model(xgrid.reshape((-1,1))).squeeze().detach(), label='Predicted')
        ax.set_xlabel('$x$')
        ax.legend()
    
    plt.show()


if __name__ == '__main__':
    main()