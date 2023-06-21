import torch
import torch.nn.functional as F
from matplotlib import animation
import matplotlib.pyplot as plt
import sys, os

def softargmax(u, xgrid, beta=1e10):
    weighted_xpos = xgrid * torch.softmax(u*beta, dim=-1)
    return weighted_xpos.sum()

def softwavefinder(u, xgrid):
    dx = xgrid[1] - xgrid[0]
    pad = F.pad(u.reshape((1,1,-1)), (0,1), mode='circular').squeeze()  # Torch's nonconstant pad only works on arrays of dim 3 and higher for some reason
    # ddu = (pad[2:]-2*pad[1:-1]+pad[:-2]) / dx**2
    du = abs((pad[1:]-pad[:-1])/dx)
    return softargmax(du, xgrid)

def create_animation_artists(ax: plt.Axes, tgrid, xgrid, *args, seafloor=None):
    artists = []
    for i in range(len(tgrid)):
        frame = []
        numline, = ax.plot(xgrid, args[0][:,i], color="#6533DB")
        frame.append(numline)
        if len(args) == 2:
           trueline, = ax.plot(xgrid, args[1][:,i], '--', alpha=.6, color="#D74848")
           frame.append(trueline)
        if seafloor is not None:
            floor = ax.fill_between(xgrid, seafloor, alpha=.7, color='sienna')
            frame.append(floor)
        ann = ax.annotate(f'$t={tgrid[i]:.2f}$', (0.5,1.03), xycoords='axes fraction', ha='center')
        frame.append(ann)
        artists.append(frame)
    return artists

def animate(name, tgrid, xgrid, *args, seafloor=None, total_time=1., ylims=(0.0,4.0)):
    '''Send true solution last'''
    time_ms = 1000*total_time # ms
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    ax.set_ylim(ylims)  # type: ignore
    artists = create_animation_artists(ax, tgrid, xgrid, *args, seafloor=seafloor)
    anim = animation.ArtistAnimation(fig, artists, interval=time_ms / len(artists), repeat=True)
    writer = animation.FFMpegWriter(fps=int(len(artists)/total_time))
    anim.save(name, writer=writer)


def disable_print():
    sys.stdout = open(os.devnull, 'w')
def enable_print():
    sys.stdout = sys.__stdout__