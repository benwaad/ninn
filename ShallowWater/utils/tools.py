import torch
import torch.nn.functional as F

def softargmax(u, xgrid, beta=1e10):
    weighted_xpos = xgrid * torch.softmax(u*beta, dim=-1)
    return weighted_xpos.sum()

def softwavefinder(u, xgrid):
    dx = xgrid[1] - xgrid[0]
    pad = F.pad(u.reshape((1,1,-1)), (0,1), mode='circular').squeeze()  # Torch's nonconstant pad only works on arrays of dim 3 and higher for some reason
    # ddu = (pad[2:]-2*pad[1:-1]+pad[:-2]) / dx**2
    du = abs((pad[1:]-pad[:-1])/dx)
    return softargmax(du, xgrid)