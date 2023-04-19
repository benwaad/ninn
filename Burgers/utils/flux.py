import torch

class Mesh:
    def __init__(self,faces, T, N):
        '''Assumes uniform faces'''
        self.faces = faces
        self.centroids = .5*(faces[:-1] + faces[1:])
        self.dx = torch.tensor(faces[1] - faces[0])
        self.M = len(self.centroids)
        self.T = T
        self.N = N

class Scheme:
    def __init__(self, F, dt, dx):
        '''F is the flux'''
        self.F = F
        self.dt = torch.tensor(dt, requires_grad=False)
        self.dx = torch.tensor(dx, requires_grad=False)
    def vectorstep(self, Un):
        raise NotImplementedError()
    def step(self, *args):
        raise NotImplementedError()
    
class LW(Scheme):
    def __init__(self, F, dt, dx):
        super().__init__(F, dt, dx)
    def step(self, west, centre, east):
        return centre - self.dt/self.dx*(self._halfstep(east,centre)-self._halfstep(centre,west))
    def _halfstep(self, u_ip1, u_i):
        return .5*(u_ip1+u_i) - .5*self.dt/self.dx*(self.F(u_ip1)-self.F(u_i))
    def vectorstep(self, Un):
        '''No ghost cells needed for periodic bdr'''
        west = torch.roll(Un, 1)
        east = torch.roll(Un,-1)
        return self.step(west, Un, east)

class Godunov(Scheme):
    def __init__(self, F, dt, dx):
        '''Made for convex fluxes with minimum in 0. Assumes batch dimension'''
        super().__init__(F, dt, dx)
        print(f'INFO: Created scheme with Δt/Δx = {dt/dx:.2f}.')
    def step(self, west, centre, east):
        return centre - self.dt/self.dx*(self._halfstep(centre,east)-self._halfstep(west,centre))
    def _halfstep(self, u_i, u_ip1):
        omega = torch.zeros_like(u_i)
        centre_max = torch.maximum(u_i, omega)
        east_min = torch.minimum(u_ip1, omega)
        return torch.maximum(self.F(centre_max), self.F(east_min))
    def vectorstep(self, Un):
        '''Assumes periodic boundary'''
        west = torch.roll(Un, 1)
        east = torch.roll(Un,-1)
        return self.step(west, Un, east)

