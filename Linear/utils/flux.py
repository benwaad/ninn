import torch

class Mesh:
    def __init__(self,faces, T, N):
        '''Assumes uniform faces'''
        self.faces = faces
        self.centroids = .5*(faces[:-1] + faces[1:])
        self.dx = torch.tensor(faces[1] - faces[0], requires_grad=False)
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
    def max(self, a, b):
        '''Convenience'''
        return torch.maximum(a,b)
    def min(self, a, b):
        '''Convenience'''
        return torch.minimum(a,b)
    def vectorstep(self, Un):
        '''Un is sent in with ghost cell paddings, Dirichlet'''
        internals = Un[:,1:-1]
        rightshift = Un[:,2:]
        leftshift = Un[:,:-2]
        zero = torch.zeros_like(internals, device=internals.device)
        F_p = self.max(self.F(self.max(internals,zero)), self.F(self.min(rightshift,zero)))
        F_m = self.max(self.F(self.max(leftshift,zero)), self.F(self.min(internals,zero)))
        next_internals = internals - self.dt/self.dx * (F_p-F_m)  # type:ignore
        leftcond = Un[:,0:1]
        rightcond = Un[:,-1:]
        return torch.cat([leftcond, next_internals, rightcond], 1)

    def step(self):
        pass

