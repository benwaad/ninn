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

class CUW(Scheme):
    def __init__(self,F, dt, dx, periodic=True):
        '''Central-Upwind scheme for systems, due to A. Kurganov, G. Petrova, B. Popov. 1D Shallow Water hard coded in.'''
        super().__init__(F,dt,dx)
        self.periodic = periodic
        print(f'INFO: Created scheme with Δt/Δx = {dt/dx:.2f}.')
    def step(self, west, centre, east):
        return centre - self.dt/self.dx*(self._halfstep(centre,east)-self._halfstep(west,centre))
    def _halfstep(self, Q_i, Q_ip1):
        aplus, aminus = self.get_a_plus_minus(Q_i, Q_ip1)
        flux = (aplus*self.F(Q_i)-aminus*self.F(Q_ip1))/(aplus-aminus) + aplus*aminus/(aplus-aminus)*(Q_ip1-Q_i)
        return torch.nan_to_num(flux, nan=0.0)   # NaN happens when h=0 both left and right, then flux is 0
        # return flux
        
    def vectorstep(self, Qn):
        '''Batch dimension is outer'''
        west = torch.roll(Qn, 1, dims=0)
        east = torch.roll(Qn,-1, dims=0)
        # if not self.periodic:
        #     west[0] = west[1]
        #     east[-1] = east[-2]
        if not self.periodic:
            west[0] = west[1]
            east[-1] = east[-2]
            # west[0,1] = -west[0,1]
            # east[-1,1] = -east[-1,1]
            west[0,1] = 0.
            east[-1,1] = 0.
        return self.step(west, Qn, east)
    def get_eigenvals(self, Q):
        # h can be zero (Q[:,0]), so we mask it away, letting hu^2 default to zero
        # mask = (Q[:,0] >= 1e-6)
        # hu = torch.zeros_like(Q[:,0])
        hu = Q[:,1]/Q[:,0]
        big =  hu + torch.sqrt(9.81*Q[:,0])
        small =  hu - torch.sqrt(9.81*Q[:,0])
        return big, small
    def get_a_plus_minus(self, QL, QR):
        lambda_m_left, lambda_1_left = self.get_eigenvals(QL)
        lambda_m_right, lambda_1_right = self.get_eigenvals(QR)
        zeros = torch.zeros_like(lambda_m_left)
        aplus = torch.maximum(torch.maximum(lambda_m_left, lambda_m_right), zeros)
        aminus = torch.minimum(torch.minimum(lambda_1_left, lambda_1_right), zeros)
        return aplus.reshape([-1,1]), aminus.reshape([-1,1])    # Reshapes help broadcasting in _halfstep

    





