import numpy as np
import matplotlib.pyplot as plt


class Mesh:
    def __init__(self, domain, N, cfl=0.4, tmax=2):
        self.domain = domain
        self.N = N      # Number of points in computational domain
        self.cfl = cfl
        self.dt = -1
        self.tmax = tmax

        self.dx = (domain[1]-domain[0]) / N
    
class Equation:
    def __init__(self, a_func, initial, source):
        '''Currently only periodic BC'''
        self.a_func = a_func
        self.initial = initial
        self.source = source


class Config:
    def __init__(self, mesh: Mesh, equation: Equation):
        self.mesh = mesh
        self.equation = equation
        cfl = self.mesh.cfl
        self.mesh.dt = cfl * self.mesh.dx / self.equation.a_func

# -------------- SOLVERS --------------

class Solver:
    def __init__(self, config: Config):
        self.mesh = config.mesh
        self.equation = config.equation

    def solve(self):
        raise NotImplementedError

class ExplicitUpwind(Solver):
    def __init(self, config):
        super().__init__(config)
    
    def get_A(self, N, r):
        # First implement for constant a>0 (a_func is a positive number)
        a = self.equation.a_func
        A = np.zeros((N,N))
        # hi = np.ones(N-2) * r*(abs(a)-a)/2
        # di = np.ones(N-1) * (-r*abs(a)+1)
        # lo = np.ones(N-2) * r*(abs(a)+a)/2
        di = np.ones(N)   * (1-r*a)
        lo = np.ones(N-1) * (r*a)

        dslice = np.arange(len(A))
        A[dslice, dslice] = di
        # A[dslice[:-1], dslice[1:]] = hi
        A[dslice[1:], dslice[:-1]] = lo
        A[0,-1] = r*a
        return A
    
    def solve(self):

        dom = self.mesh.domain
        dx = self.mesh.dx
        dt = self.mesh.dt
        N = self.mesh.N
        tsteps = int(self.mesh.tmax/dt)
        # t = np.linspace(0, self.mesh.tmax, tsteps)
        A = self.get_A(N, dt/dx)
        # print(A)
        # Ainv = np.linalg.inv(A)
        # Solution container
        U = np.zeros((N, tsteps))
        x = np.linspace(dom[0], dom[1]-dx, N)
        print()
        # print(x)
        U[:,0] = np.sin(2*np.pi*x)
        # np.savetxt('debug.txt', A, fmt='%.2f')
        # Solution loop
        for i in range(tsteps-1):
            U[:,i+1] = A @ U[:,i]
        
        return U


        

if __name__ == '__main__':
    plt.style.use('ggplot')
    cfl = 0.4
    mesh = Mesh([-1,1], 50, cfl=cfl, tmax=1)
    eq = Equation(1, lambda x: np.sin(x), lambda x,t: 0)
    conf = Config(mesh, eq)
    print(f'dt = {conf.mesh.dt}')
    solver = ExplicitUpwind(conf)
    U = solver.solve()
    for i in [0, 8, 16, 24]:
        plt.plot(U[:,i], alpha=0.5, label=f'$t={i*mesh.dt}$')
    plt.legend(loc='upper right')
    from kjetilplot import savePlot
    savePlot('upwind.pdf')
    plt.show()




