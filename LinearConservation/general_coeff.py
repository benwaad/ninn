import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tfk = tf.keras

plt.style.use('ggplot')

class Equation:
    def __init__(self, a_func, initial, source):
        '''Currently only periodic BC'''
        self.a_func = a_func
        self.initial = initial
        self.source = source

class Mesh:
    '''Stores information about a regular mesh, made for periodic BCs'''
    def __init__(self, equation: Equation, domain, M, dt, tmax=2):
        self.eq = equation
        self.domain = domain
        self.tmax = tmax
        self.M = M      # Number of points in computational domain
        self.N = tmax / dt
        self.dt = dt
        self.dx = (domain[1]-domain[0]) / M
        amax = np.max(abs(eq.a_func(*self.get_meshgrid())))
        self.cfl = amax * self.dt / self.dx
        # self.dt = cfl * self.dx / self.eq.a_func
    def get_xgrid(self):
        return np.linspace(self.domain[0], self.domain[1], self.M+1)[:-1]
    def get_tgrid(self):
        N = self.tmax / self.dt
        return np.linspace(0, self.tmax, int(N))
    def get_meshgrid(self):
        x = self.get_xgrid()
        t = self.get_tgrid()
        return np.meshgrid(t, x)
    @classmethod
    def from_cfl(cls, equation, domain, M, cfl=0.4, tmax=2):
        dx = (domain[1]-domain[0]) / M
        x = np.linspace(domain[0], domain[1], M+1)[:-1]
        t = np.linspace(0, tmax, M)
        t, x = np.meshgrid(t, x)
        amax = np.max(abs(equation.a_func(t, x)))
        dt = cfl * dx / amax
        return cls(equation, domain, M, dt, tmax=tmax)


### API up to now:
# eq = Equation( --- )
# mesh = Mesh(eq, --- )
# 
# Now we implement the solver

# -------------------- Solvers --------------------
class Solver:
    def __init__(self, mesh: Mesh):
        self.eq = mesh.eq
        self.mesh = mesh

    def solve(self):
        raise NotImplementedError

class ExplicitUpwind(Solver):
    def __init__(self, mesh: Mesh):
        super().__init__(mesh)
    
    def get_A(self, t):
        r = self.mesh.dt / self.mesh.dx
        M = self.mesh.M
        # x = self.mesh.get_xgrid()
        # t = self.mesh.get_tgrid()
        x = self.mesh.get_xgrid()
        a = self.eq.a_func(t, x)
        A = np.zeros((M,M))
        hi = np.ones(M-1) * r*(abs(a[:-1])-a[:-1])/2
        di = np.ones(M) * (-r*abs(a)+1)
        lo = np.ones(M-1) * r*(abs(a[1:])+a[:1])/2
        dslice = np.arange(len(A))
        A[dslice, dslice] = di
        A[dslice[:-1], dslice[1:]] = hi
        A[dslice[1:], dslice[:-1]] = lo
        # Relies on the fact that a is periodic as well
        if a[0] >= 0:
            A[0,-1] = r*a[-1]
        else:
            A[-1,0] = r*a[0]
        return A
    
    def solve(self):
        # A = self.get_A()
        M = self.mesh.M
        dt = self.mesh.dt
        tsteps = len(self.mesh.get_tgrid())
        x = self.mesh.get_xgrid()
        t = self.mesh.get_tgrid()

        U = np.zeros((M, tsteps))
        U[:,0] = self.eq.initial(x)

        for i in range(tsteps-1):
            A = self.get_A(t[i])
            q_vec = self.eq.source(t[i], x)
            U[:,i+1] = A @ U[:,i] + q_vec
        return U

# -------------------------------------------------



if __name__ == '__main__':
    newparams = {'figure.figsize': (8.0, 4.0), 'axes.grid': True,
             'lines.markersize': 8, 'lines.linewidth': 1.5,
             'font.size': 10}
    plt.rcParams.update(newparams)

    # a_func = lambda t,x: t*x/(t*x)
    def a_func(t, x):
        return 1 + np.sin(2*np.pi*x)
    a_func = np.vectorize(a_func)
    initial = lambda x: np.sin(2*np.pi*x)
    # source = lambda t,x: 0*t*x
    def source(x, t):
        return 0.
    source = np.vectorize(source)
    eq = Equation(a_func, initial, source)
    mesh = Mesh.from_cfl(eq, domain=[-1,1], M=100, cfl=0.3, tmax=1)
    solver = ExplicitUpwind(mesh)
    # t = mesh.get_tgrid()
    x = mesh.get_xgrid()
    t = mesh.get_tgrid()
    U = solver.solve()

    fig, ax = plt.subplots()
    for idx in [0, 10, 20]:
        ax.plot(x, U[:,idx], alpha=.7, label=f'$t={t[idx]:.2f}$')
        # ax.plot(x, U[:,40], alpha=.7, label=f'$t={t[10]:.2f}$')
        # ax.plot(x, U[:,80], alpha=.7, label=f'$t={t[20]:.2f}$')
    ax.legend()
    plt.show()


