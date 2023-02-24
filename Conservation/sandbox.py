import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tfk = tf.keras

plt.style.use('ggplot')

class Mesh:
    def __init__(self, domain, M, cfl=0.4, tmax=2):
        self.domain = domain
        self.M = M      # Number of points in computational domain
        self.cfl = cfl
        self.tmax = tmax

        self.dx = (domain[1]-domain[0]) / M
    def get_xgrid(self):
        return np.linspace(self.domain[0], self.domain[1], self.M)
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
        self.dt = cfl * self.mesh.dx / self.equation.a_func

# -------------- SOLVERS --------------

class Solver:
    def __init__(self, config: Config):
        self.mesh = config.mesh
        self.equation = config.equation
        self.dt = config.dt

    def solve(self):
        raise NotImplementedError

class ExplicitUpwind(Solver):
    def __init(self, config):
        super().__init__(config)
    
    def get_A(self, M, r):
        # First implement for constant a>0 (a_func is a positive number)
        a = self.equation.a_func
        A = np.zeros((M,M))
        hi = np.ones(M-1) * r*(abs(a)-a)/2
        di = np.ones(M) * (-r*abs(a)+1)
        lo = np.ones(M-1) * r*(abs(a)+a)/2
        # di = np.ones(M)   * (1-r*a)
        # lo = np.ones(M-1) * (r*a)

        dslice = np.arange(len(A))
        A[dslice, dslice] = di
        A[dslice[:-1], dslice[1:]] = hi
        A[dslice[1:], dslice[:-1]] = lo
        A[0,-1] = r*a
        return A
    
    def solve(self):

        dom = self.mesh.domain
        dx = self.mesh.dx
        dt = self.dt
        M = self.mesh.M
        tsteps = int(self.mesh.tmax/dt)
        # t = np.linspace(0, self.mesh.tmax, tsteps)
        A = self.get_A(M, dt/dx)
        
        # Solution container
        U = np.zeros((M, tsteps))
        x = np.linspace(dom[0], dom[1]-dx, M)
        U[:,0] = self.equation.initial(x)
        
        # Source
        # q_func = lambda t,x: np.where(abs(x)<.2, 1/(10+t) * np.sin(10*np.pi*t), 0)
        q_func = lambda t,x: np.where(abs(x)<.2, np.sign(x)*.08*np.sin(8*np.pi*t), 0)
        # Solution loop
        for i in range(tsteps-1):
            q_vec = q_func(i*dt,x)
            U[:,i+1] = A @ U[:,i] + q_vec
        
        return U

class Approximator(tfk.Model):
    def __init__(self, structure):
        """A Neural Network model for approximating a function.

        Args:
            structure (iterable): Iterable describing the structure of the network.
                                  If structure=[1,10,10,1], the network will have
                                  two hidden layers, each with 10 nodes, along with a
                                  one-dimensional input and output.
        """
        super().__init__(name='Approximator')
        self.num_hidden = len(structure) - 1
        init = tfk.initializers.HeUniform(seed=42)
        # self.inputs = tfk.Input(shape=(structure[0],))
        self.hidden = [
            tfkl.Dense(structure[i+1], kernel_initializer=init) # type:ignore
            for i in range(self.num_hidden)
        ]
    
    def call(self, inputs):
        x = inputs
        for i in range(self.num_hidden-1):
            x = self.hidden[i](x)
            x = tfk.activations.relu(x, alpha=0.3)
        return self.hidden[-1](x)

class Measurements:
    def __init__(self, config: Config):
        self.config = config
        self.mesh = config.mesh
    @classmethod
    def from_anal(cls, u, config):
        
        return u(config)


def interface():
    # How do i want to design the interface?
    
    # Create config (mesh and equation)
    # Generate data points based on config
    # Scale data and __train approximator on it__
    # Generate 'predictions' based on trained approximator

    cfl = 0.4
    mesh = Mesh([-1,1], 200, cfl=cfl, tmax=3)
    eq = Equation(1, lambda x: np.sin(2*np.pi*x), lambda x,t: 0)
    conf = Config(mesh, eq)







if __name__ == '__main__':
    cfl = 0.4
    mesh = Mesh([-1,1], 200, cfl=cfl, tmax=3)
    eq = Equation(1, lambda x: np.sin(2*np.pi*x), lambda x,t: 0)
    conf = Config(mesh, eq)
    print(f'dt = {conf.dt}')
    solver = ExplicitUpwind(conf)
    U = solver.solve()
    
    xgrid = mesh.get_xgrid()
    fig = plt.figure()
    ax = plt.axes(xlim=mesh.domain, ylim=(-2.5,2.5))
    ln, = ax.plot([],[])
    ax.vlines([-.2,.2], -2.5, 2.5)
    ax.fill_between(xgrid, -2.5, 2.5, where=abs(xgrid)<.2, alpha=.3)
    leg = None
    def update(i):
        ln.set_data(xgrid, U[:,i])
        ln.set_label(f'$t={i*conf.dt:.2f}')
        global leg
        if leg is None:
            leg = ax.legend(loc='upper right')
        # ax.legend(loc='upper right')
        legtext = leg.get_texts()[0]
        legtext.set_text(f'$t={i*conf.dt:.2f}$')
        return ln, legtext
    
    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(plt.gcf(), update, frames=U.shape[1], interval=40, blit=True)
    # from kjetilplot import savePlot
    # savePlot('upwind.pdf')
    plt.show()


