# type: ignore
from sandbox import *

def animate():
    plt.style.use('ggplot')
    cfl = 0.4
    mesh = Mesh([-1,1], 200, cfl=cfl, tmax=3)
    eq = Equation(1, lambda x: np.sin(x), lambda x,t: 0)
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
    animate.leg = None
    def update(i):
        ln.set_data(xgrid, U[:,i])
        ln.set_label(f'$t={i*conf.dt:.2f}')
        if animate.leg is None:
            animate.leg = ax.legend(loc='upper right')
        # ax.legend(loc='upper right')
        legtext = animate.leg.get_texts()[0]
        legtext.set_text(f'$t={i*conf.dt:.2f}$')
        return ln, legtext
    
    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(plt.gcf(), update, frames=U.shape[1], interval=40, blit=True)
    from kjetilplot import savePlot
    # savePlot('upwind.pdf')
    plt.show()

if __name__ == '__main__':
    animate()