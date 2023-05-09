from train import *
from utils.kjetilplot import savePlot
from pathlib import Path
import copy
from scipy.integrate import simpson
import math
from utils.tools import softwavefinder

def get_burgers_img_path():
    cwd = Path(__file__).parent
    imgpath = Path.joinpath(cwd, 'img')
    return imgpath.resolve()
def prepend_path(name):
    path = get_burgers_img_path()
    total = Path.joinpath(path, name)
    return str(total.resolve())


def plot_setup(config: Config):
    M, N = config.mesh.M, config.mesh.N
    tgrid = torch.linspace(0,config.mesh.T, N, requires_grad=False)
    xgrid = config.mesh.centroids
    tt, xx = torch.meshgrid(tgrid, xgrid)
    V = get_dataset(config)
    
    fig2 = plt.figure(figsize=(5,5))
    right = fig2.add_subplot(111) # type: ignore
    im = right.pcolormesh(xx,tt,config.source(tt,xx), cmap='viridis')
    fig2.colorbar(im)
    right.set_xlabel('$x$')
    right.set_ylabel('$t$')

    fig = plt.figure(figsize=(5,5))
    left = fig.add_subplot(111) # type: ignore
    im = left.pcolormesh(xx,tt, V.T, cmap='viridis')
    fig.colorbar(im)
    left.set_xlabel('$x$')
    left.set_ylabel('$t$')
    return fig2, fig

def plot_preds_and_path(model: models.Ensemble, config: Config, plot_points=False):
    ref = 100
    def qhat(t,x):
        tbroad = t.expand(x.shape)
        inp = torch.column_stack((tbroad,x))
        return model(inp).flatten()
    predconfig = copy.deepcopy(config)
    predconfig.source = qhat
    with torch.no_grad():
        predpath = get_dataset(predconfig)
    
    tgrid = torch.linspace(0, config.mesh.T, ref, requires_grad=False)
    xgrid = torch.linspace(-1,1,ref, requires_grad=False)
    tt,xx = torch.meshgrid(tgrid, xgrid)
    coords = torch.cartesian_prod(tgrid, xgrid)
    predfig = plt.figure(figsize=(5,5))
    with torch.no_grad():
        preds = model(coords).reshape(tt.shape)
    ax = predfig.add_subplot(111) # type: ignore
    im = ax.pcolormesh(xx,tt, preds, cmap='viridis')
    if plot_points:
        N_scatter = int(1.5*config.mesh.T*config.mesh.M/(2*.7)) # Errorprone, copies amax and cfl defined in main() (used to plot training points)
        tgrid_scatter = torch.linspace(0, config.mesh.T, N_scatter, requires_grad=False)
        scatter_coords = torch.cartesian_prod(tgrid_scatter[:-1], config.mesh.centroids)
        ax.scatter(scatter_coords[:,1], scatter_coords[:,0], s=10, c='red', marker='x', linewidths=.5) # type: ignore
    predfig.colorbar(im)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    # ax.invert_yaxis()
    # ax.view_init(elev=16, azim=-160)

    tgrid = torch.linspace(0, config.mesh.T, config.mesh.N, requires_grad=False)
    tt,xx = torch.meshgrid(tgrid, config.mesh.centroids)
    pathfig = plt.figure(figsize=(5,5))
    ax = pathfig.add_subplot(111) # type: ignore
    im = ax.pcolormesh(xx,tt, predpath.T, cmap='viridis')
    pathfig.colorbar(im)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    # ax.invert_yaxis()
    # ax.view_init(elev=16, azim=-160)
    return predfig, pathfig

def plot_errors(model: models.Ensemble, config: Config, plot_points=False):
    ref = 100
    def qhat(t,x):
        tbroad = t.expand(x.shape)
        inp = torch.column_stack((tbroad,x))
        return model(inp).flatten()
    predconfig = copy.deepcopy(config)
    predconfig.source = qhat
    truepath = get_dataset(config)
    with torch.no_grad():
        predpath = get_dataset(predconfig)
    
    tgrid = torch.linspace(0, config.mesh.T, ref, requires_grad=False)
    xgrid = torch.linspace(-1,1,ref, requires_grad=False)
    tt,xx = torch.meshgrid(tgrid, xgrid, indexing='ij')
    coords = torch.cartesian_prod(tgrid, xgrid)
    with torch.no_grad():
        true = config.source(tt,xx)
        preds = model(coords).reshape(tt.shape)
    prederr = (preds-true)**2
    prederr = prederr.detach().numpy()
    predint = simpson([simpson(zt, xgrid.detach().numpy()) for zt in prederr], tgrid.detach().numpy())
    print('Source error (L²): ', predint)
    
    predfig = plt.figure(figsize=(5,5))
    ax = predfig.add_subplot(111) # type: ignore
    im = ax.pcolormesh(xx,tt, prederr, cmap='viridis')
    if plot_points:
        N_scatter = int(1.5*config.mesh.T*config.mesh.M/(2*.7)) # Errorprone, copies amax and cfl defined in main() (used to plot training points)
        tgrid_scatter = torch.linspace(0, config.mesh.T, N_scatter, requires_grad=False)
        scatter_coords = torch.cartesian_prod(tgrid_scatter[:-1], config.mesh.centroids)
        ax.scatter(scatter_coords[:,1], scatter_coords[:,0], s=10, c='red', marker='x', linewidths=.5) # type: ignore
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    predfig.colorbar(im)

    tgrid = torch.linspace(0, config.mesh.T, config.mesh.N, requires_grad=False)
    tt,xx = torch.meshgrid(tgrid, config.mesh.centroids, indexing='ij')
    patherr = (truepath-predpath)**2
    patherr = patherr.detach().numpy()
    pathint = simpson([simpson(zx, tgrid.detach().numpy()) for zx in patherr], predconfig.mesh.centroids)
    print('Path error (L²): ', pathint)
    pathfig = plt.figure(figsize=(5,5))
    ax = pathfig.add_subplot(111) # type: ignore
    im = ax.pcolormesh(xx,tt, (predpath.T-truepath.T)**2, cmap='viridis')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    pathfig.colorbar(im)
    return predfig, pathfig

def plot_history(hist):
    fig2 = plt.figure(figsize=(7,5))
    with plt.style.context('ggplot'): # type: ignore
        right = fig2.add_subplot(111) # type: ignore
        right.plot(hist)
        right.set_yscale('log')
    return fig2

def get_L2_errors(M, epochs):
    a = 1.5
    T = 1
    cfl = .7
    N = int(a*T*M/(2*cfl))
    ref = 500
    print(f'INFO: Using N = {N}.')
    dt = T / (N-1)
    faces = torch.linspace(-1,1,M, requires_grad=False)

    init = lambda x: torch.sin(2*torch.pi*x)
    source = lambda t,x: torch.sin(2*torch.pi*t)*torch.sin(2*torch.pi*x)
    mesh = flux.Mesh(faces, T, N)
    scheme = flux.Godunov(lambda u:u**2/2,dt,mesh.dx)
    config = Config(init, source, mesh, scheme)

    # dataset = get_dataset(config)
    # coords, target = make_pointwise(dataset, config)
    # model = models.DenseNet()
    # Possible pretraining here
    model, hist = get_trained_ensemble(config, n_models=5, epochs=epochs)
    
    
    tgrid = torch.linspace(0, config.mesh.T, ref, requires_grad=False)
    xgrid = torch.linspace(-1,1,ref, requires_grad=False)
    tt,xx = torch.meshgrid(tgrid, xgrid, indexing='ij')
    coords = torch.cartesian_prod(tgrid, xgrid)
    with torch.no_grad():
        true = config.source(tt,xx)
        preds = model(coords).reshape(tt.shape)
    # Prediction error
    prederr = (true-preds)**2
    prederr = prederr.detach().numpy()
    predint = simpson([simpson(zt, xgrid.detach().numpy()) for zt in prederr], tgrid.detach().numpy())
    # Solution error
    M_path = 200
    cfl = .5
    N = math.ceil(a*T*M_path/(2*cfl))
    dt = T / (N-1)
    faces = torch.linspace(-1,1,M_path+1, requires_grad=False)   # M+1 faces -> M centroids
    mesh = flux.Mesh(faces, T, N)
    scheme = flux.Godunov(lambda u:u**2/2,dt,mesh.dx)
    config = Config(init, source, mesh, scheme)
    def qhat(t,x):
        tbroad = t.expand(x.shape)
        inp = torch.column_stack((tbroad,x))
        return model(inp).flatten()
    predconfig = copy.deepcopy(config)
    predconfig.source = qhat
    truepath = get_dataset(config)
    with torch.no_grad():
        predpath = get_dataset(predconfig)
    tgrid_path = torch.linspace(0,predconfig.mesh.T,predconfig.mesh.N)
    patherr = (truepath-predpath)**2
    patherr = patherr.detach().numpy()
    pathint = simpson([simpson(zx, tgrid_path.detach().numpy()) for zx in patherr], predconfig.mesh.centroids)

    return predint, pathint

def get_true_norms():
    a = 1.5
    T = 1
    cfl = .7
    M = 500
    N = int(a*T*M/(2*cfl))
    print(f'INFO: Using N = {N}.')
    dt = T / (N-1)
    faces = torch.linspace(-1,1,M, requires_grad=False)

    init = lambda x: torch.sin(2*torch.pi*x)
    source = lambda t,x: torch.sin(2*torch.pi*t)*torch.sin(2*torch.pi*x)
    mesh = flux.Mesh(faces, T, N)
    scheme = flux.Godunov(lambda u:u**2/2,dt,mesh.dx)
    config = Config(init, source, mesh, scheme)

    tgrid = torch.linspace(0, config.mesh.T, M, requires_grad=False)
    xgrid = torch.linspace(-1,1,M, requires_grad=False)
    tt,xx = torch.meshgrid(tgrid, xgrid, indexing='ij')
    true = config.source(tt,xx)**2
    trueint = simpson([simpson(zt, xgrid.detach().numpy()) for zt in true], tgrid.detach().numpy())

    cfl = .5
    N = math.ceil(a*T*M/(2*cfl))
    dt = T / (N-1)
    faces = torch.linspace(-1,1,M+1, requires_grad=False)   # M+1 faces -> M centroids
    mesh = flux.Mesh(faces, T, N)
    scheme = flux.Godunov(lambda u:u**2/2,dt,mesh.dx)
    config = Config(init, source, mesh, scheme)
    truepath = get_dataset(config)**2
    tgrid_path = torch.linspace(0,config.mesh.T,config.mesh.N)
    pathint = simpson([simpson(zx, tgrid_path.detach().numpy()) for zx in truepath.detach().numpy()], config.mesh.centroids)
    return trueint, pathint


    

def refine():
    source_int, path_int = get_true_norms()
    source, path = [], []
    # Ms = [10, 50, 100, 150, 200, 250, 300]
    # Compute reference solution
    
    Ms = [2**n for n in range(3, 10)]
    epochs = np.flip(np.linspace(8, 250, len(Ms), dtype=np.int32))
    for M, ep in zip(Ms, epochs):
        print(f'\nRunning refinement case M = {M}.\n')
        source_err, path_err = get_L2_errors(M, ep)
        source.append(source_err/source_int)
        path.append(path_err/path_int)
    
    fig1 = plt.figure(figsize=(7,5))
    fig2 = plt.figure(figsize=(7,5))
    with plt.style.context('ggplot'): # type: ignore
        ax1 = fig1.add_subplot(111) # type: ignore
        ax1.plot(Ms, source, 'x-')
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_ylabel('Relative $L^2$ error')
        ax1.set_xlabel('$M$')
        ax2 = fig2.add_subplot(111) # type: ignore
        ax2.plot(Ms, path, 'x-')
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_ylabel('Relative $L^2$ error')
        ax2.set_xlabel('$M$')
    return fig1, fig2



def main():
    inp = input('Perform refinement experiment? (costly, [y]/n) ')
    if inp in ['y', 'yes']:
        source, path = refine()
        plt.show()
        return
    
    amax = 1.5
    T = 1
    cfl = .7
    M = 100
    N = int(amax*T*M/(2*cfl))
    print(f'INFO: Using N = {N}.')
    dt = T / (N-1)
    faces = torch.linspace(-1,1,M, requires_grad=False)

    init = lambda x: -torch.sin(torch.pi*x)
    source = lambda t,x: torch.sin(2*torch.pi*t)*torch.sin(2*torch.pi*x)
    mesh = flux.Mesh(faces, T, N)
    scheme = flux.Godunov(lambda u:u**2/2,dt,mesh.dx)
    config = Config(init, source, mesh, scheme)
    sourcefig, pathfig = plot_setup(config)
    path = get_burgers_img_path()
    print('INFO: Image root is', path)

    
    # dataset = get_dataset(config)
    # coords, target = make_pointwise(dataset, config)
    # model = models.DenseNet()
    # Possible pretraining here
    model, hist = get_trained_ensemble(config, n_models=5)
    predfig, uhatfig = plot_preds_and_path(model, config)
    prederrfig, uhaterrfig = plot_errors(model, config)
    histfig = plot_history(hist)

    inp = input('Show figures? ([y]/n) ')
    if inp in ['y', 'yes']:
        plt.show()
    inp = input('Save figures? ([y]/n) ')
    if inp in ['y', 'yes']:
        sourcefig.savefig(prepend_path('biharmonic_source.pdf'))
        pathfig.savefig(prepend_path('biharmonic_path_true.pdf'))

        predfig.savefig(prepend_path('biharmonic_preds.pdf'))
        uhatfig.savefig(prepend_path('biharmonic_path_hat.pdf'))

        prederrfig.savefig(prepend_path('biharmonic_source_error.pdf'))
        uhaterrfig.savefig(prepend_path('biharmonic_path_error.pdf'))
        histfig.savefig(prepend_path('biharmonic_history.pdf'))
    # plt.show()
    # inp = input('Save figures? ([y]/n) ')
    # if inp in ['y', 'yes', '']:
    #     sourcefig.savefig(prepend_path('biharmonic_source.pdf'))
    #     pathfig.savefig(prepend_path('biharmonic_path_true.pdf'))
    #     predfig.savefig(prepend_path('biharmonic_preds.pdf'))
    #     uhatfig.savefig(prepend_path('biharmonic_path_hat.pdf'))


#############################################################################
######################### Solution dependent source #########################
#############################################################################

def plot_setup_soldep(config: Config):
    M, N = config.mesh.M, config.mesh.N
    tgrid = torch.linspace(0,config.mesh.T, N, requires_grad=False)
    xgrid = config.mesh.centroids
    tt, xx = torch.meshgrid(tgrid, xgrid)
    V = get_dataset_soldep(config)
    
    ugrid = torch.linspace(-2,2,200, requires_grad=False)
    fig2 = plt.figure(figsize=(5,5))
    with plt.style.context('ggplot'): # type: ignore
        right = fig2.add_subplot(111) # type: ignore
        right.plot(ugrid, config.source(0, 0, ugrid))
        right.set_xlabel('$u$')
    # right.set_ylabel('$t$')

    fig = plt.figure(figsize=(5,5))
    left = fig.add_subplot(111) # type: ignore
    im = left.pcolormesh(xx,tt, V.T, cmap='viridis')
    fig.colorbar(im)
    left.set_xlabel('$x$')
    left.set_ylabel('$t$')
    return fig2, fig

def plot_preds_and_path_soldep(model: models.Ensemble, config: Config):
    ref = 100
    def qhat(t,x,u):
        tbroad = t.expand(x.shape)
        inp = torch.column_stack((tbroad,x,u))
        return model(inp).flatten()
    predconfig = copy.deepcopy(config)
    predconfig.source = qhat
    with torch.no_grad():
        predpath = get_dataset_soldep(predconfig)
    
    # tgrid = torch.linspace(0, config.mesh.T, ref, requires_grad=False)
    # xgrid = torch.linspace(-1,1,ref, requires_grad=False)
    # tt,xx = torch.meshgrid(tgrid, xgrid)
    # coords = torch.cartesian_prod(tgrid, xgrid)
    ugrid = torch.linspace(-2,2,300)
    # zero = torch.zeros_like(ugrid)
    one = torch.ones_like(ugrid)
    # coords = torch.column_stack((zero, zero, ugrid))
    tvals_plot = [0.0, 0.5, 1.0]
    xvals_plot = [-1., 0., 1.]
    args = {t: [] for t in tvals_plot}
    for t in args.keys():
        args[t] = [torch.column_stack((t*one, xvals_plot[k]*one, ugrid)) for k in range(len(xvals_plot))]
        # args[t].append(torch.column_stack((t*one, xvals_plot[0]*one, ugrid)))
        # args[t].append(torch.column_stack((t*one, xvals_plot[1]*one, ugrid)))
        # args[t].append(torch.column_stack((t*one, xvals_plot[2]*one, ugrid)))
    # preds = {t: [] for t in tvals_plot}
    with plt.style.context('ggplot'): # type: ignore
        # predfig = plt.figure(figsize=(5,5))
        # ax = predfig.add_subplot(111) # type: ignore
        predfig, axs = plt.subplots(1, 3, figsize=(12, 5))
        for t, ax in zip(tvals_plot,axs):
            with torch.no_grad():
                arglist = args[t]
                preds = [model(arglist[k]).reshape(ugrid.shape) for k in range(len(xvals_plot))]
                # predl = model(arglist[0]).reshape(ugrid.shape)
                # predc = model(arglist[1]).reshape(ugrid.shape)
                # predr = model(arglist[2]).reshape(ugrid.shape)
            for x, pred in zip(xvals_plot,preds):
                ax.plot(ugrid.detach(), pred.detach(), label=f'$x={x}$')
            ax.plot(ugrid.detach(), config.source(0,0,ugrid.detach()), '--', label='True')
            ax.set_title(f'$t={t}$')
            ax.set_xlabel('$u$')
            ax.legend(loc='lower left')
        # ax.set_ylabel('$t$')
    # ax.invert_yaxis()
    # ax.view_init(elev=16, azim=-160)

    V = get_dataset_soldep(config)
    wavefront = [softwavefinder(u,config.mesh.centroids) for u in V.T]
    tgrid = torch.linspace(0, config.mesh.T, config.mesh.N, requires_grad=False)
    # Crazy oneliner to find the timegrid index of the shock (t=1/pi)
    # Used to plot the true shock line in the path plot
    shock_index = (tgrid>1/torch.pi).nonzero()[0,0]
    tt,xx = torch.meshgrid(tgrid, config.mesh.centroids)
    pathfig = plt.figure(figsize=(5,5))
    ax = pathfig.add_subplot(111) # type: ignore
    im = ax.pcolormesh(xx,tt, predpath.T, cmap='viridis')
    ax.plot(wavefront[shock_index:], tgrid[shock_index:], color='black')
    pathfig.colorbar(im)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    # ax.invert_yaxis()
    # ax.view_init(elev=16, azim=-160)
    return predfig, pathfig

def main_soldep():
    amax = 1.5
    T = 1
    cfl = .7
    M = 100
    N = int(amax*T*M/(2*cfl))
    print(f'INFO: Using N = {N}.')
    dt = T / (N-1)
    faces = torch.linspace(-1,1,M, requires_grad=False)

    init = lambda x: -torch.sin(torch.pi*x)
    source = lambda t,x,u: torch.sqrt(torch.abs(u))
    mesh = flux.Mesh(faces, T, N)
    scheme = flux.Godunov(lambda u:u**2/2,dt,mesh.dx)
    config = Config(init, source, mesh, scheme)
    sourcefig, pathfig = plot_setup_soldep(config)
    path = get_burgers_img_path()
    print('INFO: Image root is', path)

    model, hist = get_trained_ensemble_soldep(config, n_models=5)
    predfig, uhatfig = plot_preds_and_path_soldep(model, config)
    # prederrfig, uhaterrfig = plot_errors(model, config)
    histfig = plot_history(hist)
    plt.show()


if __name__ == '__main__':
    main()