from train import *
from scipy.integrate import simpson
from utils.flux import Mesh
from pathlib import Path
import copy

def get_img_path():
    cwd = Path(__file__).parent
    imgpath = Path.joinpath(cwd, 'img')
    return imgpath.resolve()
def prepend_path(name):
    path = get_img_path()
    total = Path.joinpath(path, name)
    return str(total.resolve())

def plot_setup(config: Config):
    M, N = config.mesh.M, config.mesh.N
    tgrid = torch.linspace(0,config.mesh.T, N, requires_grad=False)
    xgrid = config.mesh.centroids
    tt, xx = torch.meshgrid(tgrid, xgrid, indexing='ij')
    V = get_dataset(config)
    
    fig2 = plt.figure(figsize=(5,5))
    right = fig2.add_subplot(111) # type: ignore
    im = right.pcolormesh(xx,tt,config.source(tt,xx), cmap='viridis')
    right.set_xlabel('$x$')
    right.set_ylabel('$t$')
    fig2.colorbar(im)

    fig = plt.figure(figsize=(5,5))
    left = fig.add_subplot(111) # type: ignore
    im = left.pcolormesh(xx,tt, V.T, cmap='viridis')
    left.set_xlabel('$x$')
    left.set_ylabel('$t$')
    fig.colorbar(im)
    return fig2, fig

def plot_preds_and_path(model: models.Ensemble, config: Config):
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
    tt,xx = torch.meshgrid(tgrid, xgrid, indexing='ij')
    coords = torch.cartesian_prod(tgrid, xgrid)
    predfig = plt.figure(figsize=(5,5))
    with torch.no_grad():
        preds = model(coords).reshape(tt.shape)
    ax = predfig.add_subplot(111) # type: ignore
    im = ax.pcolormesh(xx,tt, preds, cmap='viridis')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    predfig.colorbar(im)

    tgrid = torch.linspace(0, config.mesh.T, config.mesh.N, requires_grad=False)
    tt,xx = torch.meshgrid(tgrid, config.mesh.centroids, indexing='ij')
    pathfig = plt.figure(figsize=(5,5))
    ax = pathfig.add_subplot(111) # type: ignore
    im = ax.pcolormesh(xx,tt, predpath.T, cmap='viridis')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    pathfig.colorbar(im)
    return predfig, pathfig

def plot_errors(model: models.Ensemble, config: Config):
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

def get_L2_errors(M):
    a = 1
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
    scheme = flux.LW(lambda u:a*u,dt,mesh.dx)
    config = Config(init, source, mesh, scheme)

    # dataset = get_dataset(config)
    # coords, target = make_pointwise(dataset, config)
    # model = models.DenseNet()
    # Possible pretraining here
    model, hist = get_trained_ensemble(config, n_models=5)
    
    
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
    cfl = .5
    N = math.ceil(a*T*M/(2*cfl))
    dt = T / (N-1)
    faces = torch.linspace(-1,1,M+1, requires_grad=False)   # M+1 faces -> M centroids
    mesh = flux.Mesh(faces, T, N)
    scheme = flux.LW(lambda u:a*u,dt,mesh.dx)
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

def refine():
    source, path = [], []
    # Ms = [10, 50, 100, 150, 200, 250, 300]
    Ms = [2**n for n in range(3, 10)]
    for M in Ms:
        print(f'\nRunning refinement case M = {M}.\n')
        source_err, path_err = get_L2_errors(M)
        source.append(source_err)
        path.append(path_err)
    
    fig1 = plt.figure(figsize=(7,5))
    fig2 = plt.figure(figsize=(7,5))
    with plt.style.context('ggplot'): # type: ignore
        ax1 = fig1.add_subplot(111) # type: ignore
        ax1.plot(Ms, source, 'x-')
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_ylabel('$L^2$ error')
        ax1.set_xlabel('$M$')
        ax2 = fig2.add_subplot(111) # type: ignore
        ax2.plot(Ms, path, 'x-')
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_ylabel('$L^2$ error')
        ax2.set_xlabel('$M$')
    return fig1, fig2

def main():
    inp = input('Perform refinement experiment? (costly, [y]/n) ')
    if inp in ['y', 'yes', '']:
        source, path = refine()
        plt.show()
        return

    a = 1
    T = 1
    cfl = .7
    M = 100
    N = int(a*T*M/(2*cfl))
    print(f'INFO: Using N = {N}.')
    dt = T / (N-1)
    faces = torch.linspace(-1,1,M, requires_grad=False)

    init = lambda x: torch.sin(2*torch.pi*x)
    source = lambda t,x: torch.sin(2*torch.pi*t)*torch.sin(2*torch.pi*x)
    mesh = flux.Mesh(faces, T, N)
    scheme = flux.LW(lambda u:a*u,dt,mesh.dx)
    config = Config(init, source, mesh, scheme)
    sourcefig, pathfig = plot_setup(config)
    path = get_img_path()
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
    if inp in ['y', 'yes', '']:
        plt.show()
    inp = input('Save figures? ([y]/n) ')
    if inp in ['y', 'yes', '']:
        sourcefig.savefig(prepend_path('biharmonic_source.pdf'))
        pathfig.savefig(prepend_path('biharmonic_path_true.pdf'))

        predfig.savefig(prepend_path('biharmonic_preds.pdf'))
        uhatfig.savefig(prepend_path('biharmonic_path_hat.pdf'))

        prederrfig.savefig(prepend_path('biharmonic_source_error.pdf'))
        uhaterrfig.savefig(prepend_path('biharmonic_path_error.pdf'))
        histfig.savefig(prepend_path('biharmonic_history.pdf'))



if __name__ == '__main__':
    main()
    
