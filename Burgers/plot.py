from train import *
from utils.kjetilplot import savePlot
from pathlib import Path

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
    right.pcolormesh(xx,tt,config.source(tt,xx), cmap='viridis')
    right.set_xlabel('$x$')
    right.set_ylabel('$t$')

    fig = plt.figure(figsize=(5,5))
    left = fig.add_subplot(111) # type: ignore
    left.pcolormesh(xx,tt, V.T, cmap='viridis')
    left.set_xlabel('$x$')
    left.set_ylabel('$t$')
    return fig2, fig

def plot_preds_and_path(model: models.Ensemble, config: Config):
    ref = 100
    def qhat(t,x):
        tbroad = t.expand(x.shape)
        inp = torch.column_stack((tbroad,x))
        return model(inp).flatten()
    predconfig = config
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
    ax.pcolormesh(xx,tt, preds, cmap='viridis')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    # ax.invert_yaxis()
    # ax.view_init(elev=16, azim=-160)

    tgrid = torch.linspace(0, config.mesh.T, config.mesh.N, requires_grad=False)
    tt,xx = torch.meshgrid(tgrid, config.mesh.centroids)
    pathfig = plt.figure(figsize=(5,5))
    ax = pathfig.add_subplot(111) # type: ignore
    ax.pcolormesh(xx,tt, predpath.T, cmap='viridis')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    # ax.invert_yaxis()
    # ax.view_init(elev=16, azim=-160)
    return predfig, pathfig





def main():
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
    plt.show()
    inp = input('Save figures? ([y]/n) ')
    if inp in ['y', 'yes', '']:
        sourcefig.savefig(prepend_path('biharmonic_source.pdf'))
        pathfig.savefig(prepend_path('biharmonic_path_true.pdf'))
        predfig.savefig(prepend_path('biharmonic_preds.pdf'))
        uhatfig.savefig(prepend_path('biharmonic_path_hat.pdf'))


if __name__ == '__main__':
    main()
    