from train import *
from scipy.integrate import simpson
from utils.flux import Mesh


def plot(tt,xx,preds, config, history, model):
    # tt, xx, preds = evaluate(config, model)

    fig = plt.figure(figsize=(5,5))
    left = fig.add_subplot(111, projection='3d') # type: ignore
    left.plot_surface(xx, tt, preds.reshape(xx.shape))
    left.set_xlabel('$x$')
    left.set_ylabel('$t$')
    left.view_init(elev=20, azim=-130)
    
    # Plot of path
    def qhat(t,x):
        tbroad = t.expand(x.shape)
        inp = torch.column_stack((tbroad,x))
        return model(inp).flatten()
    pathconfig = config
    pathconfig.source = qhat
    path = get_dataset(pathconfig)
    tgrid = torch.linspace(0, pathconfig.mesh.T, pathconfig.mesh.N)
    tt, xx = torch.meshgrid(tgrid, pathconfig.mesh.centroids)

    fig3 = plt.figure(figsize=(5,5))
    ax = fig3.add_subplot(111, projection='3d') # type: ignore
    ax.plot_surface(xx, tt, path.detach().T)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    # ax.invert_xaxis()
    ax.view_init(elev=20, azim=-130)


    fig2 = plt.figure(figsize=(7,5))
    with plt.style.context('ggplot'): # type: ignore
        right = fig2.add_subplot(111) # type: ignore
        right.plot(history)
        right.set_yscale('log')

    plt.tight_layout()
    plt.show()

def evaluate(config, model, ref=50):
    with torch.no_grad():
        tgrid = torch.linspace(0, config.mesh.T, ref)
        xgrid = torch.linspace(-1,1,ref)
        tt,xx = torch.meshgrid(tgrid, xgrid)
        coords = torch.cartesian_prod(tgrid, xgrid)
        preds = model(coords)
    return tt,xx, preds

def main(M, cfl=.7):
    source = lambda t,x: np.sin(2*np.pi*t)*np.sin(2*np.pi*x)
    a = 1
    T = 1
    cfl = .7
    # M = 100
    N = math.ceil(a*T*M/(2*cfl))
    dt = T / (N-1)
    faces = torch.linspace(-1,1,M+1, requires_grad=False)   # M+1 faces -> M centroids

    init = lambda x: torch.sin(2*torch.pi*x)
    mesh = flux.Mesh(faces, T, N)
    scheme = flux.LW(lambda u:a*u,dt,mesh.dx)
    print(f'INFO: Using N = {N}, dt = {dt:}')
    config = Config(init, source, mesh, scheme)
    
    dataset = get_dataset(config)
    # Possible pretraining here
    RUNS = 5
    refinement = 100
    tt, xx = None, None
    histories = []
    predictions = []
    model_list = []
    for i in range(RUNS):
        print(f'INFO: Beginning run {i+1}.')
        model = models.DenseNet()
        history = train(model, dataset, config)
        model_list.append(model)
        tt, xx, preds = evaluate(config, model, ref=refinement)
        predictions.append(preds)
        histories.append(torch.tensor(history,requires_grad=False))

    histories = torch.stack(histories)
    predictions = torch.stack(predictions)
    # Save model, do tests, etc.
    mean_hist = torch.mean(histories,0)
    mean_preds = torch.mean(predictions,0)
    
    tgrid = torch.linspace(0, config.mesh.T, refinement)
    xgrid = torch.linspace(-1,1,refinement)
    coords = torch.cartesian_prod(tgrid, xgrid)
    # Source error
    true = torch.zeros((coords.shape[0], 1))
    for row in range(len(true)):
        true[row,0] = source(coords[row,0], coords[row,1])
    mean_preds = mean_preds.reshape((len(tgrid),len(xgrid)))
    true = true.reshape((len(tgrid),len(xgrid)))
    z = (mean_preds-true)**2
    z = z.detach().numpy()
    integ = simpson([simpson(zt,xgrid.detach().numpy()) for zt in z], tgrid.detach().numpy())
    # Solution error
    def qhat(t,x):
        tbroad = t.expand(x.shape)
        inp = torch.column_stack((tbroad,x))
        outputs = torch.stack([model_list[i](inp).flatten() for i in range(len(model_list))])
        return torch.mean(outputs,0)
    cfl = .5
    M = 400
    N = math.ceil(a*T*M/(2*cfl))
    dt = T / (N-1)
    faces = torch.linspace(-1,1,M+1, requires_grad=False)   # M+1 faces -> M centroids
    mesh = flux.Mesh(faces, T, N)
    scheme = flux.LW(lambda u:a*u,dt,mesh.dx)
    print(f'PATHERROR_INFO: Using N = {N}, dt = {dt:}')
    config = Config(init, source, mesh, scheme)
    dataset = get_dataset(config)
    pathconfig = config
    pathconfig.source = qhat
    path = get_dataset(pathconfig)
    # print(path.shape, dataset.shape)
    # print(torch.mean((path-dataset)**2))
    zpath = (path-dataset)**2
    zpath = zpath.detach().numpy()
    tgrid_path = torch.linspace(0,pathconfig.mesh.T,pathconfig.mesh.N)
    path_integ = simpson([simpson(zx,tgrid_path) for zx in zpath], pathconfig.mesh.centroids)

    # print('Source error:', integ)
    # print('Path error:', path_integ)
    return integ, path_integ

if __name__ == '__main__':
    
    source, path = [], []
    Ms = [10, 50, 100, 150, 200, 250, 300]
    for M in Ms:
        print(f'\nRunning refinement case M = {M}.\n')
        source_err, path_err = main(M)
        source.append(source_err)
        path.append(path_err)
    
    fig1 = plt.figure(figsize=(7,5))
    fig2 = plt.figure(figsize=(7,5))
    with plt.style.context('ggplot'): # type: ignore
        ax1 = fig1.add_subplot(111) # type: ignore
        ax1.plot(Ms, source, 'x-')
        ax1.set_yscale('log')
        ax1.set_ylabel('$L^2$ error')
        ax1.set_xlabel('$M$')
        ax2 = fig2.add_subplot(111) # type: ignore
        ax2.plot(Ms, path, 'x-')
        ax2.set_yscale('log')
        ax2.set_ylabel('$L^2$ error')
        ax2.set_xlabel('$M$')
    
    plt.show()

    # plot(tt,xx, mean_preds, config, mean_hist)
    # plot(tt,xx, (mean_preds-true)**2, config, mean_hist)
    # tt, xx = torch.meshgrid(tgrid_path, config.mesh.centroids)
    # plot(tt, xx, zpath, config, mean_hist)
