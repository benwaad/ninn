import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
import time
tfk = tf.keras

# dx = 0.01
# dt = 0.001
amax = 1.5
T = .5
cfl = .7
M = 100
N = int(amax*T*M/(2*cfl))

F = lambda u: u**2/2
# init = lambda x: -np.sin(np.pi*x)
init = lambda x: np.where(x<0, 1., 0.)
# q = lambda p: 2*np.exp(-20*(p[1]-.5)**2) * np.cos(2*np.pi*p[0])
# q = lambda p: np.where(abs(p[1]+.5)<.2, 0, 0) * np.cos(2*np.pi*p[0])
q = lambda p: .5*np.exp(-10*(p[1]+.5)**2)# * np.cos(2*np.pi*p[0])
# q = lambda p: p[0]*0

def get_model():
    tfkl = tfk.layers
    inp = tfkl.Input(shape=[2])
    h1 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(inp)
    h2 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(h1)
    h3 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(h2)
    out = tfkl.Dense(1, activation='linear')(h3)
    model = tfk.Model(inputs=inp, outputs=out)
    return model

def LW_step(u_w, u_p, u_e, q_p, dt, dx):
    r = dt/dx
    F_w, F_p, F_e = F(u_w), F(u_p), F(u_e)
    u_iph = (u_p+u_e)/2
    u_imh = (u_p+u_w)/2
    tmp = u_p - r/2*(F_e-F_w)
    nosource = tmp + r/2*(u_iph*(F_e-F_p) - u_imh*(F_p-F_w))
    return nosource + dt*q_p

def solve(bdr='periodic'):
    # def q(p):
    #     return 10*np.exp(-50*(p[1]+.5)**2)
    xgrid = np.linspace(-1,1,M+1)[:-1]
    tgrid = np.linspace(0,T,N)
    dx = xgrid[1] - xgrid[0]
    dt = tgrid[1] - tgrid[0]
    
    V = np.zeros((len(xgrid), len(tgrid)))
    V[:,0] = init(xgrid)
    if bdr=='periodic':
        print('Using periodic bdr')

    for n, t in enumerate(tgrid[:-1]):
        for i, x in enumerate(xgrid[1:-1], start=1):
            q_p = q((t, x))
            V[i,n+1] = LW_step(V[i-1,n], V[i,n], V[i+1,n], q_p, dt, dx)
        if bdr=='periodic':
            V[0,n+1] = LW_step(V[-1,n], V[0,n], V[1,n], q((t,-1)), dt, dx)
            V[-1,n+1] = LW_step(V[-2,n], V[-1,n], V[0,n], q((t,1-dx)), dt, dx)
        else:
            V[0,n+1] = LW_step(V[0,n], V[0,n], V[1,n], q((t,-1)), dt, dx)
            V[-1,n+1] = LW_step(V[-2,n], V[-1,n], V[-1,n], q((t,1-dx)), dt, dx)
    return tgrid, xgrid, V
# region basic
def test_solve(model_dir, hist_dir):
    tgrid, xgrid, V = solve()
    # utils.animate('Conservation/custom_test.mp4', tgrid, xgrid, V)
    # tt, xx = np.meshgrid(tgrid, xgrid)
    # tt, xx = tt.flatten(), xx.flatten()
    # coords = np.column_stack([tt, xx])
    dx = xgrid[1] - xgrid[0]
    dt = tgrid[1] - tgrid[0]

    # We now build X and y
    X = np.zeros(((N-1)*M, 2), dtype='float32')
    y = np.zeros(((N-1)*M, 4), dtype='float32')  # [N, W, P, E]
    # print(f'dx = {dx}')
    for n, t in enumerate(tgrid[:-1]):
        for i, x in enumerate(xgrid):
            row = n*M + i
            X[row,:] = t, x
            i_w = -1 if i==0 else i-1
            i_e = 0 if i==M-1 else i+1
            y[row,:] = V[i,n+1], V[i_w,n], V[i,n], V[i_e,n]
            # print(f'({i_w:2d}, {i:2d}, {i_e:2d}) --> ({xgrid[i_w]:.2f}, {xgrid[i]:.2f}, {xgrid[i_e]:.2f})')
    

    model = get_model()
    opt = tfk.optimizers.legacy.Adam(0.001)
    dt = tf.constant(dt, dtype='float32')
    dx = tf.constant(dx, dtype='float32')
    def loss_fnc(target, preds):
        N, W, P, E = tf.unstack(target, axis=1)
        N_hat = LW_step(W, P, E, preds, dt, dx)
        return tfk.losses.mse(N, N_hat)
    loss_avg = tf.keras.metrics.Mean()
    model.compile(loss=loss_fnc, optimizer=opt)

    EPOCHS = 50
    BATCH_SIZE = 32
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(BATCH_SIZE)
    history = []
    start = time.time()
    for epoch in range(EPOCHS):
        loss_avg.reset_states()
        for p, target in dataset:
            with tf.GradientTape() as tape:
                preds = model(p)
                loss = loss_fnc(target, preds)
            grad = tape.gradient(loss, model.trainable_weights)
            opt.apply_gradients(zip(grad, model.trainable_variables))
            loss_avg.update_state(loss)
        history.append(loss_avg.result())
        if (epoch+1)%10 == 0:
            tf.print(f'Epoch {epoch+1:2d}, loss = {loss_avg.result():5.2e}, elapsed = {time.time()-start:5.1f}')
        loss_avg.reset_states()
    #np.save(hist_dir, np.array(history),allow_pickle=True)
    #model.save(model_dir)

    # Integrates the results along certain time slices
    q_hat = lambda p: model(np.array([[p[0], p[1]]])).numpy().flatten()  # type: ignore
    times, ints = integrate(q_hat)
    print('Time slices: ', times)
    print('Integrals  : ', ints)

    xgrid_test = np.linspace(-1,1,31)[:-1]
    tgrid_test = np.linspace(0,T,30)
    xgrid_test, tgrid_test = np.meshgrid(xgrid_test, tgrid_test)
    xgrid_test, tgrid_test = xgrid_test.flatten(), tgrid_test.flatten()
    X_test = np.column_stack([tgrid_test, xgrid_test])
    preds = model.predict(X_test).flatten()
    # fig, (ax,ax2) = plt.subplots(1, 2, projection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.plot_trisurf(X_test[:,0], X_test[:,1], preds) # type:ignore
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    ax2.plot_trisurf(X_test[:,0], X_test[:,1], np.apply_along_axis(q,1,X_test)) # type:ignore
    ax2.set_xlabel('$t$')
    ax2.set_ylabel('$x$')
    plt.show()


def integrate(q_hat ):
    from scipy import integrate
    ints = []
    times = [0, .05, .1, .15, .2]
    x = np.linspace(-1,1,100)
    for ti in times:
        t = np.ones(len(x))*ti
        samples = np.apply_along_axis(q_hat, 1, np.column_stack([t,x])).flatten()
        ints.append(integrate.simpson(samples, x))
    return times, ints


def check():
    # def q(p):
    #     return 10*np.exp(-50*(p[1]+.5)**2)
    
    M = 100
    x = np.linspace(-1,1,M)
    dx = x[1] - x[0]
    
    T = 0.1
    N = int(T / (.2*dx/3))
    t = np.linspace(0,T,N)

    t, x = np.meshgrid(t, x)
    t, x = t.flatten(), x.flatten()
    coords = np.column_stack([t,x])
    q_true = np.apply_along_axis(q, axis=1, arr=coords)

    model = get_model()
    opt = tfk.optimizers.legacy.Adam(0.001)
    loss_fnc = tfk.losses.mse
    loss_avg = tf.keras.metrics.Mean()
    model.compile(loss=loss_fnc, optimizer=opt)

    EPOCHS = 50
    BATCH_SIZE = 64
    dataset = tf.data.Dataset.from_tensor_slices((coords, q_true))
    dataset = dataset.batch(BATCH_SIZE)
    history = []
    
    for epoch in range(EPOCHS):
        loss_avg.reset_states()
        for p, target in dataset:
            with tf.GradientTape() as tape:
                preds = model(p)
                loss = loss_fnc(target, preds)
            grad = tape.gradient(loss, model.trainable_weights)
            opt.apply_gradients(zip(grad, model.trainable_variables))
            loss_avg.update_state(loss)
        history.append(loss_avg.result())
        if (epoch+1)%10 == 0:
            tf.print(f'Epoch {epoch+1}, loss = {loss_avg.result():5.2e}')
        loss_avg.reset_states()

    
    preds = model.predict(coords).flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(coords[:,0], coords[:,1], preds) # type:ignore
    plt.show()
# endregion

# ----------------------------------------------------------------
# ---------------------- WAVEFRONT TRAINING ----------------------
# ----------------------------------------------------------------

def softargmax(x, xgrid, beta=1e10):
    # x = tf.convert_to_tensor(x)
    # x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
    return tf.reduce_sum(tf.nn.softmax(x*beta) * xgrid, axis=-1)

def softwavefinder(u, xgrid):
    _u = tf.squeeze(u)
    dx = xgrid[1]-xgrid[0]
    padded = tf.pad(_u, [[1,1]], mode='symmetric')
    ddu = (padded[2:]-2*padded[1:-1]+padded[:-2])/(dx**2)
    return softargmax(ddu, xgrid)

def train_against_waterfront():
    tgrid, xgrid, V = solve(bdr='cont')
    dt = tgrid[1] - tgrid[0]
    dx = xgrid[1] - xgrid[0]
    # Convert V to wavefront data
    threshold = .05
    frontfinder = lambda v: xgrid[np.argwhere(v<threshold)[0,0]]
    fronts = np.apply_along_axis(frontfinder, axis=0, arr=V)
    iters = np.arange(len(fronts))
    # We have to assume that only the fronts are given
    # Then, based on this and the number of iterations we should run
    # to get there (equiv to giving time points), we iterate network from init
    initial = init(xgrid)
    model = get_model()
    opt = tfk.optimizers.legacy.Adam(0.001)
    loss_fnc = tfk.losses.mse
    loss_avg = tfk.metrics.Mean()
    EPOCHS = 15
    BATCH_SIZE = 32
    dataset = tf.data.Dataset.from_tensor_slices((iters, fronts))
    # dataset = dataset.batch(BATCH_SIZE)
    pull = lambda x: tf.expand_dims(x,0)
    tol = 1e-12
    acc_counter = 0
    iter_counter = 0
    history = []
    start = time.time()
    for epoch in range(EPOCHS):
        loss_avg.reset_states()
        for it, front in dataset:
            u_p = initial.copy()
            with tf.GradientTape(persistent=True) as tape:
                for i in range(it):
                    u_w = np.roll(u_p, 1)
                    u_e = np.roll(u_p, -1)
                    u_w[0] = u_w[1]
                    u_e[-1] = u_e[-2]
                    tvec = np.repeat(tgrid[i], M)
                    qvec = model(np.column_stack((tvec,xgrid)))
                    u_p = LW_step(u_w,u_p,u_e,tf.squeeze(qvec),dt,dx)
                front_pred = softwavefinder(u_p, xgrid)
                wavefront_loss = loss_fnc(pull(front), pull(front_pred))
            grad = tape.gradient(wavefront_loss,model.trainable_weights)
            iter_counter += 1
            if wavefront_loss > tol:
                opt.apply_gradients(zip(grad, model.trainable_variables))
                acc_counter += 1
                loss_avg.update_state(wavefront_loss)
        if (epoch+1)%5==0:
            tf.print(f'Epoch {epoch+1:2d}, loss = {loss_avg.result():5.2e}, elapsed = {time.time()-start:6.2f}')
        loss_avg.reset_states()
    
    tf.print(f'Accepted {acc_counter} out of {iter_counter} steps ({int(acc_counter/iter_counter*100):2d}%)\n')
    xgrid_test = np.linspace(-1,1,31)[:-1]
    tgrid_test = np.linspace(0,T,30)
    xx, tt = np.meshgrid(xgrid_test, tgrid_test)
    xx, tt = xx.flatten(), tt.flatten()
    X_test = np.column_stack([tt,xx])
    preds = model.predict(X_test).flatten()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.plot_trisurf(X_test[:,0], X_test[:,1], preds) # type:ignore
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.view_init(elev=17, azim=-7) # type: ignore
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    ax2.plot_trisurf(X_test[:,0], X_test[:,1], np.apply_along_axis(q,1,X_test)) # type:ignore
    ax2.set_xlabel('$t$')
    ax2.set_ylabel('$x$')
    ax2.view_init(elev=17, azim=-7) # type: ignore

    # Construct wave realization
    tf.print('Constructing wave realizations')
    qhat = lambda p: model(np.array([[p[0], p[1]]])).numpy().flatten() # type:ignore
    dt = tgrid_test[1] - tgrid_test[0]
    dx = xgrid_test[1] - xgrid_test[0]
    truewave = np.zeros((len(xgrid_test), len(tgrid_test)))
    truewave[:,0] = init(xgrid_test)
    predwave = truewave.copy()
    for n, t in enumerate(tgrid_test[:-1]):
        for i, x in enumerate(xgrid_test[1:-1], start=1):
            q_p = q((t, x))
            qhat_p = qhat((t,x))
            truewave[i,n+1] = LW_step(truewave[i-1,n], truewave[i,n], truewave[i+1,n], q_p, dt, dx)
            predwave[i,n+1] = LW_step(predwave[i-1,n], predwave[i,n], predwave[i+1,n], qhat_p, dt, dx)
        truewave[0,n+1] = LW_step(truewave[0,n], truewave[0,n], truewave[1,n], q((t,-1)), dt, dx)
        truewave[-1,n+1] = LW_step(truewave[-2,n], truewave[-1,n], truewave[-1,n], q((t,1-dx)), dt, dx)
        predwave[0,n+1] = LW_step(predwave[0,n], predwave[0,n], predwave[1,n], qhat((t,-1)), dt, dx)
        predwave[-1,n+1] = LW_step(predwave[-2,n], predwave[-1,n], predwave[-1,n], qhat((t,1-dx)), dt, dx)
    

    utils.animate('Conservation/tests/gaussian_source/wavefront_learning.mp4', tgrid_test, xgrid_test, predwave, truewave)

    plt.show()









if __name__ == '__main__':
    # check()
    hist_dir = f'Conservation/tests/T_ref/histories/T-{T:.2f}'
    model_dir = f'Conservation/models/T_ref/T-{T:.2f}'
    # test_solve(model_dir, hist_dir)
    train_against_waterfront()