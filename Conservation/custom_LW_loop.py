import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
import time
tfk = tf.keras

# dx = 0.01
# dt = 0.001
amax = 3
T = 0.2
cfl = .2
M = 100
N = int(amax*T*M/(2*cfl))

F = lambda u: u**2/2
init = lambda x: -np.sin(np.pi*x)
# q = lambda p: 2*np.exp(-20*(p[1]+.5)**2) * np.cos(2*np.pi*p[0])
q = lambda p: np.where(abs(p[1]+.5)<.2, .5, 0)

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

def solve():
    # def q(p):
    #     return 10*np.exp(-50*(p[1]+.5)**2)
    xgrid = np.linspace(-1,1,M+1)[:-1]
    tgrid = np.linspace(0,T,N)
    dx = xgrid[1] - xgrid[0]
    dt = tgrid[1] - tgrid[0]
    
    V = np.zeros((len(xgrid), len(tgrid)))
    V[:,0] = init(xgrid)

    for n, t in enumerate(tgrid[:-1]):
        for i, x in enumerate(xgrid[1:-1], start=1):
            q_p = q((t, x))
            V[i,n+1] = LW_step(V[i-1,n], V[i,n], V[i+1,n], q_p, dt, dx)
        
        V[0,n+1] = LW_step(V[-1,n], V[0,n], V[1,n], q((t,-1)), dt, dx)
        V[-1,n+1] = LW_step(V[-2,n], V[-1,n], V[0,n], q((t,1-dx)), dt, dx)
    return tgrid, xgrid, V

def test_solve():
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
    
    model.save('Conservation/models/single_eval')
    preds = model.predict(X).flatten()
    # fig, (ax,ax2) = plt.subplots(1, 2, projection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.plot_trisurf(X[:,0], X[:,1], preds) # type:ignore
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    ax2.plot_trisurf(X[:,0], X[:,1], np.apply_along_axis(q,1,X)) # type:ignore
    ax2.set_xlabel('$t$')
    ax2.set_ylabel('$x$')
    plt.show()





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



if __name__ == '__main__':
    # check()
    test_solve()