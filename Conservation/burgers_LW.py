import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils

plt.style.use('ggplot')

class Solver:
    def __init__(self, F, init, q, cfl=.4, M=100):
        self.F = F
        self.init = init
        self.q = q
        self.xgrid = np.linspace(-1,1,M+1)[:-1]
        dx = self.xgrid[1] - self.xgrid[0]
        self.dx = dx
        T = .2
        dt = cfl*dx/np.max(np.abs(init(self.xgrid)))
        self.tgrid = np.linspace(0, T, int(T/dt))
        self.dt = self.tgrid[1] - self.tgrid[0]
        # self.r = self.dt/dx

    def LW(self, u_bl, u_bc, u_br, q_tc, q_bl, q_bc, q_br):
        '''Here u and q are the solution and source respectively. Subscripts refer to stencil position (t=top,b=bottom,l=left,c=center,r=right).'''
        dt, dx = self.dt, self.dx
        r = dt/dx
        F_bc, F_br, F_bl = self.F(u_bc), self.F(u_br), self.F(u_bl)
        # q_tc, q_bc = self.q(t+dt,x), self.q(t,x)
        # q_bl, q_br = self.q(t,x-dx), self.q(t,x+dx)
        u_iph = (u_bc+u_br)/2
        u_imh = (u_bc+u_bl)/2
        tmp = u_bc - r/2*(F_br-F_bl)
        nosource = tmp + r/2*(u_iph*(F_br-F_bc) - u_imh*(F_bc-F_bl))
        return nosource + dt/2*(q_bc+q_tc) - dt/4*(u_iph*(q_br-q_bc) - u_imh*(q_bc-q_bl))
    
    def solve(self):
        tgrid = self.tgrid
        x_c = self.xgrid
        x_l = np.roll(x_c, 1)   # Rolls to the right, x_l[0] == x_c[-1]
        x_r = np.roll(x_c, -1)  # Rolls to the left, x_r[-1] == x_c[0]
        dt, dx = self.dt, self.dx
        U = np.zeros((len(x_c), len(tgrid)))
        U[:,0] = self.init(x_c)
        for i, t in enumerate(tgrid[:-1]):
            q_tc, q_bc = self.q(t+dt,x_c), self.q(t,x_c)
            q_bl, q_br = self.q(t,x_l), self.q(t,x_r)
            u_bl = np.roll(U[:,i], 1)
            u_br = np.roll(U[:,i], -1)
            U[:,i+1] = self.LW(u_bl, U[:,i], u_br, q_tc, q_bl, q_bc, q_br)
            # U[0,i+1] = self.LW(U[-1,i], U[0,i], U[1,i])
            # U[-1,i+1] = self.LW(U[-2,i], U[-1,i], U[0,i])
            # U[:,i+1] += self.dt*self.q(tgrid[i], xgrid)
        return U
    
    def get_source(self):
        tgrid = self.tgrid
        xgrid = self.xgrid
        return self.q(*np.meshgrid(tgrid,xgrid))

def learn_nonlinear_source():
    # F = lambda u: u**2/2
    def F(u):
        return u**2/2
    # F = lambda u: u
    # init = lambda x: -np.sin(np.pi*x)
    def init(x):
        return -np.sin(np.pi*x)
    # q = lambda t,x: .5*np.sin(np.pi*t*x)
    # def q(t,x):
    #     hat = np.where(abs(x+.5)<0.2, 0.5, 0)
    #     return hat# * np.cos(np.pi*t)
    q = lambda t,x: 5*np.exp(-20*(x+.5)**2)# * np.cos(2*np.pi*t)
    # q = lambda t,x: 0.
    M = 100
    solver = Solver(F, init, q, cfl=.1, M=M)
    V = solver.solve()
    tgrid, xgrid = solver.tgrid, solver.xgrid
    Xn, Xw, Xp, Xe = utils.LW_create_coordinate_arrays(tgrid[:-1], xgrid)
    # Target needs to contain all four stencil points of Lax-Wendroff
    y = np.zeros((Xp.shape[0], 4))
    for t_idx, t in enumerate(tgrid[:-1]):
        for x_idx, x in enumerate(xgrid):
            y_ind = x_idx + M*t_idx
            left_idx = M-1 if x_idx==0 else x_idx-1   # Ensures periodic BCs
            right_idx = 0 if x_idx==M-1 else x_idx+1   # Ensures periodic BCs
            # [TC, BL, BC, BR]
            y[y_ind,:] = [V[x_idx,t_idx+1], V[left_idx,t_idx], V[x_idx,t_idx], V[right_idx,t_idx]]
    input_dict = {"input_1": Xn, "input_2": Xw, "input_3": Xp, "input_4": Xe}
    dataset = tf.data.Dataset.from_tensor_slices((input_dict, y))
    # ------------- TRAINING -------------
    return train_and_predict(q, solver, dataset)


def train_and_predict(q, solver, dataset: tf.data.Dataset):
    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    def loss(target, predicted):
        u_tc, u_bl, u_bc, u_br = tf.unstack(target, axis=1)
        q_tc, q_bl, q_bc, q_br = tf.unstack(predicted, axis=1)
        uhat = solver.LW(u_bl, u_bc, u_br, q_tc, q_bl, q_bc, q_br)
        return tf.keras.losses.mean_squared_error(u_tc, uhat)
    qhat_train = utils.get_multisymbolic_model()
    qhat_train.compile(loss=loss, optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.005))
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True), utils.LowkeyLogger(10)]
    # callbacks.append()
    hist = qhat_train.fit(dataset, epochs=100, verbose=0, callbacks=callbacks) # type: ignore
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(hist.history['loss'])
    ax.set_yscale('log')
    plt.show()
    # ------------ PREDICTING ------------
    print('\nPREDICTING...')
    num_symb_inp = 4
    qhat = tf.keras.models.Sequential(qhat_train.layers[num_symb_inp:-1]) # Picks out the core layers
    xgrid_test = np.linspace(-2,2,200)
    tgrid_test = np.linspace(0,3,200)
    X_test = utils.create_coordinate_array(tgrid_test, xgrid_test)
    true = q(*np.meshgrid(tgrid_test, xgrid_test))
    predicted = qhat.predict(X_test).reshape(true.shape)    # type: ignore
    qhat.save('Conservation/models/multisymbolic')
    # ------------- PLOTTING -------------
    utils.animate('Conservation/gaussian_source/burgers_learned.mp4', tgrid_test, xgrid_test, predicted, true)
    





if __name__ == '__main__':
    # TODO: Check if learning works by using explicit upwind instead of solver.LW
    # TODO: Lax-Wendroff needs special treatment of the source because of the Taylor substitution!!
    # TODO: New strategy: we need model evaluations in different stencil points. Either modify symbolic input (symbolic_input.py) or write custom loop (boring)
    learn_nonlinear_source()