import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tfk = tf.keras
plt.style.use('ggplot')

class Equation:
    def __init__(self, a, initial, source):
        '''Periodic BC, thus also periodic initial and source as well'''
        self.a = a
        self.initial = initial
        self.source = source


def get_A(a, M, r):
    """Gets the iteration matrix of explicit upstream w periodic BC.

    Args:
        a (double): wave speed
        M (int): # of gridpoints
        r (double): dt / dx

    Returns:
        ndarray: matrix of shape (M,M)
    """
    A = np.zeros((M,M))
    hi = np.ones(M-1) * r*(abs(a)-a)/2
    di = np.ones(M) * (-r*abs(a)+1)
    lo = np.ones(M-1) * r*(abs(a)+a)/2
    # Fills the matrix without a for loop
    dslice = np.arange(len(A))
    A[dslice, dslice] = di
    A[dslice[:-1], dslice[1:]] = hi
    A[dslice[1:], dslice[:-1]] = lo
    # Relies on the fact that a is periodic as well
    if a >= 0:
        A[0,-1] = r*a
    else:
        A[-1,0] = r*a
    return A

# region DATA
# def refine_from_threshold(arr, threshold):
#     # Will become useful for nonuniform xgrids
#     size = len(arr)
#     idx = np.arange(size)
#     refined_arr = []
#     for i in range(size-1):
#         while arr[i+1] - arr[i] > threshold:
#             pass

def get_data(equation, tvals, xgrid, cfl=0.4):
    # Assumes uniform gridpoints in x. TODO: implement source
    dx = np.max(np.diff(xgrid))                          # Calculates maximum dx
    dt_max = cfl * dx / equation.a                       # Uses CFL condition to get max dt
    A_max = get_A(equation.a, len(xgrid), dt_max/dx)     # Creates A using maximum dt
    U = np.zeros((len(xgrid), len(tvals)+1))             # Each col a solution for some t
    U[:,0] = equation.initial(xgrid)
    t = 0
    for i, t_next in enumerate(tvals):
        dt = t_next - t
        if dt <= dt_max:
            A = get_A(equation.a, len(xgrid), dt/dx)
            U[:,i+1] = A@U[:,i]
        else:
            tmp = U[:,i]
            while t+dt_max < t_next:
                tmp = A_max@tmp
                t += dt_max
            A = get_A(equation.a, len(xgrid), (t_next-t)/dx)
            U[:,i+1] = A@tmp
        t = t_next
    return U
# endregion

# region NETWORK
def get_model():
    tfkl = tfk.layers
    inp = tfkl.Input(shape=(2,))
    h1 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(inp)
    h2 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(h1)
    h3 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(h2)
    out = tfkl.Dense(1, activation='linear')(h3)
    model = tfk.Model(inputs=inp, outputs=out)
    # model.compile(loss=loss, optimizer=tfk.optimizers.legacy.Adam(learning_rate=0.005))
    return model

# Logger from internet
class LowkeyLogger(tfk.callbacks.Callback):
  def __init__(self, n):
    self.n = n   # print loss & acc every n epochs

  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.n == 0:
      curr_loss = logs.get('loss')
    #   curr_acc = logs.get('acc') * 100
      tf.print("epoch = %4d  loss = %5.2e " \
        % (epoch, curr_loss))

# endregion


# region tests
def solvertest():
    eq = Equation(1., lambda x: np.sin(2*np.pi*x), None)
    tvals = [0.1, 0.4, 0.5]
    xgrid = np.linspace(-1, 1, 50)
    U = get_data(eq, tvals, xgrid)
    for i in range(U.shape[1]):
        plt.plot(xgrid, U[:,i], label=f'$t={0 if i==0 else tvals[i-1]:.2f}$')
    plt.legend()
    plt.show()

def networktest():
    qhat = get_model()
    # qhat.summary()
    # tf.keras.utils.plot_model(qhat, show_shapes=True)
    # Checks if we can learn 0 function
    n_points = 50
    X_train = tf.random.uniform(shape=[n_points,1], minval=-1, maxval=1)
    def q_func(x):
        return tf.where(abs(x)<.2, .1, 0)
    # y_train = tf.zeros(shape=X_train.shape)
    y_train = q_func(X_train)
    qhat.compile(loss='mse', optimizer=tfk.optimizers.legacy.Adam(learning_rate=0.005))
    callbacks = [LowkeyLogger(50), tfk.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)]
    hist = qhat.fit(X_train, y_train, epochs=500, batch_size=5, verbose=0, callbacks=callbacks) # type: ignore
    X_test = np.linspace(-1.5,1.5,200)
    y_test = np.zeros((X_test.shape))
    preds = qhat(X_test.reshape((-1,1)))

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=[10,5])
    ax1.plot(X_test, y_test, '--', label='True')
    ax1.plot(X_test, preds, label='Predicted')
    ax1.legend()
    ax2.plot(hist.history['loss'])
    ax2.set_yscale('log')
    plt.show()


# endregion


if __name__ == '__main__':
    networktest()


