import sys

from sklearn import preprocessing
import tensorflow as tf
from scipy.integrate import trapezoid

from utils import *
plt.style.use('ggplot')

def test():
    def F(y):
        return 5*y**2
    y0 = 0.1
    dt = 0.01
    T = 1

    # y = FE_integrate(F, y0, dt, T)
    # t = np.linspace(0, T, int(T/dt))
    meas = Measurements.from_ode(F, y0, dt, T)
    pfig, pax = plt.subplots(1, 1, figsize=(5,5))
    pax.plot(meas.t, meas.y, label='FE path')
    pax.legend()
    plt.show()

def check_model():
    model = Approximator([1, 10, 10, 1])
    test_in = tf.constant([[5]])
    test_out = model(test_in)
    print(test_out)
    model.summary()

def check_approx():
    target_fnc = lambda y: 5*y**2
    data = tf.linspace(0., 1., 100)
    data = tf.expand_dims(data, axis=0)
    target = target_fnc(data)

    model = Approximator([1, 50, 100, 50, 1])
    optim = tfk.optimizers.legacy.Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optim)
    model.fit(data, target, batch_size=5, epochs=1000, verbose=0, callbacks=[MyLogger(50)]) # type: ignore
    # model.summary()

    evals = model(data)
    mse = 10/len(data) * tf.reduce_sum(tf.square(evals-target))
    print(f'Final mse: {mse}')


def CAN_I_EVEN_REGRESS():
    # MAEK THE DATA BITCH
    x = np.linspace(-50, 50, 100)
    x = x.reshape((-1,1))
    y = x**2
    # print(f'{x.min()=}  {x.max()=}  {y.min()=}  {y.max()=}')

    xscaler = preprocessing.MinMaxScaler()
    x = xscaler.fit_transform(x)
    yscaler = preprocessing.MinMaxScaler()
    y = yscaler.fit_transform(y)
    # print(f'{x.min()=}  {x.max()=}  {y.min()=}  {y.max()=}')

    inp = tfkl.Input(shape=1)
    h1 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(inp)
    h2 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(h1)
    h3 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(h2)
    # s = h2 + h3
    # h4 = tfkl.Dense(16, activation='relu', kernel_initializer='he_uniform')(h2)
    out = tfkl.Dense(1, activation='linear')(h3)
    model = tfk.Model(inputs=inp, outputs=out)
    model.compile(loss='mse', optimizer=tfk.optimizers.legacy.Adam(learning_rate=0.005))

    tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    hist = model.fit(x, y, epochs=600, batch_size=10, verbose=0, callbacks=[MyLogger(100)]) # type:ignore
    yhat = model.predict(x)

    x_plot = xscaler.inverse_transform(x)
    y_plot = yscaler.inverse_transform(y)
    yhat_plot = yscaler.inverse_transform(yhat)

    print(f'\nMSE: {tfk.losses.mean_squared_error(tf.reshape(y_plot, [-1]), tf.reshape(yhat_plot, [-1]))}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.scatter(x_plot, y_plot, label=r'$x^2$')
    ax1.scatter(x_plot, yhat_plot, label='Network')
    ax1.legend()
    ax2.plot(hist.history['loss'], label='Loss')
    ax2.set_yscale('log')
    ax2.legend()

    plt.show()

def okay_now_letsgo():
    # First make the ode data
    F = lambda y: y**2
    meas = Measurements.from_ode(F, 0.25, 0.01, 3)
    # Okay this looks like it works
    # Now we should be able to learn the path
    # tscaler = preprocessing.MinMaxScaler()
    # yscaler = preprocessing.MinMaxScaler()
    # t_scaled = tscaler.fit_transform(meas.t.reshape((-1,1)))
    # y_scaled = yscaler.fit_transform(meas.y.reshape((-1,1)))
    # Network stuff
    # inp = tfkl.Input(shape=1)
    # h1 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(inp)
    # h2 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(h1)
    # h3 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(h2)
    # out = tfkl.Dense(1, activation='linear')(h3)
    # model = tfk.Model(inputs=inp, outputs=out)
    # Fitting stuff
    # model.compile(loss='mse', optimizer=tfk.optimizers.legacy.Adam(learning_rate=0.005))
    # hist = model.fit(t_scaled, y_scaled, epochs=600, batch_size=10, verbose=0, callbacks=[MyLogger(100)]) # type:ignore
    # yhat_scaled = model.predict(t_scaled)
    # yhat = yscaler.inverse_transform(yhat_scaled)

    # Okay fantastic! We can fit the path!!
    # Now can we learn the source term???
    ynext = meas.y[1:].reshape((-1,1))
    ycurr = meas.y[:-1].reshape((-1,1))
    # Scale the motherfuckers
    data_scaler = preprocessing.MinMaxScaler()
    target_scaler = preprocessing.MinMaxScaler()
    ycurr_scaled = data_scaler.fit_transform(ycurr)
    ynext_scaled = target_scaler.fit_transform(ynext)
    target_scaled = np.concatenate([ycurr_scaled, ynext_scaled], axis=1)
    # print(f'{data_scaled.min()=}  {data_scaled.max()=}  {target_scaled.shape=}  {target_scaled.min()=}   {target_scaled.max()=}')
    # Define loss and the model, and train it
    def loss(target, predicted):
        ycurr = target[:,0:1]   # Slices needed to preserve dimension
        ynext = target[:,1:2]
        ynext_hat = ycurr + 0.01 * predicted
        return tfk.losses.mean_squared_error(ynext, ynext_hat)
    
    inp = tfkl.Input(shape=1)
    h1 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(inp)
    h2 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(h1)
    h3 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(h2)
    out = tfkl.Dense(1, activation='linear')(h3)
    model = tfk.Model(inputs=inp, outputs=out)
    model.compile(loss=loss, optimizer=tfk.optimizers.legacy.Adam(learning_rate=0.005))
    hist = model.fit(ycurr_scaled, target_scaled, epochs=150, batch_size=10, verbose=0, callbacks=[MyLogger(100)]) # type:ignore

    # Now lets see if we actually learned the square function
    # How should we scale this??
    yhat_scaled = model.predict(ycurr_scaled)
    # print(yhat_scaled.shape)

    path_hat = np.zeros(int(3/0.01))
    path_hat[0] = 0.25
    for i in range(len(path_hat[:-1])):
        dimcorrected = np.array([path_hat[i]]).reshape((-1,1))
        _ycurr_scaled = data_scaler.transform(dimcorrected)
        _ynext_scaled = _ycurr_scaled + 0.01 * model(_ycurr_scaled)     # type:ignore
        path_hat[i+1] = target_scaler.inverse_transform(_ynext_scaled)
    

    l2_error = meas.dt * np.sum((meas.y - path_hat)**2)
    print(f'L² error: {l2_error:.2e}')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    # fig.suptitle(f'$L_2$ error: {l2_error:.2e}')
    ax1.plot(meas.t, meas.y, label='Actual path')
    ax1.plot(meas.t, path_hat, label='Network')
    ax1.legend()
    ax2.plot(hist.history['loss'], label='Loss')
    ax2.set_yscale('log')
    ax2.legend()

    plt.show()
    # plt.plot(meas.t, meas.y, label='FE')
    # plt.plot(meas.t, 1/(4-meas.t), label='Exact')
    # plt.plot(meas.t, yhat, label='Network')
    # plt.legend()
    # plt.show()

def what_are_we_training():
    y0 = 0.25
    dt = 0.01
    T = 3
    F = lambda y: y**2
    meas = Measurements.from_ode(F, y0, dt, T)
    # Here we want to check which function the network actually becomes
    # It is then important to use the same scaler for both the input and target
    scaler = preprocessing.MinMaxScaler()
    y_scaled = scaler.fit_transform(meas.y.reshape((-1,1)))
    ycurr_scaled = y_scaled[:-1,[0]]
    ynext_scaled = y_scaled[1:,[0]]
    target_scaled = np.concatenate([ycurr_scaled, ynext_scaled], axis=1)
    #print(scaler.data_min_, scaler.data_max_, scaler.data_range_, scaler.data_min_-scaler.data_max_)
    
    def loss(target, predicted):
        ycurr = target[:,0:1]   # Slices needed to preserve dimension
        ynext = target[:,1:2]
        ynext_hat = ycurr + 0.01 * predicted
        return tfk.losses.mean_squared_error(ynext, ynext_hat)
    
    inp = tfkl.Input(shape=1)
    h1 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(inp)
    h2 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(h1)
    h3 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(h2)
    out = tfkl.Dense(1, activation='linear')(h3)
    model = tfk.Model(inputs=inp, outputs=out)
    model.compile(loss=loss, optimizer=tfk.optimizers.legacy.Adam(learning_rate=0.005))
    hist = model.fit(ycurr_scaled, target_scaled, epochs=150, batch_size=10, verbose=0, callbacks=[MyLogger(100)]) # type:ignore

    # Now lets see what we actually learned.
    # Should be F(y_raw) / (y_max - y_min)
    yhat_scaled = model.predict(ycurr_scaled)
    postscaled = F(scaler.inverse_transform(ycurr_scaled)) / scaler.data_range_[0]
    
    
    # l2_error = (tgrid[1]-tgrid[0]) * np.sum((yhat_scaled - postscaled)**2)
    l2_error = trapezoid((yhat_scaled-postscaled).flatten()**2, x=ycurr_scaled.flatten())
    print(f'L² error: {l2_error:.2e}')
    plt.plot(ycurr_scaled,yhat_scaled, label=r'$\hat{F}(\tilde{y}_n)$')
    plt.plot(ycurr_scaled,postscaled, label=r'$\frac{F(T^{-1}(\tilde{y}_n))}{y_{max}-y_{min}}$')
    plt.legend()
    plt.show()

def get_error_from_dt(dt):
    # This is to avoid the retracing issue when redefining the compiled loss
    y0 = 0.25
    T = 3
    F = lambda y: y**2
    meas = Measurements.from_ode(F, y0, dt, T)
    scaler = preprocessing.MinMaxScaler()
    y_scaled = scaler.fit_transform(meas.y.reshape((-1,1)))
    ycurr_scaled = y_scaled[:-1,[0]]
    ynext_scaled = y_scaled[1:,[0]]
    target_scaled = np.concatenate([ycurr_scaled, ynext_scaled], axis=1)

    def loss(target, predicted):
        ycurr = target[:,0:1]   # Slices needed to preserve dimension
        ynext = target[:,1:2]
        ynext_hat = ycurr + dt * predicted
        return tfk.losses.mean_squared_error(ynext, ynext_hat)

    inp = tfkl.Input(shape=1)
    h1 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(inp)
    h2 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(h1)
    h3 = tfkl.Dense(30, activation='relu', kernel_initializer='he_uniform')(h2)
    out = tfkl.Dense(1, activation='linear')(h3)
    model = tfk.Model(inputs=inp, outputs=out)
    model.compile(loss=loss, optimizer=tfk.optimizers.legacy.Adam(learning_rate=0.005))
    callbacks = [MyLogger(30), tfk.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)]
    hist = model.fit(ycurr_scaled, target_scaled, epochs=130, batch_size=10, verbose=0, callbacks=callbacks) # type:ignore

    # Needed to accurately asses the prediction error
    yax = np.linspace(0,1,1000).reshape((-1,1))
    yax_scaled = scaler.transform(yax)
    yhat_scaled = model.predict(yax_scaled)
    postscaled = F(scaler.inverse_transform(yax_scaled)) / scaler.data_range_[0]
    
    func_error = trapezoid((yhat_scaled-postscaled).flatten()**2, x=yax_scaled.flatten())

    path_dt = 0.001
    path_hat = np.zeros(int(3/path_dt))
    path_hat[0] = 0.25
    path = path_hat.copy()
    for i in range(len(path_hat[:-1])):
        # Analytical
        path[i+1] = path[i] + path_dt * F(path[i])
        # Network
        dimcorrected = np.array([path_hat[i]]).reshape((-1,1))
        _ycurr_scaled = scaler.transform(dimcorrected)
        _ynext_scaled = _ycurr_scaled + path_dt * model(_ycurr_scaled)     # type:ignore
        path_hat[i+1] = scaler.inverse_transform(_ynext_scaled)
    
    
    path_error = trapezoid((path_hat-path)**2, x=np.linspace(0,3,len(path)))

    return func_error, path_error

def refinement():
    # dt_range = 5.0**np.arange(-1, -5, -1, dtype=np.int32)
    # dt_range = np.logspace()
    dt_range = np.geomspace(0.005, 0.9, 6)
    func_errors = []
    path_errors = []
    for dt in dt_range:
        print(f'\n-------- Starting run with {dt=:4f} --------')
        fe, pe = get_error_from_dt(dt)
        func_errors.append(fe)
        path_errors.append(pe)
        print(f'Finished with function error {fe} and path error {pe}')
        tf.keras.backend.clear_session()
    print(func_errors)
    print(path_errors)

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.set_title('Function error')
    ax1.plot(dt_range, func_errors, 'x-')
    ax1.set_xlabel(r'$\Delta t$')
    ax1.set_ylabel(r'$L^2$ error')
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ax2.set_title('Path error')
    ax2.plot(dt_range, path_errors, 'x-')
    ax2.set_xlabel(r'$\Delta t$')
    ax2.set_ylabel(r'$L^2$ error')
    ax2.set_yscale('log')
    ax2.set_xscale('log')

    plt.show()
    


if __name__ == '__main__':
    #test()
    # check_model()
    # check_approx()
    # CAN_I_EVEN_REGRESS()
    # okay_now_letsgo()
    # what_are_we_training()
    # Call from command line by supplying dt
    # get_error_from_dt(float(sys.argv[1]))
    refinement()