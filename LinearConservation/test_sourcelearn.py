# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from math import ceil
from learning_source import *
plt.style.use('ggplot')
# %%
cfl = 0.4
T = 2
M = 50
def source(t,x):
    hat = np.where(abs(x)<0.3, 0.2, 0.)
    return hat * np.cos(np.pi*t)
eq = Equation(1., lambda x: np.sin(2*np.pi*x), source)    # Equivalent to learning 0 source
xgrid = np.linspace(-1,1,M)
dx = xgrid[1] - xgrid[0]
dt_max = cfl * dx / eq.a
n_tvals = ceil(T / dt_max) + 1
tvals = np.linspace(0, T, n_tvals)      # Contains all tvalues, also T
# tvals = np.delete(tvals, np.arange(n_tvals)[::3])

diff = np.diff(tvals)
dt = diff[0]
print(f'dt_max: {dt_max}')
print(f'max: {np.max(diff)}, min: {np.min(diff)}')

# %%
tvals_data = np.repeat(tvals[:-1], len(xgrid))
xgrid_data = np.tile(xgrid, n_tvals-1)
print(tvals_data.shape, xgrid_data.shape)
X = np.stack([tvals_data, xgrid_data]).T
print(X.shape)

# %%
# Generate data and format it for training
# Try it first for uniform time grid
A = get_A(eq.a, M, dt / dx)
V = np.zeros((M,n_tvals))
V[:,0] = eq.initial(xgrid)
for i in range(n_tvals-1):
    V[:,i+1] = A@V[:,i] + eq.source(tvals[i], xgrid)
# plt.plot(xgrid, V[:,0])
# plt.plot(xgrid, V[:,10])
print(V.shape)

# %%
# Now: building the target
# Recall that we cannot iterate on the last time interval
# Thus we cannot include the last timestep in the data or target
y = np.zeros((X.shape[0], 4))
for t_ind, t in enumerate(tvals[:-1]):
    for x_ind, x in enumerate(xgrid):
        y_ind = x_ind + M*t_ind
        mm1 = M-1 if x_ind==0 else x_ind-1   # Ensures periodic BCs (positive a)
        adt_dx = eq.a*(tvals[t_ind+1]-t) / dx
        v_m_n = V[x_ind, t_ind]
        v_mm1_n = V[mm1, t_ind]
        v_m_np1 = V[x_ind, t_ind+1]
        y[y_ind,:] = [adt_dx, v_m_n, v_mm1_n, v_m_np1]

# %%
print(X.shape)
print(y.shape)


# %%
# We now train to see if the model is able to recover the zero function
def loss(target, predicted):
    # Target contains 4 values, predicted
    # Slices needed to preserve dimensions
    # adt_dx, v_m_n, v_mm1_n, v_m_np1 = tf.split(target, 4, 1)
    adt_dx = target[:,0:1]
    vmn = target[:,1:2]
    upred = vmn - adt_dx*(vmn-target[:,2:3]) + predicted    # Recall predicted is qhat
    return tf.keras.losses.mean_squared_error(target[:,3:4], upred)

# %%
qhat = get_model()
qhat.compile(loss=loss, optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.005))
# qhat.summary()

# %%
callbacks = [tfk.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True), LowkeyLogger(10)]
hist = qhat.fit(X, y, epochs=70, batch_size=50, verbose=0, callbacks=callbacks) # type: ignore

# %%

# checkx = np.linspace(-2,2,200)
# checkt0 = np.zeros_like(checkx)
# checkt1 = checkt0 + 1.

# res0 = qhat(np.stack([checkt0,checkx]).T)
# res1 = qhat(np.stack([checkt1,checkx]).T)

# plt.style.use('ggplot')
# fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
# ax1.plot(checkx, res0, label='$t=0$')
# ax1.plot(checkx, res1, label='$t=1$')
# ax1.plot(checkx, eq.source(0,checkx), '--', label='True ($t=0$)')
# ax1.plot(checkx, eq.source(1,checkx), '--', label='True ($t=1$)')
# ax1.set_ylim(-.5,.5)
# ax1.legend()

# ax2.plot(hist.history['loss'])
# ax2.set_yscale('log')

# plt.show()

# %%
import matplotlib.animation as animation
checkx = np.linspace(-2,2,400)
checkt = np.linspace(0,2, 200)
_tarr = np.zeros_like(checkx)
numsol = np.zeros((len(checkx), len(checkt)))
for i, t in enumerate(checkt):
    numsol[:,i] = qhat(np.column_stack([_tarr+t, checkx])).numpy().T    # type: ignore
def create_animation_artists(ax: plt.Axes, tgrid, xgrid, sol):
    artists = []
    for i in range(len(tgrid)):
        numline, = ax.plot(xgrid, sol[:,i], color="#33CDDB")
        exline, = ax.plot(xgrid, source(tgrid[i],xgrid), '--', color='#FF0000', alpha=.6)
        ann = ax.annotate(f'$t={tgrid[i]:.2f}$', (0.5,1.03), xycoords='axes fraction', ha='center')
        artists.append([numline, exline, ann])
    return artists

fig, ax = plt.subplots(1,1,figsize=(5,5))
artists = create_animation_artists(ax, checkt, checkx, numsol)
anim = animation.ArtistAnimation(fig, artists, interval=100, repeat=True)
writer = animation.FFMpegWriter(fps=60)
anim.save('oscillation_hat_source.mp4', writer=writer)
plt.show()

# %%
