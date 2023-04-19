import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

tfk = tf.keras

def create_animation_artists(ax: plt.Axes, tgrid, xgrid, *args):
    artists = []
    for i in range(len(tgrid)):
        frame = []
        numline, = ax.plot(xgrid, args[0][:,i], color="#6533DB")
        frame.append(numline)
        if len(args) == 2:
           trueline, = ax.plot(xgrid, args[1][:,i], '--', alpha=.6, color="#D74848")
           frame.append(trueline)
        ann = ax.annotate(f'$t={tgrid[i]:.2f}$', (0.5,1.03), xycoords='axes fraction', ha='center')
        frame.append(ann)
        artists.append(frame)
    return artists

def animate(name, tgrid, xgrid, *args):
    '''Send true solution last'''
    total_time = 2000 # ms
    fps = 5
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    artists = create_animation_artists(ax, tgrid, xgrid, *args)
    anim = animation.ArtistAnimation(fig, artists, interval=fps*total_time / len(artists), repeat=True)
    writer = animation.FFMpegWriter(fps=fps)
    anim.save(name, writer=writer)

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

def get_multisymbolic_model():
    tfkl = tfk.layers
    # Define input
    north, west, point, east = [tfkl.Input(shape=2) for i in range(4)]
    h1 = tfkl.Dense(40, activation='relu', kernel_initializer='he_uniform')
    h2 = tfkl.Dense(40, activation='relu', kernel_initializer='he_uniform')
    h3 = tfkl.Dense(40, activation='relu', kernel_initializer='he_uniform')
    out = tfkl.Dense(1, activation='linear')
    layers = [h1, h2, h3, out]
    def connect(inp):
      x = inp
      for l in layers:
        x = l(x)
      return x
    concat = tf.concat([connect(north), connect(west), connect(point), connect(east)], axis=1)
    model = tfk.Model(inputs=[north, west, point, east], outputs=concat)
    # model.compile(loss=loss, optimizer=tfk.optimizers.legacy.Adam(learning_rate=0.005))
    return model

def create_coordinate_arrays(tgrid, xgrid):
    t, x = np.meshgrid(tgrid, xgrid)
    t, x = t.flatten(), x.flatten()
    coords = np.column_stack([t, x])
    return coords

def LW_create_coordinate_arrays(tgrid, xgrid):
    '''P is the central stencil point; north, east, west are corresponding.
       Returns [north, west, point, east] coord arrays'''
    k = tgrid[1] - tgrid[0]
    P = np.meshgrid(tgrid, xgrid)
    tp, xp = [a.flatten() for a in P]
    tn = tp+k
    xw = np.roll(xp, len(tgrid))
    xe = np.roll(xp, -len(tgrid))

    north = np.column_stack([tn, xp])
    west = np.column_stack([tp, xw])
    point = np.column_stack([tp, xp])
    east = np.column_stack([tp, xe])
    return north, west, point, east

# Logger from internet
class LowkeyLogger(tfk.callbacks.Callback):
  def __init__(self, n):
    self.n = n   # print loss & acc every n epochs
    self.n_epoch_begin = time.time()
  
  def on_train_begin(self, logs={}):
    #  return super().on_train_begin(logs)
    self.begin = time.time()

  def on_epoch_begin(self, epoch, logs=None):
    #  return super().on_epoch_begin(epoch, logs)
    if epoch % self.n == 1:
      self.n_epoch_begin = time.time()
  
  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.n == 0:
      curr_loss = logs.get('loss')
      n_epoch_finished = time.time()
      n_epoch_time = n_epoch_finished - self.n_epoch_begin
      elapsed = n_epoch_finished - self.begin
    #   curr_acc = logs.get('acc') * 100
      tf.print("epoch = %4d  loss = %5.2e  epoch_avg = %3ds  elapsed = %4ds " \
        % (epoch, curr_loss, n_epoch_time/self.n, elapsed))


# ----------------------------------------------------------------
# ---------------------- Wavefront utils -------------------------
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
