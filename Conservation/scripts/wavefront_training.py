import numpy as np
import tensorflow as tf
import utils
import flux

class Config:
    def __init__(self, init, mesh, scheme):
        self.init = init
        self.mesh = mesh
        self.scheme = scheme

def get_wavefront_dataset(config):
    pass
def train_wavefront(model, dataset, config):
    pass


if __name__ == '__main__':
    amax = 1.5
    T = 1
    cfl = .7
    M = 100
    N = int(amax*T*M/(2*cfl))
    dt = 2 / (N-1)
    faces = tf.cast(tf.linspace(-1,1,M),tf.float32)
    # tgrid = tf.cast(tf.linspace(0,T,N),tf.float32)
    # dt = tgrid[1] - tgrid[0]    # type: ignore

    init = lambda x: tf.where(x<0,-1.,1.)
    mesh = flux.Mesh(faces)
    scheme = flux.Godunov(lambda u:u**2/2,dt,mesh.dx)
    config = Config(init, mesh, scheme)
    
    dataset = get_wavefront_dataset(config)
    model = utils.get_model()
    # Possible pretraining here
    model = train_wavefront(model, dataset, config)
    # Save model, do tests, etc.



