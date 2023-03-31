import numpy as np
import tensorflow as tf

class Mesh:
    def __init__(self,faces):
        '''Assumes uniform faces'''
        self.faces = faces
        self.centroids = .5*(faces[:-1] + faces[1:])
        self.dx = tf.constant(faces[1] - faces[0],dtype=tf.float32)

class Scheme:
    def __init__(self, F, dt, dx):
        '''F is the flux'''
        self.F = F
        self.dt = tf.cast(dt,tf.float32)
        self.dx = tf.cast(dx,tf.float32)
    def step(self, *args):
        raise NotImplementedError()
    
class LW(Scheme):
    def __init__(self, F, dt, dx):
        super().__init__(F, dt, dx)
    def step(self, west, centre, east):
        pass

class Godunov(Scheme):
    def __init__(self, F, dt, dx):
        '''Made for convex fluxes with minimum in 0. Assumes batch dimension'''
        super().__init__(F, dt, dx)
        print(f'INFO: Created scheme with Δt/Δx = {dt/dx:.2f}.')
    def max(self, a, b):
        '''Convenience'''
        return tf.math.maximum(a,b)
    def min(self, a, b):
        '''Convenience'''
        return tf.math.minimum(a,b)
    def vectorstep(self, Un):
        '''Un is sent in with ghost cell paddings, Dirichlet'''
        internals = Un[:,1:-1]
        rightshift = Un[:,2:]
        leftshift = Un[:,:-2]
        F_p = self.max(self.F(self.max(internals,0.)), self.F(self.min(rightshift,0.)))
        F_m = self.max(self.F(self.max(leftshift,0.)), self.F(self.min(internals,0.)))
        next_internals = internals - self.dt/self.dx * (F_p-F_m)  # type:ignore
        leftcond = Un[:,0:1]
        rightcond = Un[:,-1:]
        return tf.concat([leftcond, next_internals, rightcond], axis=1)

    def step(self):
        pass

