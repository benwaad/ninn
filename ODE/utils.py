import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

def FE_integrate(F, y0, dt, T):
    '''F = F(y)'''
    y = np.zeros(int(T/dt))
    y[0] = y0
    if isinstance(F, tfk.Model):
        for i in range(len(y[:-1])):
            y[i+1] = y[i] + dt*tf.squeeze(F(tf.expand_dims(y[i], axis=0)))
    else:
        for i in range(len(y[:-1])):
            y[i+1] = y[i] + dt*F(y[i])
    return y

class Measurements:
    def __init__(self, yarray, tarray):
        '''Stores measurements along with T and dt'''
        self.t = tarray
        self.y = yarray
        self.T = tarray[-1]
        self.dt = tarray[1] - tarray[0]
    @classmethod
    def from_anal(cls, y, tarray):
        '''Generate measurements from y=y(t)'''
        return cls(y(tarray), tarray)
    @classmethod
    def from_ode(cls, F, y0, dt, T):
        '''Generate measurements from y'=F(y)'''
        y = FE_integrate(F, y0, dt, T)
        t = np.linspace(0, T, len(y))
        return cls(y, t)
    
# Can I make a neural network that aproximates a function?
class Approximator(tfk.Model):
    def __init__(self, structure):
        """A Neural Network model for approximating a function.

        Args:
            structure (iterable): Iterable describing the structure of the network.
                                  If structure=[1,10,10,1], the network will have
                                  two hidden layers, each with 10 nodes, along with a
                                  one-dimensional input and output.
        """
        super().__init__(name='Approximator')
        self.num_hidden = len(structure) - 1
        init = tfk.initializers.HeUniform(seed=42)
        # self.inputs = tfk.Input(shape=(structure[0],))
        self.hidden = [
            tfkl.Dense(structure[i+1], kernel_initializer=init) # type:ignore
            for i in range(self.num_hidden)
        ]
    
    def call(self, inputs):
        x = inputs
        for i in range(self.num_hidden-1):
            x = self.hidden[i](x)
            x = tfk.activations.relu(x, alpha=0.3)
        return self.hidden[-1](x)
    

# Logger from internet
class MyLogger(tfk.callbacks.Callback):
  def __init__(self, n):
    self.n = n   # print loss & acc every n epochs

  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.n == 0:
      curr_loss = logs.get('loss')
    #   curr_acc = logs.get('acc') * 100
      tf.print("epoch = %4d  loss = %5.2e " \
        % (epoch, curr_loss))