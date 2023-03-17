import numpy as np
import tensorflow as tf




class Scheme:
    def __init__(self, flux_fnc, dt, dx):
        self.F = flux_fnc
        self.dt = dt
        self.dx = dx
        self.s = dt/dx
    def step(self, *args):
        raise NotImplementedError()
    
class LW(Scheme):
    def __init__(self, flux_fnc, dt, dx):
        super().__init__(flux_fnc, dt, dx)
    def step(self, west, centre, east):
        pass

class Rusanov(Scheme):
    def __init__(self, flux_fnc, dt, dx):
        super().__init__(flux_fnc, dt, dx)
    def step(self):
        pass


