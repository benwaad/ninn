# AI assisted numerics for inverse problems in nonlinear transport

This repository was created to support my Master's thesis *AI assisted numerics for inverse problems in nonlinear transport* with numerical experiments and visualizations.
The thesis was written during the spring of 2023 under the supervision of Kjetil Olsen Lye at Sintef.

## Contents
The contents of the repository is organized in folders corresponding to the different sections of the thesis.
The code is written in Python, possibly with influence from established packages in Matlab.

### ODE

Uses neural networks to learn the right hand side of an ordinary differential equation.
Explores how the size of the training set impacts the error of the network approximation.

### Conservation

Contains utilities for numerically solving hyperbolic conservation laws.
Influenced by the Matlab package [COMPACK](https://github.com/ulriksf/compack).


