import subprocess as sub

import numpy as np
import tensorflow as tf


def refine():
    process = sub.run(['python', 'tests.py', '0.1'],
                      stdout=sub.PIPE,
                      stderr=sub.PIPE,
                      universal_newlines=True)
    print(process)
    print('\n-------- OUTPUT --------')
    print(process.stdout)
    print('\n-------- ERRORS --------')
    print(process.stderr)

if __name__ == '__main__':
    refine()