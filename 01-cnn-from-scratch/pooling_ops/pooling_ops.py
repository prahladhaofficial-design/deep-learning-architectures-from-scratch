import numpy as np


# Pooling function  returns the maximum value in the window
def max_pool_func(window):
    return np.max(window)


# Pooling function  returns the average value in the window
def avg_pool_func(window):
    return np.mean(window)
