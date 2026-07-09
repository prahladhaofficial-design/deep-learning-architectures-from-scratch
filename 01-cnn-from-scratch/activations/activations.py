import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh_func(x):
    exp_pos = np.exp(np.clip(x, -500, 500))
    exp_neg = np.exp(np.clip(-x, -500, 500))
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)


def tanh_func_derivative(x):
    t = tanh_func(x)
    return 1.0 - t**2


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def prelu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def prelu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


def softmax(x):
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)
