# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
import numpy as np
from nn import nn, preprocess

# TODO: Write your test functions and associated docstrings below.

def test_forward():
    pass


def test_single_forward():
    net = nn.NeuralNetwork(nn_arch = [{"input_dim": 4, "output_dim": 1, "activation": "relu"}],
                           lr = 0.01, seed = 26, batch_size = 1, epochs = 1, loss_function = "mse")
    A_curr, Z_curr = net._single_forward(np.array([[1, 2, 3, 4]]), np.array([[6]]), np.array([[4, 3, 2, 1]]),net.arch[0]['activation'])
    assert A_curr[[0]] and Z_curr[[0]] == 26


def test_single_backprop():
    pass


def test_predict():
    pass


def test_binary_cross_entropy():
    pass


def test_binary_cross_entropy_backprop():
    pass


def test_mean_squared_error():
    pass


def test_mean_squared_error_backprop():
    pass


def test_one_hot_encode():
    pass


def test_sample_seqs():
    pass

test_single_forward()