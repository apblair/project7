# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
import numpy as np
from nn import nn, preprocess

# TODO: Write your test functions and associated docstrings below.

def test_forward():
    pass


def test_single_forward():
    '''
    Check NeuralNetwork._single_forward against manual calculation using a relu activation function
    '''
    net = nn.NeuralNetwork(nn_arch = [{"input_dim": 4, "output_dim": 1, "activation": "relu"}],
                           lr = 0.01, seed = 26, batch_size = 1, epochs = 1, loss_function = "mse")
    A_curr, Z_curr = net._single_forward(np.array([[1, 2, 3, 4]]), # W_curr
                                        np.array([[6]]), # b_curr
                                        np.array([[4, 3, 2, 1]]), # A_prev
                                        net.arch[0]['activation'])
    assert A_curr[[0]] and Z_curr[[0]] == 26


def test_single_backprop():
    '''
    Check NeuralNetwork._single_backprop against manual calculation using a relu activation function
    '''
    net = nn.NeuralNetwork(nn_arch = [{"input_dim": 4, "output_dim": 1, "activation": "relu"}],
                           lr = 0.01, seed = 26, batch_size = 1, epochs = 1, loss_function = "mse")
    dA_prev, dW_curr, db_curr = net._single_backprop(np.array([[1, 2, 3, 4]]), # W_curr
                                                    np.array([[6]]), # b_curr
                                                    np.array([[26]]), # z_curr
                                                    np.array([[4, 3, 2, 1]]), # A_prev
                                                    np.array([[26]]), # dA_curr
                                                    net.arch[0]['activation'])
    assert dA_prev.sum() and dW_curr.sum() == 260
    assert dA_prev[0][0] and dW_curr[0][-1] == db_curr[[0]]

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
test_single_backprop()