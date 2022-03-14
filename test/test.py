# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
import numpy as np
from nn import nn, preprocess

# TODO: Write your test functions and associated docstrings below.

def test_forward():
    '''
    Check NeuralNetwork.forward with a relu activation function using manual calculation 
    '''
    network = nn.NeuralNetwork(nn_arch = [{"input_dim": 4, "output_dim": 2, "activation": "relu"},
                                        {"input_dim": 2, "output_dim": 1, "activation": "relu"}],
                                        lr = 0.01, seed = 26, batch_size = 1, epochs = 1, loss_function = "mse")
    network._param_dict = {"W1": np.array([[1, 2, 3, 4]]), "b1": np.array([[6],[6]]),
                            "W2": np.array([[1,2]]), "b2":np.array([[6]])}
    A_final, cache = network.forward(np.array([4,3,2,1]))
    assert np.array_equal(cache['A1'], cache['Z1'])
    assert np.array_equal(cache['A2'], cache['Z2'], A_final)

def test_single_forward():
    '''
    Check NeuralNetwork._single_forward with a relu activation function using manual calculation 
    '''
    network = nn.NeuralNetwork(nn_arch = [{"input_dim": 4, "output_dim": 1, "activation": "relu"}],
                           lr = 0.01, seed = 26, batch_size = 1, epochs = 1, loss_function = "mse")
    A_curr, Z_curr = network._single_forward(np.array([[1, 2, 3, 4]]), # W_curr
                                        np.array([[6]]), # b_curr
                                        np.array([[4, 3, 2, 1]]), # A_prev
                                        network.arch[0]['activation'])
    assert A_curr[[0]] == Z_curr[[0]]


def test_single_backprop():
    '''
    Check NeuralNetwork._single_backprop with a relu activation function using manual calculation 
    '''
    network = nn.NeuralNetwork(nn_arch = [{"input_dim": 4, "output_dim": 1, "activation": "relu"}],
                           lr = 0.01, seed = 26, batch_size = 1, epochs = 1, loss_function = "mse")
    dA_prev, dW_curr, db_curr = network._single_backprop(np.array([[1, 2, 3, 4]]), # W_curr
                                                    np.array([[6]]), # b_curr
                                                    np.array([[26]]), # z_curr
                                                    np.array([[4, 3, 2, 1]]), # A_prev
                                                    np.array([[26]]), # dA_curr
                                                    network.arch[0]['activation'])
    assert dA_prev.sum() == dW_curr.sum()
    assert dA_prev[0][0] and dW_curr[0][-1] == db_curr[[0]]

def test_predict():
    pass


def test_binary_cross_entropy():
    '''
    Check NeuralNetwork._binary_cross_entropy using manual calculation
    '''
    network = nn.NeuralNetwork(nn_arch = [{"input_dim": 4, "output_dim": 1, "activation": "relu"}],
                           lr = 0.01, seed = 26, batch_size = 1, epochs = 1, loss_function = "mse")
    y = np.array([1, 0, 1, 0])
    y_hat = np.array([0.5, 0.01, 0.01, 0.9])
    assert np.round(network._binary_cross_entropy(y,y_hat),2) == 1.47


def test_binary_cross_entropy_backprop():
    pass


def test_mean_squared_error():
    '''
    Check NeuralNetwork._mean_squared_error using sklearn metrics mean squared error implementation 
    '''
    from sklearn.metrics import mean_squared_error
    network = nn.NeuralNetwork(nn_arch = [{"input_dim": 4, "output_dim": 1, "activation": "relu"}],
                           lr = 0.01, seed = 26, batch_size = 1, epochs = 1, loss_function = "mse")
    y = np.array([3, -0.5, 2, 7])
    y_hat = np.array([2.5, 0.0, 2, 8])
    assert network._mean_squared_error(y, y_hat) == mean_squared_error(y, y_hat)

def test_mean_squared_error_backprop():
    pass


def test_one_hot_encode():
    """
    Check preprocess.one_hot_encode_seqs using manual calculation
    """
    encoding = preprocess.one_hot_encode_seqs(['AGA'])
    assert np.allclose(encoding, np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]))


def test_sample_seqs():
    pass

test_single_forward()
test_single_backprop()
test_forward()
test_mean_squared_error()
test_binary_cross_entropy()
test_one_hot_encode()