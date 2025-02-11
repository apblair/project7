# BMI 203 Project 7: Neural Network


# Importing Dependencies

import numpy as np
from typing import List, Dict, Tuple
from numpy.typing import ArrayLike


# Neural Network Class Definition
class NeuralNetwork:
    """
    This is a neural network class that generates a fully connected Neural Network.
    Parameters:
        nn_arch: List[Dict[str, float]]
            This list of dictionaries describes the fully connected layers of the artificial neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32}, {'input_dim': 32, 'output_dim': 8}] will generate a
            2 layer deep fully connected network with an input dimension of 64, a 32 dimension hidden layer
            and an 8 dimensional output.
        lr: float
            Learning Rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.
        epsilon: float
            Binary cross entropy epsilon parameter to prevent log(0)
    Attributes:
        arch: list of dicts
            This list of dictionaries describing the fully connected layers of the artificial neural network.
    """
    def __init__(self,
                 nn_arch: List[Dict[str, int]],
                 lr: float,
                 seed: int,
                 batch_size: int,
                 epochs: int,
                 loss_function: str,
                 epsilon= 1e-5):
        # Saving architecture
        self.arch = nn_arch
        # Saving hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size,
        self._epsilon = epsilon
        # Initializing the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD!! IT IS ALREADY COMPLETE!!
        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.
        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """
        # seeding numpy random
        np.random.seed(self._seed)
        # defining parameter dictionary
        param_dict = {}
        # initializing all layers in the NN
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            # initializing weight matrices
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            # initializing bias matrices
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1
        return param_dict

    def _select_function(self, function_type=None, activation=None, loss=None):
        """
        Return a function
        Args:
            function_type : str
                Name of activation function for current layer
            activation : str
                Name of activation function for current layer
            loss : str
                Name of activation function for current layer
        """
        if function_type == 'forward':
            if activation == "relu":
                return self._relu
            elif activation == "sigmoid":
                return self._sigmoid

        elif function_type == "backprop":
            if activation == "relu":
                return self._relu_backprop
            elif activation == "sigmoid":
                return self._sigmoid_backprop
        
        elif function_type == "loss":
            if loss == "mse":
                return self._mean_squared_error
            elif loss == "bce":
                return self._binary_cross_entropy
        
        elif function_type == 'back loss':
            if loss == "mse":
                return self._mean_squared_error_backprop
            elif loss == "bce":
                return self._binary_cross_entropy_backprop


    def _single_forward(self,
                        W_curr: ArrayLike,
                        b_curr: ArrayLike,
                        A_prev: ArrayLike,
                        activation: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.
        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.
        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        Z_curr = A_prev.dot(W_curr.T) + b_curr.T # linear transformation
        forward_activation_function = self._select_function(function_type='forward', activation=activation)
        A_curr = forward_activation_function(Z_curr)
        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.
        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].
        Returns:
            A_prev: ArrayLike
                Previous layer activation matrix.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        cache = {"A0":X} # initialize cache with input matrix at zero index
        A_prev = X
        for idx, layer in enumerate(self.arch):
            # Run forward propagation
            layer_idx = idx + 1
            A_curr, Z_curr = self._single_forward(self._param_dict['W'+str(layer_idx)], # W_curr
                                                self._param_dict['b'+str(layer_idx)], # b_curr
                                                A_prev, # A_prev
                                                layer['activation']) # activation
            # Store Z and A matrices in cache dictionary for use in backprop
            cache["A" + str(layer_idx)] = A_curr 
            cache["Z" + str(layer_idx)] = Z_curr
            A_prev = A_curr # set layer activation matrix
        return A_prev, cache

    def _single_backprop(self,
                         W_curr: ArrayLike,
                         b_curr: ArrayLike,
                         Z_curr: ArrayLike,
                         A_prev: ArrayLike,
                         dA_curr: ArrayLike,
                         activation_curr: str) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.
        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.
        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        backprop_activation_function = self._select_function(function_type='backprop', activation=activation_curr)
        dZ_curr = backprop_activation_function(dA_curr, Z_curr)
        dA_prev, dW_curr, db_curr = np.dot(dZ_curr, W_curr), np.dot(dZ_curr.T, A_prev), np.sum(dZ_curr, axis = 0).reshape(b_curr.shape)
        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.
        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.
        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        loss_function = self._select_function(function_type='back loss', loss=self._loss_func)
        dA_curr = loss_function(y, y_hat)
        grad_dict = {}
        for idx, layer in list(enumerate(self.arch))[::-1]:
            layer_idx = idx + 1
            dA_prev, dW_curr, db_curr = self._single_backprop(self._param_dict['W'+str(layer_idx)], # W_curr
                                                            self._param_dict['b'+str(layer_idx)], # b_curr
                                                            cache['Z'+ str(layer_idx)], # Z_curr
                                                            cache['A'+ str(layer_idx-1)], # A_prev n-1
                                                            dA_curr, # dA_curr
                                                            layer['activation']) # activation_curr
            grad_dict['dW' + str(layer_idx)] = dW_curr
            grad_dict['db' + str(layer_idx)] = db_curr
            grad_dict['dA_prev' + str(layer_idx)] = dA_prev
            dA_curr = dA_prev

        return grad_dict


    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and thus does not return anything
        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        Returns:
            None
        """
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            self._param_dict['W'+str(layer_idx)] = self._param_dict['W'+str(layer_idx)] - (self._lr * grad_dict['dW'+str(layer_idx)])
            self._param_dict['b'+str(layer_idx)] = self._param_dict['b'+str(layer_idx)] - (self._lr * grad_dict['db'+str(layer_idx)])

    def fit(self,
            X_train: ArrayLike,
            y_train: ArrayLike,
            X_val: ArrayLike,
            y_val: ArrayLike) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network via training for the number of epochs defined at
        the initialization of this class instance.
        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.
        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        per_epoch_loss_train = []
        per_epoch_loss_val = []
        number_of_batches = int(len(y_train) / self._batch_size[0])
        for epoch in range(self._epochs):

            # shuffle and subset training data
            shuffled_training_indices = np.random.permutation(len(y_train))
            X_batch_n_epoch = np.array_split(X_train[shuffled_training_indices], number_of_batches)
            y_batch_n_epoch = np.array_split(y_train[shuffled_training_indices], number_of_batches)
            
            # train model
            for X, y in zip(X_batch_n_epoch, y_batch_n_epoch):
                y_hat, cache = self.forward(X)
                grad_dict = self.backprop(y, y_hat, cache)
                self._update_params(grad_dict)
            
            # Predict for each epoch
            y_hat_train = self.predict(X_train)
            y_hat_val = self.predict(X_val)    
            
            # Compute training and validation loss for each epoch
            loss_function = self._select_function(function_type='loss', loss=self._loss_func)
            per_epoch_loss_train.append(loss_function(y_train, y_hat_train))
            per_epoch_loss_val.append(loss_function(y_val, y_hat_val))

        return per_epoch_loss_train, per_epoch_loss_val       
        

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network model.
        Args:
            X: ArrayLike
                Input data for prediction.
        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, cache = self.forward(X)
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.
        Args:
            Z: ArrayLike
                Output of layer linear transform.
        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = 1 / (1 + np.exp(-Z))
        return nl_transform

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.
        Args:
            Z: ArrayLike
                Output of layer linear transform.
        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = np.maximum(0, Z)
        return nl_transform

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.
        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.
        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = self._sigmoid(Z)*(1-self._sigmoid(Z))
        return dZ

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.
        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.
        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = (Z > 0).astype(int) * dA
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.
        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.
        Returns:
            loss: float
                Average loss over mini-batch.
        """
        loss = - np.average(y * np.log(y_hat + self._epsilon) + (1 - y) * np.log(1 - y_hat + self._epsilon)) # add epsilon to log function to prevent log(0)
        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative.
        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.
        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        dA = (y_hat - y)/(y_hat - y_hat**2)
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.
        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.
        Returns:
            loss: float
                Average loss of mini-batch.
        """
        loss = np.square(y-y_hat).mean()
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative.
        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.
        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        dA = 2 * (y_hat - y) / len(y)
        return dA

    def _loss_function(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Loss function, computes loss given y_hat and y. This function is
        here for the case where someone would want to write more loss
        functions than just binary cross entropy.
        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.
        Returns:
            loss: float
                Average loss of mini-batch.
        """
        pass

    def _loss_function_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        This function performs the derivative of the loss function with respect
        to the loss itself.
        Args:
            y (array-like): Ground truth output.
            y_hat (array-like): Predicted output.
        Returns:
            dA (array-like): partial derivative of loss with respect
                to A matrix.
        """
        pass