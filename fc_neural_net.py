import torch.nn as nn
import torchvision.datasets as datasets
import numpy as np
from sklearn.metrics import accuracy_score
"""Fully parameterizable neural network

"""
class MLP:
    """ Initialise the multi-layer neural network class with required parameters.

        Args:
            inputs: Number of input neurons.
            hidden_layers_list: List of hidden layers with number of neurons.
            outputs: Number of output neurons.
            act_functs: List of activation functions for each layer.

        Attributes:
            inputs: Number of input neurons.
            hidden_layers_list = List of hidden layers with number of neurons.
            self.outputs = Number of output neurons.
            self.weights = List of weights between layers.
            self.biases = List of biasses between layers.
            self.total_layers = Total numebr of layers in the architecture.
            self.activations = List of outputs from each layer.
            self.derivatives = List of derivatives of weights for each layer.
            self.bias_derivatives = List of derivatives of biasses for each layer.
            self.loss_history = History of training loss.
            self.act_functs = List of activation functions for each layer.
            self.v_dw = List of momentum for weights.
            self.v_db = List of momentum for biasses.
            self.s_dw = List of Rms prop for weights.
            self.s_db = List of Rms prop for biasses.

        """
    def __init__ (self, inputs=2, hidden_layers_list=[2], outputs=1, act_functs=["sigmoid", "sigmoid"]):

        self.inputs = inputs
        self.hidden_layers_list = hidden_layers_list
        self.outputs = outputs
        self.weights = []
        self.biases = []
        self.total_layers = 2 + len(self.hidden_layers_list)
        self.activations = []
        self.derivatives = []
        self.bias_derivatives = []
        self.loss_history = []
        self.act_functs = act_functs
        self.v_dw = []
        self.v_db = []
        self.s_dw = []
        self.s_db = []

        layers = [self.inputs] + self.hidden_layers_list + [self.outputs]

        # Create the list of weights
        for i in range(self.total_layers - 1):
            layer_weight = np.random.normal(size=(layers[i], layers[i+1]))/np.sqrt(layers[i])
            self.weights.append(layer_weight)

        # Create the list of weight derivatives
        for i in range(self.total_layers - 1):
            derivative = np.zeros((layers[i], layers[i+1]))
            self.derivatives.append(derivative)

        # Create the list of biasses
        for i in range(self.total_layers - 1):
            bias = np.random.normal(size = layers[i+1])
            self.biases.append(bias)

        # Create the list of bias derivatives
        for i in range(self.total_layers - 1):
            bias_derivative = np.zeros(layers[i+1])
            self.bias_derivatives.append(bias_derivative)

        # Create list of activations
        for i in range(self.total_layers):
            activation = np.zeros(layers[i])
            self.activations.append(activation)

        # Create lists for adam optimizer
        self.v_dw = self.derivatives.copy()
        self.v_db = self.bias_derivatives.copy()
        self.s_dw = self.derivatives.copy()
        self.s_db = self.bias_derivatives.copy()

    def forward(self, X):
        """Forward propagation of the network.

        Args:
            X: Inputs dataset

        Returns:
            Network output as probability distribution.

        """

        activation = X
        self.activations[0] = activation

        layer = 1
        for w, b in zip(self.weights, self.biases):

            next_layer_input = np.dot(activation, w) + b

            if layer == self.total_layers - 1:
                activation = self.softmax(next_layer_input)
            elif self.act_functs[layer-1] == "sigmoid":
                activation = self.sigmoid(next_layer_input)
            elif self.act_functs[layer-1] == "relu":
                activation = self.relu(next_layer_input)
            self.activations[layer] = activation
            layer +=1

        return activation

    def back_propagate(self, y):
        """Forward propagation of the network.

        Args:
            y: The labels.

        Returns:
            None. Updates bias and weight derivatives.

        """
        error = 1
        for i in reversed(range(self.total_layers - 1)):
            activation = self.activations[i+1]
            if i == self.total_layers - 2:
                delta = activation - y
            elif self.act_functs[i] == "sigmoid":
                delta = error * self.sigmoid_derivative(activation)
            elif self.act_functs[i] == "relu":
                delta = error * self.relu_derivative(activation)
            current_layer_act = self.activations[i]
            self.derivatives[i] = (current_layer_act.T).dot(delta)
            self.bias_derivatives[i] = delta.sum(axis=0)

            error = np.dot(delta, self.weights[i].T)


    def g_d(self, lr):
        """Update the weights with gradient descent.

        Args:
            lr: Learning rate.

        Returns:
            None. Updates biasses and weights arrays.

        """
        for i in range(len(self.derivatives)):
            self.weights[i] -= lr * (self.derivatives[i])
            # With weight decay regularization
            #self.weights[i] -= lr * (self.derivatives[i]) - (self.weights[i] * lr /10000 )
            self.biases[i] -= lr * (self.bias_derivatives[i])

    def adam(self, lr, t):
        """Update the weights with adam optimizer

        Args:
            lr: Learning rate.
            t: Batch number.

        Returns:
            None. Updates biasses and weights arrays with adam optimizer.

        """
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-8

        for i in range(len(self.derivatives)):

            # momentum
            self.v_dw[i] = beta_1 * self.v_dw[i] + (1 - beta_1) * self.derivatives[i]
            self.v_db[i] = beta_1 * self.v_db[i] + (1 - beta_1) * self.bias_derivatives[i]

            # rms prop
            self.s_dw[i] = beta_2 * self.s_dw[i] + (1 - beta_2) * (self.derivatives[i] ** 2)
            self.s_db[i] = beta_2 * self.s_db[i] + (1 - beta_2) * (self.bias_derivatives[i] ** 2)

            # bias corrections
            self.v_dw_corr = self.v_dw[i] / (1 - (beta_1 ** t))
            self.v_db_corr = self.v_db[i] / (1 - (beta_1 ** t))
            self.s_dw_corr = self.s_dw[i] / (1 - (beta_2 ** t))
            self.s_db_corr = self.s_db[i] / (1 - (beta_2 ** t))
            
            # weights & biases update
            self.weights[i] -= lr * (self.v_dw_corr / (np.sqrt(self.s_dw_corr) + epsilon))
            self.biases[i] -= lr * (self.v_db_corr / (np.sqrt(self.s_db_corr) + epsilon))

    def get_minibatches(self, X, y, no_batches):
        """Split the data into multiple batches

        Args:
            X: Input dataset.
            y: Labels.
            no_batches: Number of required batches.

        Returns:
            X and y minibatches.

        """
        X_minibatch = np.split(X, no_batches)
        y_minibatch = np.split(y, no_batches)
        return X_minibatch, y_minibatch

    def fit(self, X, y, X_test, y_test, epochs=50, lr=0.01, batches=50, opt="adam", stop = [True, 5]):
        """Fit the network to the dataset and start training.

        Args:
            X: Input dataset.
            y: Labels.
            X_test: Test inputs.
            y_test: Test labels.
            epochs: Number of desired epochs for training.
            lr: Learning rate.
            batches: Number of required mini-batches.
            opt: If adam optimizer is desired. Without adam, SGD is used.
            stop: If early stop is required and how many epochs patience desired.

        Returns:
            loss_history and loss_history_epochs.

        """
        loss_history = []
        loss_history_epochs = []
        X_mini_batch, y_mini_batch= self.get_minibatches(X, y, batches)
        max_accuracy = 0
        early_stop_counter = 0
        for epoch in range(epochs):
            for t, (X, y) in enumerate(zip(X_mini_batch, y_mini_batch), start=1):
                loss = self.train(X, y, lr=lr, opt = opt, t=t)
                loss_history.append(loss)
            epoch_loss =  round(loss_history[-1],4)
            loss_history_epochs.append(epoch_loss)
            train_accuracy = accuracy_score(self.predict(X_train), y_train_labels)
            val_accuracy = accuracy_score(self.predict(X_valid), y_valid)
            test_accuracy = accuracy_score(self.predict(X_test), y_test)
            # train_accuracy = self.accuracy(X_train, y_train_labels)
            # val_accuracy = self.accuracy(X_valid, y_valid)
            # test_accuracy = self.accuracy(X_test, y_test)
            print(f"Epoch:{epoch+1} loss: {epoch_loss} Train Accuracy: {train_accuracy} Validation Accuracy: {val_accuracy} Test Accuracy:{test_accuracy}")
            if stop[0]:
                if val_accuracy > max_accuracy:
                    max_accuracy = val_accuracy
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter == stop[1]:
                        print(f"Stopping: Accuracy not increased for the last {stop[1]} epochs")
                        break
        return loss_history, loss_history_epochs

    def train(self, X_train, y_train, lr, opt, t):
        """Train the network

        Args:
            X_train: Training inputs.
            y_train: Training labels.
            lr: Learning rate.
            opt: If adam optimizer is desired. Without adam, SGD is used.
            t: batch number.

        Returns:
            Training loss.

        """
        output = self.forward(X_train)
        self.back_propagate(y_train)
        if opt == "adam":
            self.adam(lr=lr, t=t)
        else:
            self.g_d(lr=lr)
        return self.loss(output, y_train)

    def predict(self, X):
        """Make predictions with the network.

        Args:
            X_train: Input data.

        Returns:
            Predicted labels.

        """
        y_pred = np.argmax(self.forward(X), axis=1)
        return y_pred

    def accuracy(self, X_test, y_test):
        """Make predictions with the network.

        Args:
            X_test: Test inputs.
            y_test: Test labels.

        Returns:
            Prediction accuracy.

        """
        y_pred = np.argmax(self.forward(X_test), axis=1)
        accuracy = np.mean(y_pred == y_test)
        return accuracy

    def relu(self, x):
        """Apply relu activation function.

        Args:
            x: inputs.

        Returns:
            Output of relu function.

        """
        return np.maximum(0,x)

    def relu_derivative(self, x):

        """Calculate derivative of relu function.

        Args:
            x: inputs.

        Returns:
            Output of relu derivative.

        """
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def sigmoid(self, x):
        """Apply sigmoid function.

        Args:
            x: inputs.
  
        Returns:
            Output of sigmoid function.

        """
        y = 1.0 / (1 + np.exp(-x))
        return y

    def sigmoid_derivative(self, x):

        return x * (1 - x)

    def softmax(self, multiplication):
        """Apply softmax activation function.

        Args:
            multiplication: inputs to softmax.

        Returns:
            Probability distribution from softmax.

        """
        return np.exp(multiplication) / np.sum(np.exp(multiplication), axis=1, keepdims=True)

    def loss(self, output, y):
        """Calculate the loss.

        Args:
            output: output from network.
            y: actual labels.

        Returns:
            Loss

        """
        loss = np.mean(-y * np.log(output) - (1.0-y) * np.log(1.0 - output))
        return loss

# Download the data.
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

# Convert to numpy arrays
X_train = mnist_trainset.data.numpy()
X_train = X_train.reshape(X_train.shape[0], -1)
y_train = nn.functional.one_hot(mnist_trainset.targets).numpy()
y_train_labels = (mnist_trainset.targets).numpy()

# Shuffle the data
rng = np.random.default_rng()
indices = np.arange(X_train.shape[0])
rng.shuffle(indices)

X_train = X_train[indices]
y_train = y_train[indices]
y_train_labels = y_train_labels[indices]

#Create validation and test sets
X_valid = X_train[:10000]
y_valid = y_train_labels[:10000]
X_train = X_train[10000:60000]
y_train = y_train[10000:60000]
y_train_labels = y_train_labels[10000:60000]

X_test = mnist_testset.data.numpy()
X_test= X_test.reshape(X_test.shape[0], -1)
y_test = mnist_testset.targets.numpy()

# Scale the image data by 255
X_train = X_train / 255
X_test = X_test / 255

class normalize():
    """Normalize with mean and standard dev.

        Returns:
            Normalized data.

        """
    def __init__(self):
        self.mean = 0
        self.std = 0

    def fit(self,X):
        self.mean = np.mean(X)
        self.std = np.std(X)
        print(f"Mean: {self.mean} - Std :{self.std}")

    def transform(self, X):
        X_normalized = (X - self.mean) / self.std
        return X_normalized

normalization = normalize()
normalization.fit(X_train)

X_train = normalization.transform(X_train)
X_test = normalization.transform(X_test)
X_valid = normalization.transform(X_valid)

mlp = MLP(784, [128,128], 10, act_functs=["sigmoid", "sigmoid"])
loss_history = mlp.fit(X_train, y_train, X_test, y_test, epochs=50, lr=0.002, batches=50, opt="adam", stop=[True,5])
