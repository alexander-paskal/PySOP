import numpy as np


class NN:
    """
    A simple neural network with an input layer, one hidden layer and an output layer

    value throughput defined as:

        N(x) = C*activation(AX+B) + D


    where X is input vector with dims InputDim x M, M is the batch dimension
    C is weight matrix with dims
    D is bias matrix with dims HiddenDim x M
    A is weight matrix with dims HiddenDim x OutputDim
    B is weight matrix with dims OutputDim x M
    activation is tanh


    loss function is mean squared error
    """

    def __init__(self, input_dim, output_dim, hidden_dim=10, alpha=0.01):
        self.w1 = np.random.random((hidden_dim, input_dim))
        self.b1 = np.random.random((hidden_dim, 1))
        self.w2 = np.random.random((output_dim, hidden_dim))
        self.b2 = np.random.random((output_dim, 1))
        self.activation = np.tanh
        self.alpha = alpha

        self.vals = {}

    def step(self, X, y):
        if X.ndim < 2:
            X = X[:,None]

        prediction = self.forward(X)
        loss = self.loss(prediction, y)
        self.backward(X, y)

        return loss

    def forward(self, X):
        if X.ndim < 2:
            X = X[:,None]
        z1 = np.matmul(self.w1, X) + self.b1
        a1 = self.activation(z1)
        z2 = np.matmul(self.w2, a1) + self.b2
        a2 = self.activation(z2)
        self.vals["z1"] = z1
        self.vals["a1"] = a1
        self.vals["z2"] = z2
        self.vals["a2"] = a2
        return a2

    def backward(self, X, y):
        z1 = self.vals["z1"]
        a1 = self.vals["a1"]
        z2 = self.vals["z2"]
        a2 = self.vals["a2"]

        dz2dw2 = a1  # 1x1
        dz2db2 = 1  # 1x1
        dz2da1 = self.w2.T  # 10x1

        da1dz1 = 1 - np.tanh(z1) **2  # 10x1
        dz1dw1 = X  # 10x1
        dz1db1 = 1  # 10x1

        dLdz2 = 2*np.sum(z2 - y)

        dw2 = dz2dw2 * dLdz2
        db2 = dz2db2 * dLdz2

        dw1 = dLdz2 * np.matmul(dz2da1, da1dz1.T) * dz1dw1
        db1 = dLdz2 * dz2da1 * da1dz1 * dz1db1

        self.w2 -= self.alpha*dw2
        self.b2 -= self.alpha*db2

        self.w1 -= self.alpha * dw1
        self.b1 -= self.alpha * db1

    def loss(self, prediction, label):
        """
        Sum Squared Error Loss
        :param prediction:
        :param label:
        :return:
        """
        sse = np.sum((prediction - label)**2)
        return sse


if __name__ == '__main__':
    nn = NN(10, 1)
    X = np.random.random((10, 1))
    y = 0.6
    nn.step(X, y)