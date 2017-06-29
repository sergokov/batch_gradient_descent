import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# Load dataset
diabetes = datasets.load_diabetes()


# Use only one feature
X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/validation sets
X_train = X[:-20]
X_val = X[-20:]

# Split the targets into training/validation sets
Y_train = diabetes.target[:-20]
Y_val = diabetes.target[-20:]

X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1), dtype=X_train.dtype)))


def calc_loss(X, Y, betta):
    return np.sum((X.dot(betta) - Y) ** 2) / len(X)


def calc_gradient_step(X, Y, beta, learning_rate):
    hypothesis = X.dot(beta)
    loss = hypothesis - Y
    gradient = X.T.dot(loss) / len(X)
    beta -= learning_rate * gradient
    return beta


def plot_results(x_train, y_train, x_val, y_val, m, b):
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.scatter(x_train, y_train,  color='black')
    plt.plot(x_train, m * x_train + b, color='blue', linewidth=3)
    plt.subplot(122)
    plt.scatter(x_val, y_val,  color='black')
    plt.plot(x_val, m * x_val + b, color='blue', linewidth=3)
    plt.show()


if __name__ == "__main__":
    betta = np.array([random.randint(-100, 100), random.randint(-100, 100)], dtype=np.float)
    learning_rate = 0.1
    epochs = 20000

    for epoch in xrange(epochs):
       betta = calc_gradient_step(X_train, Y_train, betta, learning_rate)
       loss = calc_loss(X_train, Y_train, betta)
       print 'For epoch mumber: %f, m param: %f, b param: %f, Loss: %f' % (epoch, betta[0], betta[1], loss)
       if epoch % 2000 == 0:
           plot_results(X_train[:,0], Y_train, X_val[:,0], Y_val, betta[0], betta[1])

    plot_results(X_train[:,0], Y_train, X_val[:,0], Y_val, betta[0], betta[1])
