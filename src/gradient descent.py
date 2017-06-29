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


def calc_loss(x, y, m, b):
    loss = 0
    for index in range(len(x)):
        loss += (y[index] - (m * x[index] + b)) ** 2
        loss /= float(len(x))

    return loss


def calc_gradient_step(x, y, m, b, learning_rate):
    partial_m = 0
    partial_b = 0
    for inx in range(len(x)):
        partial_m += -x[inx] * (y[inx] - (m * x[inx] + b))
        partial_b += -(y[inx] - (m * x[inx] + b))
    partial_m = 2 * partial_m / float(len(x))
    partial_b = 2 * partial_b / float(len(x))
    m += -learning_rate * partial_m
    b += -learning_rate * partial_b

    return m, b


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
    m = random.randint(-100, 100)
    b = random.randint(-100, 100)
    learning_rate = 0.1
    epochs = 10000

    for epoch in xrange(epochs):
        m, b = calc_gradient_step(X_train, Y_train, m, b, learning_rate)
        print 'For epoch mumber: %f, m param: %f, b param: %f, Loss: %f' % (epoch, m, b, calc_loss(X_train, Y_train, m, b))
        if epoch % 2000 == 0:
            plot_results(X_train, Y_train, X_val, Y_val, m, b)

    plot_results(X_train, Y_train, X_val, Y_val, m, b)
