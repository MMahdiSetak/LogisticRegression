import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))
    return sig


def loss(predicted_y, y):
    return (-y * np.log(predicted_y) - (1 - y) * np.log(1 - predicted_y)).mean()


def predict_prob(x, w):
    return sigmoid(np.dot(x, w))


def fit(x, y):
    x = np.hstack((x, np.ones((x.shape[0], 1))))
    w = np.random.uniform(size=(x.shape[1]))
    learning_rate = 0.005

    for i in range(1_000_000):
        predicted_y = predict_prob(x, w)
        gradient = np.dot(x.T, (predicted_y - y)) / y.size
        w -= learning_rate * gradient
        if i % 100 == 99:
            print(f'loss: {loss(predicted_y, y)} \t')
    return w


data = pd.read_csv("data.csv").to_numpy()[2:]

X = data[:, :-1].astype(np.float64)
converter = lambda x: 1 - (x == 'C1')
Y = converter(data[:, -1])

w = fit(X, Y)

colors = ['blue' if label == 0 else 'red' for label in Y]
plt.scatter(X[:, 0], X[:, 1], c=colors)
x0_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 300)

# Calculate the corresponding x1 values for the decision boundary
boundary = -(w[2] + w[0] * x0_values) / w[1]

# Plot the decision boundary
plt.plot(x0_values, boundary, 'green')
plt.show()
