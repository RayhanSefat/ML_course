#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# ====================== Layer Class ======================
class Layer:
    """Represents a single fully-connected layer."""

    def __init__(self, input_size, output_size):
        # He initialization for ReLU
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))
        # Cache for backprop
        self.input = None
        self.z = None

    def forward(self, x):
        """Forward pass through this layer."""
        self.input = x
        self.z = np.dot(x, self.weights) + self.biases
        return self.z

    def backward(self, grad_output, learning_rate):
        """Backward pass: update weights and biases."""
        grad_input = np.dot(grad_output, self.weights.T)
        grad_w = np.dot(self.input.T, grad_output)
        grad_b = np.sum(grad_output, axis=0, keepdims=True)
        # Gradient descent update
        self.weights -= learning_rate * grad_w
        self.biases -= learning_rate * grad_b
        return grad_input


# ====================== Neural Network Class ======================
class NeuralNetwork:
    """Implements a simple feedforward neural network."""

    def __init__(self, layer_sizes, learning_rate=0.001):
        self.layers = []
        self.learning_rate = learning_rate
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

    # ---- Activation functions ----
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    # ---- Forward pass ----
    def forward(self, x):
        activations = [x]
        for i, layer in enumerate(self.layers[:-1]):
            z = layer.forward(activations[-1])
            a = self.relu(z)
            activations.append(a)
        # Last layer: linear
        z_final = self.layers[-1].forward(activations[-1])
        activations.append(z_final)
        return activations

    # ---- Backward pass ----
    def backward(self, activations, y_true):
        y_pred = activations[-1]
        dloss = 2 * (y_pred - y_true) / len(y_true)
        grad = dloss

        # Output layer
        grad = self.layers[-1].backward(grad, self.learning_rate)

        # Hidden layers
        for i in range(len(self.layers) - 2, -1, -1):
            grad = grad * self.relu_derivative(self.layers[i].z)
            grad = self.layers[i].backward(grad, self.learning_rate)

    # ---- Training ----
    def train(self, X_train, y_train, X_test, y_test, epochs=1000, log_interval=100):
        log = []
        for epoch in range(1, epochs + 1):
            activations = self.forward(X_train)
            y_pred = activations[-1]
            loss = np.mean((y_pred - y_train) ** 2)
            self.backward(activations, y_train)

            if epoch % log_interval == 0 or epoch == 1:
                y_test_pred = self.predict(X_test)
                test_loss = np.mean((y_test_pred - y_test) ** 2)
                r2 = r2_score(y_test, y_test_pred)
                line = f"Epoch {epoch}: Train Loss={loss:.4f}, Test Loss={test_loss:.4f}, R2={r2:.4f}"
                print(line)
                log.append(line)
        return log

    # ---- Prediction ----
    def predict(self, X):
        activations = self.forward(X)
        return activations[-1]


# ====================== MAIN SCRIPT ======================
if __name__ == "__main__":
    # Configurable parameters
    DATA_FILE = "./synthetic_dataset.csv"
    TRAIN_SPLIT = 0.8
    HIDDEN_LAYERS = [16, 8]  # example structure
    EPOCHS = 2000
    LEARNING_RATE = 0.001
    RESULT_FILE = "results.txt"
    np.random.seed(42)

    # ---- Load and prepare data ----
    df = pd.read_csv(DATA_FILE)
    X = df[["x1", "x2", "x3", "x4", "x5"]].values
    y = df["y"].values.reshape(-1, 1)

    # Normalize
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X = (X - X_mean) / (X_std + 1e-8)

    # Split
    N = len(X)
    idx = np.arange(N)
    np.random.shuffle(idx)
    train_size = int(TRAIN_SPLIT * N)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # ---- Initialize and train network ----
    layer_sizes = [X.shape[1]] + HIDDEN_LAYERS + [1]
    nn = NeuralNetwork(layer_sizes, learning_rate=LEARNING_RATE)

    log = nn.train(X_train, y_train, X_test, y_test, epochs=EPOCHS)

    # ---- Save logs ----
    with open(RESULT_FILE, "w") as f:
        f.write("OOP Manual Neural Network Training Log\n")
        f.write(f"Hidden Layers: {HIDDEN_LAYERS}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Train-Test Split: {TRAIN_SPLIT}\n\n")
        for line in log:
            f.write(line + "\n")

    print(f"\nTraining complete. Results saved to {RESULT_FILE}")

    # ---- Predictions ----
    y_pred = nn.predict(X_test)

    # ---- Plots ----
    # 1. Real vs Predicted y
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Real y")
    plt.ylabel("Predicted y")
    plt.title("Real vs Predicted y")
    plt.grid(True)
    plt.savefig("real_vs_predicted_y.png", dpi=150)
    plt.close()

    # 2. Xi vs y
    for i, feature in enumerate(["x1", "x2", "x3", "x4", "x5"]):
        plt.figure(figsize=(6,4))
        plt.scatter(X_test[:, i], y_test, label="Real y", alpha=0.6)
        plt.scatter(X_test[:, i], y_pred, label="Predicted y", alpha=0.6)
        plt.xlabel(feature)
        plt.ylabel("y")
        plt.title(f"{feature} vs y (Real vs Predicted)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{feature}_vs_y.png", dpi=150)
        plt.close()

    print("Plots saved: real_vs_predicted_y.png and xᵢ_vs_y.png files.")
