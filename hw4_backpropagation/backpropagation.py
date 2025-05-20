import numpy as np

X = np.array([
    [1, 0, 0],
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]).T

d = np.array([0, 0, 1, 1]).reshape(1, 4)

W = np.random.randn(2, 3) * 0.01
V = np.random.randn(1, 2) * 0.01

lr = 0.1
max_epochs = 10000
z=np.zeros((1, 4))

for epoch in range(max_epochs):
    y = W @ X
    y_f = np.tanh(y)
    z = V @ y_f

    delta_z = z - d
    dE_dV = (delta_z @ y_f.T)

    delta_y = (V.T @ delta_z) * (1 - y_f ** 2)
    dE_dW = (delta_y @ X.T)

    V -= lr * dE_dV
    W -= lr * dE_dW

print("Output:", z)
print("target:", d)
print("error:", np.abs(z - d))
print("W:", W )
print("V:", V)