import numpy as np

X = np.array([
    [-3,-2],
    [-3,0],
    [-3,2],
    [-2,-1],
    [-2,0],
    [-2,1],
    [-1,0],
    [0,0],
    [1,0],
    [2,-1],
    [2,0],
    [2,1],
    [3,-2],
    [3,0],
    [3,2],
])

np.random.seed(1)
epsilon = 1e-2
max_iter = 100

def init_centers(X, n_clusters):
    indices = np.random.choice(len(X), n_clusters, replace=False)
    return X[indices]

def hcm(X, n_clusters=2, epsilon=1e-2, max_iter=100, alpha_c=0.3):
    centers = init_centers(X, n_clusters)
    prev_centers = np.copy(centers)
    
    for _ in range(max_iter):
        x = X[np.random.randint(0, len(X))]
        d = np.linalg.norm(x - centers, axis=1)
        j = np.argmin(d)
        centers[j] = centers[j] + alpha_c * (x - centers[j])
        if np.sum(np.linalg.norm(centers - prev_centers, axis=1)) < epsilon:
            break
        prev_centers = np.copy(centers)
    
    d = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
    u = np.zeros((len(X), n_clusters))
    u[np.arange(len(X)), np.argmin(d, axis=1)] = 1
    return u, centers

def fcm(X, n_clusters=2, m=2, epsilon=1e-2, max_iter=100):
    centers = init_centers(X, n_clusters)
    for _ in range(max_iter):
        d = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        zero_mask = (d == 0)
        u = np.zeros((len(X), n_clusters))
        for k in range(len(X)):
            if zero_mask[k].any():
                u[k, zero_mask[k]] = 1 / zero_mask[k].sum()
            else:
                u[k] = (1 / d[k] ** 2) ** (1 / (m - 1))
                u[k] /= u[k].sum()
        new_centers = np.array([
            np.sum((u[:, j] ** m)[:, np.newaxis] * X, axis=0) / np.sum(u[:, j] ** m)
            for j in range(n_clusters)
        ])
        if np.linalg.norm(new_centers - centers) < epsilon:
            break
        centers = new_centers
    return u, centers

hcm_u, hcm_centers = hcm(X)
fcm_u, fcm_centers = fcm(X)

print("HCM")
for k in range(len(X)):
    print(f"Sample {k + 1}: uk1 = {hcm_u[k, 0]}, uk2 = {hcm_u[k, 1]}")

print("\nFCM")
for k in range(len(X)):
    print(f"Sample {k + 1}: uk1 = {fcm_u[k, 0]:.2f}, uk2 = {fcm_u[k, 1]:.2f}")
