import numpy as np
import matplotlib.pyplot as plt

# === ORIGINAL DATA ===
X_orig = np.array([[150,100],[159,200],[170,350],[175,400],[179,500],[180,180],
                   [189,159],[199,110],[199,400],[199,230],[235,120],[239,340],
                   [239,360],[249,145],[249,400]], dtype=np.float64)

Y_orig = np.array([0.73,1.39,2.03,1.45,1.82,1.32,0.83,0.53,1.95,1.27,
                   0.49,1.03,1.24,0.55,1.3], dtype=np.float64).reshape(-1,1)

# === SYNTHETIC DATA GENERATION ===
X_min, X_max = X_orig.min(axis=0), X_orig.max(axis=0)
true_W = np.array([[0.01], [0.005]])
true_b = 0.2

n_samples = 1000
X_synth = np.random.uniform(low=X_min, high=X_max, size=(n_samples, 2))
noise = np.random.normal(0, 0.3, size=(n_samples,1))
Y_synth = X_synth @ true_W + true_b + noise

# === COMBINE DATA ===
X_combined = np.vstack((X_orig, X_synth))
Y_combined = np.vstack((Y_orig, Y_synth))

# === NORMALIZATION ===
X_mean = X_combined.mean(axis=0)
X_std = X_combined.std(axis=0)
X_norm = (X_combined - X_mean) / X_std

# === GRADIENT DESCENT TRAINING ===

m, n = X_norm.shape
W = np.zeros((n, 1))
b = 0.0
alpha = 0.01
epochs = 10000

for i in range(epochs):
    Y_pred = X_norm @ W + b
    error = Y_pred - Y_combined
    dw = (2/m) * (X_norm.T @ error)
    db = (2/m) * np.sum(error)
    W -= alpha * dw
    b -= alpha * db
    if i % 1000 == 0:
        loss = np.mean(error**2)
        print(f"Epoch {i}, Loss: {loss:.6f}")

print("Final Weights:", W.flatten())
print("Final Bias:", b)

# === 3D PLOTTING ===
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of normalized data
ax.scatter(X_norm[:, 0], X_norm[:, 1], Y_combined.flatten(), color='blue', alpha=0.5, label='Data points')

# Grid for surface plot (also in normalized space)
x1_grid, x2_grid = np.meshgrid(
    np.linspace(X_norm[:,0].min(), X_norm[:,0].max(), 30),
    np.linspace(X_norm[:,1].min(), X_norm[:,1].max(), 30)
)

# Compute Z = prediction on grid
Z = W[0] * x1_grid + W[1] * x2_grid + b

# Regression plane
ax.plot_surface(x1_grid, x2_grid, Z, color='red', alpha=0.5)

# Labels
ax.set_xlabel('Feature 1 (normalized)')
ax.set_ylabel('Feature 2 (normalized)')
ax.set_zlabel('Target (Y)')
ax.set_title('3D Linear Regression Fit')
plt.legend()
plt.show()
