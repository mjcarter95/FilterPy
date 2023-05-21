import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from filterpy.kf.basic import BasicKalmanFilter
from filterpy.models import linear_gaussian_ssm as lgssm

sns.set_style("whitegrid")

# Hyperparameters
T = 200
D = 1
H_true = np.array([[1.0]])
R_true = np.array([[1.0]])
F_true = np.array([[1.0]])
Q_true = np.array([[1.0]])

# Instantiate the measurement and observation models
transition_model = lgssm.TransitionModel(F_true, Q_true)
measurement_model = lgssm.MeasurementModel(H_true, R_true)

# Simulate the state and observation sequences
x, y = lgssm.simulate_data(T, transition_model, measurement_model)

# Plot the state and observation sequences as scatter
fig = plt.figure(figsize=(10, 5))
plt.plot(x, label="State")
plt.plot(y, label="Observation")
plt.title("Linear Gaussian State Space Model")
plt.tight_layout()
plt.savefig("lgssm.png")

# Instantiate the Kalman filter
H = np.array([[1.0]])
R = np.array([[1.0]])
F = np.array([[1.0]])
Q = np.array([[1.0]])
kf = BasicKalmanFilter(transition_model, measurement_model)

# Initialise the state and state covariance
x_hat = np.zeros((T, D))
P = np.zeros((T, D))

# Set the initial state and state covariance
x_hat[0] = np.random.multivariate_normal(np.zeros(D), np.eye(D))
P[0] = np.abs( np.random.multivariate_normal(np.zeros(D), np.eye(D)) )

# Run the Kalman filter
for t in range(1, T):
    x_pred, P_pred = kf.predict(x_hat[t-1], P[t-1])
    x_hat[t], P[t] = kf.update(x_pred, P_pred, y[t])

# Plot the state and observation sequences
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
plt.plot(x, label="State", linestyle="--", alpha=0.5, color="red")
plt.plot(y, label="Observation", linestyle="--", alpha=0.5, color="blue")
plt.plot(x_hat, label="Estimate", color="black")
plt.fill_between(
    np.arange(T),
    x_hat[:, 0] - 2 * np.sqrt(P[:, 0]),
    x_hat[:, 0] + 2 * np.sqrt(P[:, 0]),
    alpha=0.2,
    color="black",
)
plt.title("Kalman Filter")
plt.legend(loc="lower right")
ax.set_xlabel("Time")
ax.set_ylabel("State")
plt.tight_layout()
plt.savefig("kalman_filter.png")

# Calculate the mean squared error
mse = np.mean((x - x_hat) ** 2)
print(f"Mean Squared Error: {mse:.2f}")