import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from filterpy.kf.basic import BasicKalmanFilter
from filterpy.models import nearly_constant_vm as ncvm

sns.set_style("whitegrid")

# Hyperparameters
# Initialize models
transition_model = ncvm.TransitionModel(dt=0.1, q=np.diag([0.01, 0.01]))
measurement_model = ncvm.MeasurementModel(r=0.1)

# Simulate data
T = 100  # Number of time steps
x_true = np.zeros((T, 2))  # True states
y = np.zeros(T)  # Measurements

x_true[0] = np.array([0, 0])  # Initial state

for t in range(1, T):
    x_true[t] = transition_model.predict(x_true[t-1])
    y[t] = measurement_model.measure(x_true[t])

# Instantiate the Kalman filter
kf = BasicKalmanFilter(transition_model, measurement_model)

# Initialise the state and state covariance
x_hat = np.zeros((T, 2))
P = np.zeros((T, 2, 2))

# Set the initial state and state covariance
x_hat[0] = np.random.multivariate_normal(np.zeros(2), np.eye(2))
P[0] = np.array([[0.5, 0], [0, 0.5]])

# Run the Kalman filter
for t in range(1, T):
    x_pred, P_pred = kf.predict(x_hat[t-1], P[t-1])
    x_hat[t], P[t] = kf.update(x_pred, P_pred, y[t])

# Calculate the mean squared error
mse = np.mean((x_true - x_hat) ** 2)
print(f"Mean Squared Error: {mse:.2f}")
