import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

from tqdm import tqdm
from filterpy.pf.basic import BasicParticleFilter
from filterpy.models import linear_gaussian_ssm as lgssm

sns.set_style("whitegrid")

# Hyperparameters
T = 200
D = 1
H = np.array([[1.0]])
R = np.array([[1.0]])
F = np.array([[1.0]])
Q = np.array([[1.0]])

# Instantiate the measurement and observation models
transition_model = lgssm.TransitionModel(F, Q)
measurement_model = lgssm.MeasurementModel(H, R)

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
N = 2000
pf = BasicParticleFilter(N, transition_model, measurement_model)

# Initialise the state and state covariance
x_sim = np.zeros((T + 1, N, D))
y_sim = np.zeros((T + 1, N, D))
logw = np.zeros((T + 1, N))

# Set the initial state and state covariance
x_sim[0, :, :] = np.random.multivariate_normal(np.zeros(D), np.eye(D), N)
for i in range(N):
    y_sim[0, i, :] = measurement_model.measure(x_sim[0, i, :])
logw[0, :] = measurement_model.lpdf(x_sim[0], y[0])

# Run the Particle filter
for t in tqdm(range(0, T)):
    x_pred, y_pred = pf.predict(x_sim[t])
    x_sim[t+1], y_sim[t+1], logw[t+1] = pf.update(x_pred, y_pred, logw[t], y[t])

# Estimate the state and observation sequences
x_hat = np.zeros((T, D))
y_hat = np.zeros((T, D))
for t in range(T):
    x_hat[t] = np.mean(x_sim[t+1], axis=0)
    y_hat[t] = np.mean(y_sim[t+1], axis=0)

# Plot the state and observation sequences
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
plt.plot(x, label="State", linestyle="--", alpha=0.5, color="red")
plt.plot(y, label="Observation", linestyle="--", alpha=0.5, color="blue")
plt.plot(x_hat, label="State Estimate", color="black")
plt.plot(y_hat, label="Observation Estimate", color="green")
plt.title("Particle Filter")
plt.legend(loc="lower right")
ax.set_xlabel("Time")
ax.set_ylabel("State")
plt.tight_layout()
plt.savefig("particle_filter.png")

# Calculate the mean squared error
mse = np.mean((x - x_hat) ** 2)
print(f"Mean Squared Error: {mse:.2f}")
