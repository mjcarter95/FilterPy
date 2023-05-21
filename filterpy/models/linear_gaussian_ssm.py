import numpy as np


class TransitionModel():
    def __init__(self, F, Q):
        self.F = F  # State transition matrix
        self.Q = Q  # State covariance matrix

    def predict(self, x):
        """
        Predict the state at time t + 1.
        """
        return self.F @ x + np.random.normal(0, np.sqrt(self.Q))


class MeasurementModel():
    def __init__(self, H, R):
        self.H = H  # Observation matrix
        self.R = R  # Measurement covariance matrix

    def measure(self, x):
        """
        Measure the state at time t.
        """
        return self.H @ x + np.random.normal(0, np.sqrt(self.R))

    def lpdf(self, x, y):
        lpdf = np.zeros(x.shape[0])

        for i in range(x.shape[0]):
            lpdf[i] = -0.5 * np.log(2 * np.pi * self.R) - 0.5 * (y - self.H * x[i]) ** 2 / self.R
        
        return lpdf


def simulate_data(T, transition_model, measurement_model):
    """
    Simulate a linear Gaussian state space model.
    """

    # Initialise the state and observation sequences
    x = np.zeros((T, 1))
    y = np.zeros((T, 1))

    # Set the initial state
    x[0] = np.random.normal(0, 1)
    y[0] = measurement_model.measure(x[0])

    # Simulate the state and observation sequences
    for t in range(1, T):
        x[t] = transition_model.predict(x[t - 1])
        y[t] = measurement_model.measure(x[t])

    return x, y