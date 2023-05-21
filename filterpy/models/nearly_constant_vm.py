import numpy as np

class TransitionModel():
    def __init__(self, dt, q):
        self.dt = dt  # Time step
        self.cov = q  # Process noise covariance

    @property
    def mean(self):
        F = np.array([[1, self.dt],
                      [0, 1]])
        return F

    def predict(self, x):
        """
        Predict function for a nearly-constant velocity model.
        We assume that the control input (G) is 0. 

        Predict the state at time t + 1.
        """
        return self.mean @ x + np.random.multivariate_normal([0, 0], self.cov)


class MeasurementModel():
    def __init__(self, r):
        self.cov = r  # Measurement noise covariance

    @property
    def mean(self):
        H = np.array([[1, 0]])
        return H

    def measure(self, x):
        """
        Measure the state at time t.
        """
        return self.mean @ x + np.random.normal(0, np.sqrt(self.cov))

    def lpdf(self, x, y):
        lpdf = np.zeros(x.shape[0])

        for i in range(x.shape[0]):
            lpdf[i] = -0.5 * np.log(2 * np.pi * self.cov) - 0.5 * (y - self.mean * x[i]) ** 2 / self.cov
        
        return lpdf
