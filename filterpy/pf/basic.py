import numpy as np

from filterpy.pf.base import BaseParticleFilter


class BasicParticleFilter(BaseParticleFilter):
    def __init__(self, N, transition_model, measurement_model):
        BaseParticleFilter.__init__(self, N, transition_model, measurement_model)

    def predict(self, x):
        """
        Predict the state at time t + 1.

        Args:
            x: State at time t.
            P: State covariance at time t.

        Returns:
            x_pred: Predicted state at time t + 1.
            P_pred: Predicted state covariance at time t + 1.
        """

        x_new = np.zeros((self.N, x.shape[1]))
        y_new = np.zeros((self.N, x.shape[1]))

        for i in range(self.N):
            x_new[i] = self.transition_model.predict(x[i])
            y_new[i] = self.measurement_model.measure(x_new[i])

        return x_new, y_new

    def update(self, x_pred, y_pred, logw, y):
        """
        Update the state at time t + 1.

        Args:
            x_pred: Predicted state at time t + 1.
            P_pred: Predicted state covariance at time t + 1.
            y: Observation at time t + 1.

        Returns:
            x_hat: Estimated state at time t + 1.
        """

        # Compute the weights
        logw_new = logw + self.measurement_model.lpdf(x_pred, y)

        # Calculate the log likelihood
        log_likelihood = np.max(logw_new) + np.log(np.sum(np.exp(logw_new - np.max(logw_new))))

        # Normalise the weights
        wn = np.exp(logw_new - log_likelihood)

        # Compute the effective sample size
        ess = 1 / np.sum(wn ** 2)

        # Resample if necessary
        if ess < self.N / 2:
            i = np.linspace(0, self.N-1, self.N, dtype=int)
            i_new = np.random.choice(i, self.N, p=wn)
            x_new = x_pred[i_new]
            y_new = y_pred[i_new]
            logw_new = log_likelihood - np.log(self.N)
        else:
            x_new = x_pred.copy()
            y_new = y_pred.copy()

        return x_new, y_new, logw_new, log_likelihood



