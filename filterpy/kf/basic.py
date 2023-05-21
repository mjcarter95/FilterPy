import numpy as np

from filterpy.kf.base import BaseKalmanFilter


class BasicKalmanFilter(BaseKalmanFilter):
    def __init__(self, transition_model, measurement_model):
        BaseKalmanFilter.__init__(self, transition_model, measurement_model)

    def predict(self, x, P):
        """
        Predict the state at time t + 1.

        Args:
            x: State at time t.
            P: State covariance at time t.

        Returns:
            x_pred: Predicted state at time t + 1.
            P_pred: Predicted state covariance at time t + 1.
        """

        x_pred = self.transition_model.mean @ x
        P_pred = self.transition_model.mean @ P @ self.transition_model.mean.T + self.transition_model.cov

        return x_pred, P_pred

    def update(self, x_pred, P_pred, y):
        """
        Update the state at time t + 1.

        Args:
            x_pred: Predicted state at time t + 1.
            P_pred: Predicted state covariance at time t + 1.
            y: Observation at time t + 1.

        Returns:
            x_hat: Estimated state at time t + 1.
            P: Estimated state covariance at time t + 1.
        """
        K = P_pred @ self.measurement_model.mean.T @ np.linalg.inv(self.measurement_model.mean @ P_pred @ self.measurement_model.mean.T + self.measurement_model.cov)
        x_hat = x_pred + K @ (y - self.measurement_model.mean @ x_pred)
        P_hat = P_pred - K @ self.measurement_model.mean @ P_pred

        return x_hat, P_hat
