import numpy as np


class BaseKalmanFilter():
    def __init__(self, transition_model, measurement_model):
        self.transition_model = transition_model
        self.measurement_model = measurement_model

    def predict(self, x):
        """
        Predict the state at time t + 1.
        """
        raise NotImplementedError("predict method not implemented for this class")

    def update(self, x):
        """
        Update the state at time t + 1.
        """
        raise NotImplementedError("measure method not implemented for this class")


