import numpy as np


class BaseParticleFilter():
    def __init__(self, N, transition_model, measurement_model):
        self.N = N  # Number of particles
        self.transition_model = transition_model  # Transition model
        self.measurement_model = measurement_model  # Measurement model

    def predict(self, x):
        """
        Predict the state at time t + 1.
        """
        raise NotImplementedError("measure method not implemented for this class")

    def update(self, x, logw, y):
        """
        Update the state at time t + 1.
        """
        raise NotImplementedError("update method not implemented for this class")
