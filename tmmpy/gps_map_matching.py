from math import pi, exp, sqrt

import numpy as np

from hmm import HiddenMarkovModel, GaussianHiddenMarkovModel

from state_space import DirectedStateSpace, UndirectedStateSpace
from types import MethodType


class GPSMapMatcher:
    def __init__(self, state_space, transition_mode, emission_mode, **kwargs):
        self.SUPPORTED_EMISSION_MODES = frozenset(["projection", "gaussian"])
        self.SUPPORTED_TRANSITION_MODES = frozenset(["exponential", "uniform"])
        self.state_space = state_space

        self.initial_probability = state_space.uniform_initial_probability
        self.emission_mode = emission_mode
        self.transition_mode = transition_mode

        self.zs = []

    def attach_observations(self, observations, **kwargs):
        self.observations = observations
        if type(self.hidden_markov_model) is HiddenMarkovModel:
            self.z = observations.output_df.point.tolist()
        elif type(self.hidden_markov_model) is GaussianHiddenMarkovModel:
            self.z = list(observations.output_df[["x", "y"]].to_numpy().tolist())
        else:
            raise TypeError("Unexpected class in assigned to hidden_markov_model.")

    def create_hidden_markov_model(self):
        if self.emission_mode == "projection":
            assert hasattr(self, "emission_probability")
            self.hidden_markov_model = HiddenMarkovModel(
                self.transition_probability,
                self.emission_probability,
                self.initial_probability,
                self.state_space.states,
            )
        elif self.emission_mode == "gaussian":
            assert hasattr(self, "mean") and hasattr(self, "covariance")
            self.hidden_markov_model = GaussianHiddenMarkovModel(
                self.transition_probability,
                self.initial_probability,
                self.state_space.states,
                self.mean,
                self.covariance,
            )

    def add_observations(self):
        self.zs.append(self.z)

    def clear_observations(self):
        self.zs = []

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value
        self.state_space.sigma = value

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self.state_space.gamma = value

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value):
        self._mean = value

    @property
    def covariance(self):
        return self._covariance

    @covariance.setter
    def covariance(self, value):
        self._covariance = value

    @property
    def transition_probability(self):
        return self._transition_probability

    @transition_probability.setter
    def transition_probability(self, value):
        self._transition_probability = value

    @property
    def transition_mode(self):
        return self._transition_mode

    @transition_mode.setter
    def transition_mode(self, value):
        assert value in self.SUPPORTED_TRANSITION_MODES
        if value == "exponential":
            self.transition_probability = (
                self.state_space.exponential_decay_transition_probability
            )
        elif value == "uniform":
            self.transition_probability = (
                self.state_space.uniform_transition_probability
            )
        self._transition_mode = value
        if hasattr(self, "_emission_probability") and hasattr(self, "_transition_probability"):
            self.create_hidden_markov_model()

    @property
    def emission_probability(self):
        return self._emission_probability

    @emission_probability.setter
    def emission_probability(self, value):
        self._emission_probability = value

    @property
    def emission_mode(self):
        return self._emission_mode

    @emission_mode.setter
    def emission_mode(self, value):
        assert value in self.SUPPORTED_EMISSION_MODES
        if value == "projection":
            self.mean = None
            self.covariance = None
            self.emission_probability = self.state_space.projection_emission_probability
        elif value == "gaussian":
            self.emission_probability = None
            self.mean, self.covariance = self.state_space.gaussian_emission_parameters
        self._emission_mode = value
        if hasattr(self, "_emission_probability") and hasattr(self, "_transition_probability"):
            self.create_hidden_markov_model()
