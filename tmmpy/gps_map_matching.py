from math import pi, exp, sqrt

import numpy as np

from hmm import HiddenMarkovModel, GaussianHiddenMarkovModel

from types import MethodType

class GPSMapMatcher():
    def __init__(self, state_space, mode, **kwargs):
        self.kwargs = kwargs
        SUPPORTED_MODES = frozenset(["projection", "gaussian"])
        assert mode in SUPPORTED_MODES
        self.mode = mode
        if "sigma" in kwargs:
            self._sigma = kwargs["sigma"]
        else:
            self._sigma = 1
        self.state_space = state_space
        self._gamma = self.state_space.gamma
        self.transition_probability = state_space.transition_probability
        self.create_emission_probability(mode)
        self.create_hidden_markov_model(mode)
        self.zs = []

    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self, value):
        self._sigma = value
    
    @property
    def gamma(self):
        assert self.state_space.gamma == self._gamma
        return self._gamma
    
    @gamma.setter
    def gamma(self, value):
        self._gamma = self.state_space.gamma = value
        self.create_hidden_markov_model(self.mode)

        
    def initial_probability(self, x):
        return 1/len(self.state_space)
    
    def create_emission_probability(self, mode):
        if mode == "projection":
            assert "sigma" in self.kwargs
            def emission_probability(x, y):
                distance = self.state_space.street_network.distance_from_point_to_edge(x, y)
                return (2 * pi * self.sigma) ** (-1) * exp(
                    -distance ** 2 / (2 * self.sigma ** 2)
                )
            setattr(self, "emission_probability", emission_probability)
        elif mode == "gaussian":
            midpoints = self.state_space.street_network.edges_df.apply(lambda x: x["line"].interpolate(0.5), axis=1).map(lambda x: (x.x, x.y)).values.tolist()
            self.mean = np.array(midpoints)
            self.covariance = np.tile(self.sigma*np.eye(2), [self.mean.shape[0], 1]).reshape(
                self.mean.shape[0], self.mean.shape[1], self.mean.shape[1]
                )
    
    def create_hidden_markov_model(self, mode):
        if mode == "projection":
            assert hasattr(self, "emission_probability")
            self.hidden_markov_model = HiddenMarkovModel(
                self.transition_probability,
                self.emission_probability,
                self.initial_probability,
                self.state_space.states
            )
        elif mode == "gaussian":
            assert hasattr(self, "mean") and hasattr(self, "covariance")
            self.hidden_markov_model = GaussianHiddenMarkovModel(
                self.transition_probability,
                self.initial_probability,
                self.state_space.states,
                self.mean,
                self.covariance
            )
        
    def attach_observations(self, observations, **kwargs):
        self.observations = observations
        if type(self.hidden_markov_model) is HiddenMarkovModel:
            self.z = observations.output_df.point.values.tolist()
        elif type(self.hidden_markov_model) is GaussianHiddenMarkovModel:
            self.z = list(observations.output_df[["x", "y"]].to_numpy().tolist()
        else:
            raise TypeError("Unexpected class in assigned to hidden_markov_model.")
        
    def add_observations(self):
        self.zs.append(self.z)
        