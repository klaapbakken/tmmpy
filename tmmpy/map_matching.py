import os
import json

from data_interface import PostGISQuery
from street_network import StreetNetwork
from state_space import StreetStateSpace
from hmm import HiddenMarkovModel

from math import exp, sqrt, pi


class GPSMapMatcher:
    def __init__(
        self,
        database,
        user,
        password,
        crs,
        xmin,
        xmax,
        ymin,
        ymax,
        gamma,
        sigma
    ):
        self.crs = crs
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.bounding_box = self.xmin, self.xmax, self.ymin, self.ymax
        self._sigma = sigma
        self.state_space = self.create_assosciated_state_space(
            database, user, password, gamma
        )
        self.transition_probability = self.state_space.transition_probability
        self.hmm = HiddenMarkovModel(
            self.transition_probability,
            self.emission_probability,
            self.initial_probability,
            list(self.state_space.street_network.graph.edges.keys())
        )

    def create_assosciated_state_space(
        self, database, user, password, gamma
    ):
        data = PostGISQuery(database, user, password, self.crs, self.bounding_box)
        street_network = StreetNetwork(data)
        return StreetStateSpace(street_network, gamma)

    def emission_probability(self, x, y):
        distance = self.state_space.street_network.distance_from_point_to_edge(x, y)
        return (2 * pi * self.sigma) ** (-1) * exp(
            -distance ** 2 / (2 * self.sigma ** 2)
        )

    def initial_probability(self, x):
        return 1 / self.state_space.street_network.edges_df.shape[0]

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def gamma(self, value):
        self._sigma = value
