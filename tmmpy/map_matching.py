import os
import json

from data_interface import PostGISQuery, OverpassAPIQuery
from street_network import StreetNetwork
from state_space import StreetStateSpace
from hmm import HiddenMarkovModel

from math import exp, sqrt, pi


class GPSMapMatcher:
    """
    Class for doing map matching using GPS data.
    
    Parameters:
    ---
    database -- Name of the database
    user -- Name of the user
    password -- Password for the user
    crs -- Output coordinate reference system
    xmin -- Bottom left corner of bounding box
    xmax -- Bottom right corner of bounding box
    ymin -- Top left corner of bounding box
    ymax -- Top right corner of bounding box
    gamma -- Transition decay rate
    sigma -- Emission variance
    """

    def __init__(
        self,
        crs: str,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        gamma: float,
        sigma: float,
        method="overpass",
        **kwargs
    ):
        if method == "postgis":
            assert database in kwargs and user in kwargs and password in kwargs
        self.crs = crs
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.bounding_box = self.xmin, self.xmax, self.ymin, self.ymax
        self._sigma = sigma
        if method == "postgis":
            self.state_space = self.create_assosciated_state_space(
                kwargs["database"], kwargs["user"], kwargs["password"], gamma
            )
        elif method == "overpass":
            self.state_space = self.create_assosciated_state_space_using_overpass(gamma)
        else:
            raise ValueError(
                'Named argument "method" must be either "overpass" or "postgis"'
            )
        self.transition_probability = self.state_space.transition_probability
        self.hmm = HiddenMarkovModel(
            self.transition_probability,
            self.emission_probability,
            self.initial_probability,
            list(self.state_space.street_network.graph.edges.keys()),
        )

    def create_assosciated_state_space_using_postgis(
        self, database, user, password, gamma
    ):
        """Creating a state space using the arguments supplied in constructor."""
        data = PostGISQuery(database, user, password, self.crs, self.bounding_box)
        street_network = StreetNetwork(data)
        return StreetStateSpace(street_network, gamma)

    def create_assosciated_state_space_using_overpass(self, gamma):
        """Creating a state space using the arguments supplied in constructor."""
        data = OverpassAPIQuery(self.crs, self.bounding_box)
        street_network = StreetNetwork(data)
        return StreetStateSpace(street_network, gamma)

    def emission_probability(self, x, y):
        """Emission probability, normally distributed around closest point on edge."""
        distance = self.state_space.street_network.distance_from_point_to_edge(x, y)
        return (2 * pi * self.sigma) ** (-1) * exp(
            -distance ** 2 / (2 * self.sigma ** 2)
        )

    def initial_probability(self, x):
        """Same initial probabilities for all states."""
        return 1 / self.state_space.street_network.edges_df.shape[0]

    # def most_likely_edge_sequence(self, observations):
    #     path = self.hmm.most_likely_path(observations)
    #     self.state_space.street_network.edges_df[
    #         self.state_space.street_network.edges_df.linestring.map(lambda x: x.equals()
    #         ]

    @property
    def sigma(self):
        """Sigma getter."""
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        """Sigma setter."""
        self._sigma = value
