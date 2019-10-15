from street_network import UndirectedStreetNetwork, DirectedStreetNetwork

import numpy as np
from math import exp
from math import pi
from itertools import product
from functools import reduce
from itertools import zip_longest
from abc import ABC

from networkx.algorithms.shortest_paths.weighted import all_pairs_dijkstra_path_length


class StateSpace:
    """All states equally probable."""

    def uniform_initial_probability(self, x):
        return 1 / len(self)

    """All transitions equally probable."""

    def uniform_transition_probability(self, x, y):
        return 1 / len(self)

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if value <= 0:
            raise ValueError("Gamma must be positive.")
        else:
            self._gamma = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        if value <= 0:
            raise ValueError("Sigma must be positive.")
        else:
            self._sigma = value


class UndirectedStateSpace(StateSpace):
    """
    State space based on street network.

    Parameters:
    ---
    street_network -- A StreetNetwork object
    gamma -- The transition decay rate
    max_distance -- The maximum distance allowed before an edge is considered unreachable.

    """

    def __init__(self, network: UndirectedStreetNetwork, **kwargs):
        """See class documentation."""
        assert type(network) is UndirectedStreetNetwork
        self.street_network = network
        self.states = network.edges_df.node_set.values.tolist()
        self.shortest_path_dictionary = {
            source: target_dict
            for source, target_dict in all_pairs_dijkstra_path_length(
                self.street_network.graph, weight="length"
            )
        }
        if "gamma" in kwargs:
            self.gamma = kwargs["gamma"]
        else:
            self.gamma = 0.01
        if "sigma" in kwargs:
            self.sigma = kwargs["sigma"]
        else:
            self.sigma = 1

    def __len__(self):
        return len(self.states)

    def exponential_decay_transition_probability(self, x, y):
        """Transition probability """
        distance = min((self.shortest_path_dictionary[a][b] for a, b in product(x, y)))
        return exp(-self.gamma * distance)

    def projection_emission_probability(self, z, x):
        distance = self.street_network.distance_from_point_to_edge(z, x)
        return (2 * pi * self.sigma) ** (-1) * exp(
            -distance ** 2 / (2 * self.sigma ** 2)
        )

    @property
    def gaussian_emission_parameters(self):
        midpoints = (
            self.street_network.edges_df.apply(
                lambda x: x["linestring"].interpolate(0.5), axis=1
            )
            .map(lambda x: (x.x, x.y))
            .values.tolist()
        )
        mean = np.array(midpoints)
        covariance = np.tile(
            self.street_network.edges_df.length.mean() * np.eye(mean.shape[1]),
            [mean.shape[0], 1, 1],
        )
        return mean, covariance


class DirectedStateSpace(StateSpace):
    def __init__(self, network, **kwargs):
        assert type(network) is DirectedStreetNetwork
        self.street_network = network
        self.create_states()
        self.shortest_path_dictionary = {
            source: target_dict
            for source, target_dict in all_pairs_dijkstra_path_length(
                network.graph, weight="length"
            )
        }
        if "gamma" in kwargs:
            self.gamma = kwargs["gamma"]
        else:
            self.gamma = 0.01
        if "sigma" in kwargs:
            self.sigma = kwargs["sigma"]
        else:
            self.sigma = 1

    def __len__(self):
        return len(self.states)

    def create_states(self):
        states = []
        for _, row in self.street_network.edges_df.iterrows():
            segment = row["node_set"]
            possible_previous_connections = reduce(
                lambda x, y: x + y,
                map(
                    lambda x: list(
                        zip_longest(
                            [], self.street_network.graph[x].keys(), fillvalue=x
                        )
                    ),
                    segment,
                ),
            )
            for connection in possible_previous_connections:
                states.append((segment, connection))
        self.states = states

    def exponential_decay_transition_probability(self, x, y):
        ex = x[0]
        ey = y[0]
        distance = min(
            [self.shortest_path_dictionary[s][t] for s, t in product(ex, ey)]
        )
        return exp(-self.gamma * distance)

    def projection_emission_probability(self, z, x):
        distance = self.street_network.distance_from_point_to_edge(z, x)
        return (2 * pi * self.sigma) ** (-1) * exp(
            -distance ** 2 / (2 * self.sigma ** 2)
        )

    @staticmethod
    def node_path_to_edge_path(node_path):
        return [(a, b) for a, b in zip(node_path[:-1], node_path[1:])]

    @property
    def gaussian_emission_parameters(self):
        midpoints = []
        lengths = []
        for current, _ in self.states:
            linestring = self.street_network.linestring_lookup[tuple(sorted(current))]
            midpoint = linestring.interpolate(0.5)
            midpoints.append((midpoint.x, midpoint.y))
            lengths.append(linestring.length)
        mean = np.array(midpoints)
        covariance = np.tile(np.mean(lengths) * np.eye(mean.shape[1]), [mean.shape[0], 1, 1])
        return mean, covariance
