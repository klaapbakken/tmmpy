from street_network import StreetNetwork

import numpy as np
from math import exp
from itertools import product

from networkx.algorithms.shortest_paths.weighted import all_pairs_dijkstra_path_length


class StreetStateSpace:
    """
    State space based on street network.

    Parameters:
    ---
    street_network -- A StreetNetwork object
    gamma -- The transition decay rate
    max_distance -- The maximum distance allowed before an edge is considered unreachable.

    """

    def __init__(self, street_network: StreetNetwork, gamma: float):
        """See class documentation."""
        self.street_network = street_network
        self.shortest_path_dictionary = self.compute_shortest_path_dictionary()
        self._gamma = gamma

    def compute_shortest_path_dictionary(self):
        """Computing all shortest paths. """
        shortest_path_generator = all_pairs_dijkstra_path_length(
            self.street_network.graph, weight="length"
        )
        shortest_path_dictionary = {
            source: distance_dictionary
            for source, distance_dictionary in shortest_path_generator
        }
        return shortest_path_dictionary

    def transition_probability(self, x, y):
        """Transition probability """
        distance = min((self.shortest_path_dictionary[a][b] for a, b in product(x, y)))
        return exp(-self.gamma * distance)

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value
