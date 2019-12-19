from street_network import UndirectedStreetNetwork, DirectedStreetNetwork

import numpy as np
from math import exp
from math import pi
from math import sqrt
from itertools import product
from functools import reduce
from itertools import zip_longest
from abc import ABC

from networkx.algorithms.shortest_paths.weighted import all_pairs_dijkstra_path_length
from networkx import all_pairs_dijkstra_path


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
        self.states = list(set(map(lambda x: tuple(sorted(x)), network.graph.edges.keys())))
        self.shortest_paths = {
            source: target_dict
            for source, target_dict in all_pairs_dijkstra_path(
                self.street_network.graph, weight="length"
            )
        }
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
        return self.gamma*exp(-self.gamma * distance)

    def projection_emission_probability(self, z, x):
        distance = self.street_network.distance_from_point_to_edge(z, x)
        return sqrt(2)/(sqrt(pi)*self.sigma)*exp(-(distance ** 2)/(2 * self.sigma ** 2))

    def stitch_segments(self, segment_sequence):
        path = []
        path.append(segment_sequence[0])
        for previous, current in zip(segment_sequence[:-1], segment_sequence[1:]):
            common_nodes = set(previous).intersection(set(current))
            if len(common_nodes) == 2:
                continue
            elif len(common_nodes) == 1:
                path.append(current)
            else:
                assert len(common_nodes) == 0
                shortest_path_candidates_lengths = [(a, b, self.shortest_path_dictionary[a][b]) for a, b in product(previous, current)]
                a, b, _ = min(shortest_path_candidates_lengths, key=lambda x: x[2])
                shortest_path = self.shortest_paths[a][b]
                for node_set in list(zip(shortest_path[:-1], shortest_path[1:])):
                    path.append(tuple(sorted(node_set)))
                path.append(current)
        return path

    def benchmark_estimate(self, observation):
        estimate = []
        points = observation.output_df.point
        graph_edges = list(self.street_network.graph.edges.keys())
        for point in points:
            edges_as_tuple = map(lambda x: tuple(sorted(x)), graph_edges)
            closest_edge = min(edges_as_tuple, key=lambda x: self.street_network.distance_from_point_to_edge(point, x))
            estimate.append(closest_edge)
        return estimate

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
        self.shortest_paths = {
            source: target_dict
            for source, target_dict in all_pairs_dijkstra_path(
                self.street_network.graph, weight="length"
            )
        }
        self.compute_legal_transitions()
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
        states_set = set()
        for edge in iter(self.street_network.graph.edges.keys()):
            node_set = frozenset(edge)
            possible_previous_connections = reduce(
                lambda x, y: x + y,
                map(
                    lambda x: list(
                        zip_longest(
                            [], self.street_network.graph[x].keys(), fillvalue=x
                        )
                    ),
                    node_set,
                ),
            )
            for connection in map(frozenset, possible_previous_connections):
                if connection != node_set:
                    states_set.add((node_set, connection))
        self.states = list(states_set)

    def exponential_decay_transition_probability(self, x, y):
        distance = self.compute_distance(x, y)
        return self.gamma*exp(-self.gamma * distance)

    def projection_emission_probability(self, z, x):
        distance = self.street_network.distance_from_point_to_edge(z, x)
        return sqrt(2)/(sqrt(pi)*self.sigma)*exp(-(distance ** 2)/(2 * self.sigma ** 2))

    def exponential_decay_constrained_transition_probability(self, x, y):
        legal_transition = int(self.legal_transitions[x][y])
        return self.exponential_decay_transition_probability(x, y)*legal_transition
    
    def compute_legal_transitions(self):
        self.legal_transitions = {}
        for outer_state in self.states:
            outer_segment, _ = outer_state
            inner_dict = {}
            for inner_state in self.states:
                _, inner_connection = inner_state
                if inner_connection == outer_segment:
                    inner_dict[inner_state] = True
                elif outer_state == inner_state:
                    inner_dict[inner_state] = True
                else:
                    inner_dict[inner_state] = False
            self.legal_transitions[outer_state] = inner_dict

    def compute_distance(self, x, y):
        if x==y:
            distance = 0
        elif x[0] == y[1]:
            y_shared = self.get_shared_node(y)
            y_departure = self.get_departure_node(y)
            distance = self.shortest_path_dictionary[y_shared][y_departure]
        else:
            x_departure = self.get_departure_node(x)
            y_predecessor = self.get_predecessor_node(y)
            y_shared = self.get_shared_node(y)
            y_departure = self.get_departure_node(y)
            l1 = self.shortest_path_dictionary[x_departure][y_predecessor]
            l2 = self.shortest_path_dictionary[y_predecessor][y_shared]
            l3 = self.shortest_path_dictionary[y_shared][y_departure]
            distance = sum([l1, l2, l3])
        return distance

    def stitch_segments(self, sequence):
        path = []
        departure_node = DirectedStateSpace.get_departure_node(sequence[0])
        shared_node = DirectedStateSpace.get_shared_node(sequence[0])
        path.append(shared_node)
        path.append(departure_node)
        for previous, current in zip(sequence[:-1], sequence[1:]):
            p_segment, p_connection = previous
            c_segment, c_connection = current
            
            previous_departure_node = DirectedStateSpace.get_departure_node(previous)
            previous_shared_node = DirectedStateSpace.get_shared_node(previous)
            predecessor_node = DirectedStateSpace.get_predecessor_node(current)
            shared_node = DirectedStateSpace.get_shared_node(current)
            departure_node = DirectedStateSpace.get_departure_node(current)
            
            if previous == current:
                continue
            elif p_segment == c_connection and previous_shared_node != shared_node:
                path += [departure_node]
            elif p_segment == c_connection and previous_shared_node == shared_node:
                path += [shared_node] + [departure_node]
            elif previous_departure_node == predecessor_node:
                path += [shared_node] + [departure_node]
            else:
                path_to_predeccesor = self.shortest_paths[previous_departure_node][predecessor_node]
                path += path_to_predeccesor[1:-1] + [predecessor_node] + [shared_node] + [departure_node]        
            
        return [tuple(sorted([p,c])) for p, c in zip(path[:-1], path[1:])]

    @staticmethod
    def get_shared_node(state):
        segment, connection = state
        return list(segment.intersection(connection))[0]

    @staticmethod
    def get_predecessor_node(state):
        _, connection = state
        shared_node = frozenset([DirectedStateSpace.get_shared_node(state)])
        return list(connection.difference(shared_node))[0]

    @staticmethod
    def get_departure_node(state):
        segment, _ = state
        shared_node = frozenset([DirectedStateSpace.get_shared_node(state)])
        return list(segment.difference(shared_node))[0]

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
