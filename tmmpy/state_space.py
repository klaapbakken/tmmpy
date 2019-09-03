from street_network import StreetNetwork

import numpy as np
from math import exp


class StreetStateSpace:
    """
    State space based on street network.

    Parameters:
    ---
    street_network -- A StreetNetwork object
    gamma -- The transition decay rate
    max_distance -- The maximum distance allowed before an edge is considered unreachable.

    """
    def __init__(self, street_network: StreetNetwork, gamma: float, max_distance: float):
        """See class documentation."""
        self.street_network = street_network
        self.transition_probability = self.compute_transition_probability(gamma, max_distance)

    def initialize_distance_dictionary(self):
        """Create a distance dictionary that keeps track of reachable nodes."""
        distances = {}
        for node in iter(self.street_network.graph.node.keys()):
            distances[node] = np.inf
        return distances

    def get_reachable_nodes(self, distance_dictionary: dict):
        """Get a list of all the reachable nodes."""
        return [
            k for k, v in filter(lambda x: x[1] < np.inf, distance_dictionary.items())
        ]

    def recursive_path_search(
        self, node: int, blocked_nodes: list, path_length: float, distance_dictionary: dict, maximum_distance: float
    ):
        """A recursive search for edges in vicinity."""
        #For each node directly connected to current node
        for target in iter(self.street_network.graph[node].keys()):
            #If the target has been visited previously
            if target in blocked_nodes:
                #Move to next node
                continue
            else:
                #Length of the edge between current node and node
                edge_length = self.street_network.graph[node][target]["length"]
                #Added to the length so far
                new_path_length = path_length + edge_length
                #If the new length is less than max distance
                #and the new node has not been found by a shorter path
                if (
                    new_path_length < distance_dictionary[target]
                    and new_path_length < maximum_distance
                ):
                    #Updating the distance dictionary
                    distance_dictionary[target] = new_path_length
                    #Block revisiting previously visited nodes
                    new_blocked_nodes = blocked_nodes.copy()
                    new_blocked_nodes.add(node)
                    #Repeat
                    self.recursive_path_search(
                        target,
                        new_blocked_nodes,
                        new_path_length,
                        distance_dictionary,
                        maximum_distance,
                    )
            return distance_dictionary

    def find_all_distance_constrained_paths(self, starting_edge, maximum_distance):
        """Function for starting the recursive search from both head and tail of edge."""
        u, v = starting_edge
        distance_dictionary = self.initialize_distance_dictionary()
        distance_dictionary[u] = 0
        distance_dictionary[v] = 0

        self.recursive_path_search(u, {v}, 0, distance_dictionary, maximum_distance)
        self.recursive_path_search(v, {u}, 0, distance_dictionary, maximum_distance)
        return distance_dictionary

    def in_edges_df(self, edge, edges_df):
        """Checks whether an edge exists in a dataframe."""
        return edges_df.node_set.map(lambda x: x == tuple(sorted(edge))).any()

    def get_reachable_edges(self, distance_dictionary):
        """Calculating shortest distance to each edge."""
        #Get all reachable nodes
        reachable_nodes = self.get_reachable_nodes(distance_dictionary)
        #Get all edges that has head or tail in reachable node
        reachable_edges_df = self.street_network.edges_df[
            self.street_network.edges_df.u.map(lambda x: x in reachable_nodes)
            | self.street_network.edges_df.v.map(lambda x: x in reachable_nodes)
        ]
        #Calculate distance from node that distance dictionary is based on to all of the nodes
        #that participate in edges
        reachable_edges_df.loc[:, "u_distance"] = reachable_edges_df.u.map(
            lambda x: distance_dictionary[x]
        )
        reachable_edges_df.loc[:, "v_distance"] = reachable_edges_df.v.map(
            lambda x: distance_dictionary[x]
        )
        #Calculate the shortest distance to each edge
        reachable_edges_df.loc[:, "distance"] = reachable_edges_df.apply(
            lambda x: min(x["u_distance"], x["v_distance"]), axis=1
        )
        return reachable_edges_df

    def compute_transition_probability(self, gamma: float, max_distance: float):
        """Precomputation. Returning a glorified indexing function."""
        reachable_edges_dictionary = {
            k: v
            for k, v in zip(
                iter(self.street_network.graph.edges),
                map(
                    lambda x: self.get_reachable_edges(
                        self.find_all_distance_constrained_paths(x, max_distance)
                    ),
                    iter(self.street_network.graph.edges),
                ),
            )
        }

        def transition_probability(x, y):
            if not self.in_edges_df(y, reachable_edges_dictionary[x]):
                return 0
            else:
                df = reachable_edges_dictionary[x]
                distance = df[df.node_set.map(lambda x: x == tuple(sorted(y)))][
                    "distance"
                ].iloc[0]
                return exp(-gamma * distance)

        return transition_probability
