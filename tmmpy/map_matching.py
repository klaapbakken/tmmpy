import networkx as nx
import numpy as np

from itertools import product
from shapely.geometry import LineString
from shapely.ops import nearest_points


def shortest_path_length_between_nodes(graph, l_node, r_node):
    return nx.shortest_path_length(graph, source=l_node, target=r_node, weight="length")


def shortest_path_length_between_edges(graph, l_edge, r_edge):
    return min(
        (
            shortest_path_length_between_nodes(graph, nodes[0], nodes[1])
            for nodes in product(l_edge, r_edge)
        )
    )


def distance_from_point_to_edge(point, edge):
    closest_points = nearest_points(point, edge)
    ls = LineString([(p.x, p.y) for p in closest_points])
    return ls.length


def distance_from_point_to_point(l_point, r_point):
    ls = LineString([(p.x, p.y) for p in [l_point, r_point]])
    return ls.length


# Call this
def initialize_distance_dictionary(graph):
    distances = {}
    for node in iter(n.graph.node.keys()):
        distances[node] = np.inf
    return distances


# Then this
def all_distance_constrained_paths(
    graph, source, path_length, distance_dictionary, maximum_distance
):
    # For later: Avoid revisting old states
    for target in iter(n.graph[source].keys()):
        edge_length = graph[source][target]["length"]
        new_path_length = path_length + edge_length
        if (
            new_path_length < distance_dictionary[target]
            and new_path_length < maximum_distance
        ):
            distance_dictionary[target] = new_path_length
        if new_path_length < maximum_distance:
            all_distance_constrained_paths(
                graph, target, new_path_length, distance_dictionary, maximum_distance
            )


# And finally this
def get_reachable_nodes(distance_dictionary):
    return [k for k, v in filter(lambda x: x[1] < np.inf, distance_dictionary.items())]
