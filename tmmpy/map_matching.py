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
    for node in iter(graph.node.keys()):
        distances[node] = np.inf
    return distances


# And finally this
def get_reachable_nodes(distance_dictionary):
    return [k for k, v in filter(lambda x: x[1] < np.inf, distance_dictionary.items())]


def recursive_path_search(
    graph, node, blocked_nodes, path_length, distance_dictionary, maximum_distance
):
    for target in iter(graph[node].keys()):
        if target in blocked_nodes:
            continue
        else:
            edge_length = graph[node][target]["length"]
            new_path_length = path_length + edge_length
            if (
                new_path_length < distance_dictionary[target]
                and new_path_length < maximum_distance
            ):
                distance_dictionary[target] = new_path_length
                new_blocked_nodes = blocked_nodes.copy()
                new_blocked_nodes.add(node)
                recursive_path_search(
                    graph,
                    target,
                    new_blocked_nodes,
                    new_path_length,
                    distance_dictionary,
                    maximum_distance,
                )
        return distance_dictionary


def find_all_distance_constrained_paths(graph, starting_edge, maximum_distance):
    u, v = starting_edge
    distance_dictionary = initialize_distance_dictionary(graph)
    distance_dictionary[u] = 0
    distance_dictionary[v] = 0

    for target in iter(graph[u].keys()):
        recursive_path_search(graph, u, {v}, 0, distance_dictionary, maximum_distance)
        recursive_path_search(graph, v, {u}, 0, distance_dictionary, maximum_distance)
    return distance_dictionary


def get_reachable_edges(edges_df, distance_dictionary):
    reachable_nodes = get_reachable_nodes(distance_dictionary)
    reachable_edges_df = edges_df[
        edges_df.u.map(lambda x: x in reachable_nodes)
        | edges_df.v.map(lambda x: x in reachable_nodes)
    ]
    reachable_edges_df.loc[:, "u_distance"] = reachable_edges_df.u.map(
        lambda x: distance_dictionary[x]
    )
    reachable_edges_df.loc[:, "v_distance"] = reachable_edges_df.v.map(
        lambda x: distance_dictionary[x]
    )
    reachable_edges_df.loc[:, "distance"] = reachable_edges_df.apply(
        lambda x: min(x["u_distance"], x["v_distance"]), axis=1
    )
    return reachable_edges_df
