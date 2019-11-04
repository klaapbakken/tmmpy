from shapely.geometry import Polygon
from shapely.geometry import Point, LineString
from shapely.geometry import MultiLineString
from shapely.ops import linemerge, split, nearest_points

from scipy.stats import multivariate_normal

import numpy as np
import geopandas as gpd
import pandas as pd

from observations import TimelessGPSObservations


import fiona
import random


class GPSSimulator:
    def __init__(self, map_matcher, polling_frequency, mps):
        self.map_matcher = map_matcher
        self.street_network = map_matcher.state_space.street_network
        self.crs = self.street_network.edges_df.crs
        self.f = polling_frequency
        self.v = mps

    def define_flow(self):
        self.flow_dictionary = {}
        for node in self.street_network.graph.nodes:
            connected_nodes = self.street_network.graph[node]
            rnum = np.random.uniform(0, 1, size=(len(connected_nodes),))
            self.flow_dictionary[str(node)] = {
                str(node): float(num) for node, num in zip(connected_nodes, rnum)
            }

    def simulate_node_sequence(self, route_length):
        starting_position = list(
            random.choice(list(self.street_network.graph.edges.keys()))
        )
        random.shuffle(starting_position)
        node_sequence = []
        previous_node = starting_position[0]
        current_node = starting_position[1]
        length = 0
        node_sequence.append(current_node)
        while length < route_length:
            connected_nodes = self.street_network.graph[current_node]
            candidates = list(filter(lambda x: x != previous_node, connected_nodes))
            if candidates == []:
                candidates = [previous_node]
            candidate_weights = [
                self.flow_dictionary[str(current_node)][str(candidate)]
                for candidate in candidates
            ]
            ps = np.array(candidate_weights) / sum(candidate_weights)
            next_node = np.random.choice(np.array(candidates), p=np.array(ps))
            length += self.street_network.graph[current_node][next_node]["length"]
            previous_node = current_node
            current_node = next_node
            node_sequence.append(current_node)
        self.node_sequence = node_sequence

    def simulate_gps_tracks(self, mps, frequency, sigma):
        i_edge = self.edge_sequence[0]
        ils = self.street_network.graph[i_edge[0]][i_edge[1]]["line"]
        p = ils.interpolate(random.uniform(0, 1), normalized=True)
        split_ils = LineString([list(p.coords[0])] + [list(ils.coords[1])])

        mls = MultiLineString(
            [split_ils] + [self.street_network.graph[a][b]["line"] for a, b in self.edge_sequence[1:]]
        )
        ls = linemerge(mls)
        fractions = np.arange(0, ls.length, step=mps / frequency)
        self.positions = [ls.interpolate(x) for x in fractions]
        noise = [multivariate_normal.rvs(mean=np.array([0, 0]), cov=(sigma**2) * np.eye(2)) for _ in range(len(self.positions))]
        observations = list(
            map(lambda x: Point(x[0].x + x[1][0], x[0].y + x[1][1]), zip(self.positions, noise))
        )
        self.track = gpd.GeoSeries(observations, crs=self.crs)

    def flow_to_transition_matrix(self):
        f = self.f
        v = self.v
        state_ids = self.map_matcher.hidden_markov_model.state_ids
        states = self.map_matcher.hidden_markov_model.states
        graph = self.street_network.graph
        flow = self.flow_dictionary
        

        state_to_id_dict = {state : state_id for state_id, state in zip(state_ids, states)}

        distance_per_measurement = v/f
        M = len(states)
        P = np.zeros((M, M))

        for segment_set, connection_set in states:
            segment = tuple(segment_set)
            connection = tuple(connection_set)
            #Find the node present in both connection and segment
            shared_node = set(segment).intersection(set(connection))
            segment_length = graph[segment[0]][segment[1]]["length"]
            #The node we move on from
            departure_node = segment_set.difference(shared_node)
            assert len(departure_node) == 1
            #Convert to string, since flow is read from JSON and keys are strings.
            departure_key = str(list(departure_node)[0])
            #Nodes connected to the departure node
            connected_keys = set(flow[departure_key].keys())
            #The state we're moving on from
            i = state_to_id_dict[(segment_set, connection_set)]
            ws = []
            if len(connected_keys) == 1:
                #We've reached a dead end
                deadend_key = departure_key
                #We need to travel back and forth on dead end segment
                num_self_transitions = (2*segment_length)/distance_per_measurement
                #We must now depart from the shared node
                new_departure_key = str(list(shared_node)[0])
                #The connected keys are those nodes leading away from the shared node
                new_connected_keys = set(flow[new_departure_key].keys())
                #Extracting the weights from flow dictionary
                for key in new_connected_keys.difference(set([deadend_key])):
                    ws.append(flow[new_departure_key][key])
                ws.append(num_self_transitions*sum(ws))
                #Scaling to sum to one
                ps = np.array(ws)/np.sum(ws)
                #Finding the states we can move on to
                for n, key in enumerate(new_connected_keys.difference(set([deadend_key]))):
                    j = state_to_id_dict[(frozenset([int(new_departure_key), int(key)]), (segment_set))]
                    #Setting the probabilities
                    P[i, j] = ps[n]
                P[i, i] = ps[-1]
            else:
                #Expected number of transitions to same state
                num_self_transitions = segment_length/distance_per_measurement
                #Extracting the weights from flow dictionary
                for key in connected_keys.difference(set(map(str, shared_node))):
                    ws.append(flow[departure_key][key])
                ws.append(num_self_transitions*sum(ws))
                #Scaling to sum to one
                ps = np.array(ws)/sum(ws)
                #Finding the states we can move on to
                for n, key in enumerate(connected_keys.difference(set(map(str, shared_node)))):
                    j = state_to_id_dict[(frozenset([int(departure_key), int(key)]), (segment_set))]
                    #Setting the probabilities
                    P[i, j] = ps[n]
                P[i, i] = ps[-1]
        return P

    @property
    def edge_sequence(self):
        return [
            tuple(sorted([i, j]))
            for i, j in zip(self.node_sequence[:-1], self.node_sequence[1:])
        ]

    @property
    def measurement_edges(self):
        lines = self.street_network.edges_df.linestring.values
        node_sets = self.street_network.edges_df.node_set.values

        measurement_node_sets = []
        for position in self.positions:
            candidates = [nearest_points(position, line) for line in lines]
            lengths = list(map(lambda x: LineString([(p.x, p.y) for p in x]).length, candidates))
            _, index = min(zip(lengths, range(len(lengths))), key=lambda x: x[0])
            measurement_node_set = node_sets[index]
            measurement_node_sets.append(tuple(sorted(measurement_node_set)))
        return measurement_node_sets

    @property
    def gdf(self):
        return self.street_network.edges_df[
            self.street_network.edges_df.node_set.isin(self.edge_sequence)
        ]

    @property
    def gps_observation(self):
        x = self.track.map(lambda x: x.x)
        y = self.track.map(lambda x: x.y)
        df = pd.DataFrame({"x": x, "y": y})
        return TimelessGPSObservations(df, "x", "y", self.crs, self.crs)

    @staticmethod
    def bounding_box_to_polygon(bounding_box):
        xmin, xmax, ymin, ymax = bounding_box
        return Polygon([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])

    @staticmethod
    def transform_geometry(geometry, in_crs, out_crs):
        transformed = gpd.GeoSeries([geometry], crs=fiona.crs.from_epsg(in_crs)).to_crs(
            out_crs
        )
        return transformed[0]

    @staticmethod
    def polygon_to_bounding_box(polygon):
        bounding_box = polygon.bounds
        return (bounding_box[0], bounding_box[2], bounding_box[1], bounding_box[3])
