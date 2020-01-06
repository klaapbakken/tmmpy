from shapely.geometry import Polygon
from shapely.geometry import Point, LineString
from shapely.geometry import MultiLineString
from shapely.ops import linemerge, split, nearest_points

from scipy.stats import multivariate_normal

import numpy as np
import geopandas as gpd
import pandas as pd

import json

from observations import TimelessGPSObservations


import fiona
import random


class GPSSimulator:
    def __init__(self, state_space):
        self.state_space = state_space
        self.street_network = state_space.street_network
        self.crs = state_space.street_network.edges_df.crs

    def define_flow(self):
        self.flow_dictionary = {}
        for node in self.street_network.graph.nodes:
            connected_nodes = self.street_network.graph[node]
            rnum = np.random.uniform(0, 1, size=(len(connected_nodes),))
            self.flow_dictionary[str(node)] = {
                str(node): float(num) for node, num in zip(connected_nodes, rnum)
            }

    def load_flow(self, flow_path):
        with open(flow_path, mode="rb") as f:
            self.flow_dictionary = json.load(f)

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
        dpm = mps/frequency
        measurement_edges = []
        measurement_positions = []
        measurement_edge_indices = []

        initial_node = self.node_sequence[0]
        initial_node_coords = list(self.street_network.point_lookup[initial_node].coords)[0]

        measurement_edges.append(tuple(sorted([initial_node, self.node_sequence[1]])))
        measurement_positions.append(Point([initial_node_coords[0], initial_node_coords[1]]))

        counter = 0
        remaining_space = 0
        for previous_node, current_node in zip(self.node_sequence[:-1], self.node_sequence[1:]):
            previous_node_coords = list(self.street_network.point_lookup[previous_node].coords)[0]
            current_node_coords = list(self.street_network.point_lookup[current_node].coords)[0]
            
            line = LineString([previous_node_coords, current_node_coords])
            length = line.length
            #Can move this far
            available_space = length
            #Need to be able to move this far
            required_space = dpm - remaining_space
            #Distance covered so far on segment
            distance_covered = 0
            while available_space >= required_space:
                p = line.interpolate(distance_covered + required_space)
                distance_covered += required_space
                p_coords = list(p.coords)[0]
                measurement_edges.append(tuple(sorted([previous_node, current_node])))
                measurement_positions.append(Point([p_coords[0], p_coords[1]]))
                measurement_edge_indices.append(counter)
                available_space -= required_space
                required_space = dpm
                remaining_space = 0
            remaining_space += available_space
            counter += 1
        
        self.positions = measurement_positions
        self.measurement_edges = list(map(lambda x: tuple(sorted(x)), measurement_edges))
        self.measurement_edge_indices = measurement_edge_indices

        noise = [multivariate_normal.rvs(mean=np.array([0, 0]), cov=(sigma**2) * np.eye(2)) for _ in range(len(measurement_positions))]
        observations = list(
            map(lambda x: Point(x[0].x + x[1][0], x[0].y + x[1][1]), zip(measurement_positions, noise))
        )
        self.track = gpd.GeoSeries(observations, crs=self.crs)

    def flow_to_transition_matrix(self, mps, polling_frequency):
        f = polling_frequency
        v = mps
        states = self.state_space.states
        state_ids = np.arange(len(states))
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

    def get_gps_observation(self):
        x = self.track.map(lambda x: x.x)
        y = self.track.map(lambda x: x.y)
        df = pd.DataFrame({"x": x, "y": y})
        return TimelessGPSObservations(df, "x", "y", self.crs, self.crs)

    @property
    def edge_sequence(self):
        return [
            tuple(sorted([i, j]))
            for i, j in zip(self.node_sequence[:-1], self.node_sequence[1:])
        ]

    @property
    def gdf(self):
        return self.street_network.edges_df[
            self.street_network.edges_df.node_set.isin(self.edge_sequence)
        ]
