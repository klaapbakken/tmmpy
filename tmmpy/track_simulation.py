from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import MultiLineString
from shapely.ops import linemerge

from scipy.stats import multivariate_normal

import numpy as np
import geopandas as gpd
import pandas as pd

from observations import TimelessGPSObservations


import fiona
import random

class GPSSimulator():
    def __init__(self, street_network):
        self.street_network = street_network
        self.crs = self.street_network.edges_df.crs
    
    def define_flow(self):
        self.flow_dictionary = {}
        for node in self.street_network.graph.nodes:
            connected_nodes = self.street_network.graph[node]
            rnum = np.random.uniform(0, 1, size=(len(connected_nodes),))
            self.flow_dictionary[node] = {node : num for node, num in zip(connected_nodes, rnum)}
    
    def simulate_node_sequence(self, route_length):
        starting_position = list(random.choice(list(self.street_network.graph.edges.keys())))
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
            candidate_weights = [self.flow_dictionary[current_node][candidate] for candidate in candidates]
            ps = candidate_weights/sum(candidate_weights)
            next_node = np.random.choice(np.array(candidates), p=np.array(ps))
            length += self.street_network.graph[current_node][next_node]["length"]
            previous_node = current_node
            current_node = next_node
            node_sequence.append(current_node)
        self.node_sequence = node_sequence

    @property
    def edge_sequence(self):
        return [tuple(sorted([i, j])) for i, j in zip(self.node_sequence[:-1], self.node_sequence[1:])]
    
    @property
    def gdf(self):
        return self.street_network.edges_df[self.street_network.edges_df.node_set.isin(self.edge_sequence)]

    def simulate_gps_tracks(self, mps, frequency, sigma):
        mls = MultiLineString([self.street_network.graph[a][b]["line"] for a, b in self.edge_sequence])
        ls = linemerge(mls)
        ls.length
        fractions = np.arange(0, ls.length, step=mps/frequency)
        self.positions = [ls.interpolate(x) for x in fractions]
        noise = multivariate_normal.rvs(mean=np.array([0,0]), cov=sigma*np.eye(2))
        observations = list(map(lambda x: Point(x.x + noise[0], x.y + noise[1]), self.positions))
        self.track = gpd.GeoSeries(observations, crs=self.crs)
    
    @property
    def gps_observation(self):
        x = self.track.map(lambda x: x.x)
        y = self.track.map(lambda x: x.y)
        df = pd.DataFrame({"x" : x, "y" : y})
        return TimelessGPSObservations(df, "x", "y", self.crs, self.crs)

    @staticmethod
    def bounding_box_to_polygon(bounding_box):
        xmin, xmax, ymin, ymax = bounding_box
        return Polygon([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])

    @staticmethod
    def transform_geometry(geometry, in_crs, out_crs):
        transformed = gpd.GeoSeries([geometry], crs=fiona.crs.from_epsg(in_crs)).to_crs(out_crs)
        return transformed[0]

    @staticmethod
    def polygon_to_bounding_box(polygon):
        bounding_box = polygon.bounds
        return (bounding_box[0], bounding_box[2], bounding_box[1], bounding_box[3])