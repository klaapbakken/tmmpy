import os
import json
secret_location = os.path.abspath(os.path.join("..", "secret.json"))
with open(secret_location) as f:
    secret_password = json.load(f)["password"]

from data_interface import PostGISQuery
from street_network import StreetNetwork
from state_space import StreetStateSpace
from hmm import HiddenMarkovModel

from math import exp, sqrt, pi

ALLOWED_HIGHWAY_VALUES = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
    "service",
    "living_street",
    "road",
    "cycleway",
    "footway",
    "path"
]

ALLOWED_CYCLEWAY_VALUES = [
    "lane",
    "opposite",
    "opposite_lane",
    "track",
    "opposite_track",
    "share_busway",
    "opposite_share_busway",
    "shared_lane"
]

FILTER_DICTIONARY = {
    "highway" : ALLOWED_HIGHWAY_VALUES,
    "cycleway" : ALLOWED_CYCLEWAY_VALUES
}


class GPSMapMatcher():
    def __init__(self, xmin, xmax, ymin, ymax, crs, gamma, sigma, max_distance):
        self.crs = crs
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.state_space = self.create_assosciated_state_space("osm", "postgres", secret_password, gamma, max_distance)
        self.transition_probability = self.state_space.transition_probability
        self.emission_probability = self.compute_emission_probability(sigma)
        self.initial_probability = self.compute_initial_probability()
        self.hmm = HiddenMarkovModel(
            self.transition_probability,
            self.emission_probability,
            self.initial_probability,
            self.state_space.street_network.edges_df.node_set.values.tolist()
            )

    def create_assosciated_state_space(self, database, user, password, gamma, max_distance):
        data_source = PostGISQuery(database, user, password, self.crs, filter_dictionary=FILTER_DICTIONARY)
        data_source.get_ways_intersecting(self.xmin, self.xmax, self.ymin, self.ymax)
        street_network = StreetNetwork(data_source, self.crs)
        return StreetStateSpace(street_network, gamma, max_distance)
    
    def compute_emission_probability(self, sigma):
        def emission_probability(x, y):
            distance = self.state_space.street_network.distance_from_point_to_edge(x, y)
            return (2*pi*sigma)**(-1)*exp(-distance**2/(2*sigma**2))
        return emission_probability

    def compute_initial_probability(self):
        def initial_probability(x):
            return 1/self.state_space.street_network.edges_df.shape[0]
        return initial_probability