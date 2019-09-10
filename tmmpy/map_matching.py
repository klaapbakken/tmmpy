import os
import json

import pandas as pd

from data_interface import PostGISQuery, OverpassAPIQuery, NVDBAPIQuery
from street_network import StreetNetwork
from state_space import StreetStateSpace
from hmm import HiddenMarkovModel


import numpy as np
from itertools import product

import matplotlib.pyplot as plt

from math import exp, sqrt, pi
from networkx.algorithms.shortest_paths.generic import shortest_path


class GPSMapMatcher:
    """
    Class for doing map matching using GPS data.
    
    Parameters:
    ---
    database -- Name of the database (optional keyword argument, must be present if method=\"postgis\")
    user -- Name of the user (optional keyword argument, must be present if method=\"postgis\")
    password -- Password for the user (optional keyword argument, must be present if method=\"postgis\")
    crs -- Output coordinate reference system
    xmin -- Bottom left corner of bounding box (optional keyword argument, either this or observations must be supplied)
    xmax -- Bottom right corner of bounding box (optional keyword argument, either this or observations must be supplied)
    ymin -- Top left corner of bounding box (optional keyword argument, either this or observations must be supplied)
    ymax -- Top right corner of bounding box (optional keyword argument, either this or observations must be supplied)
    observations -- GPSObservations object (optional keyword argument, either this or coordinates must be supplied)
    gamma -- Transition decay rate
    sigma -- Emission variance
    """

    def __init__(
        self, epsg: str, gamma: float, sigma: float, method="overpass", **kwargs
    ):
        assert (
            all(map(lambda x: x in kwargs, ["xmin", "ymin", "xmax", "ymax"]))
            or "observations" in kwargs
        )
        self.epsg = epsg
        self._sigma = sigma
        if "observations" in kwargs and method in {"overpass", "postgis"}:
            self.attach_observations(kwargs["observations"])
            self.bounding_box = self.observations.lonlat_bounding_box
        elif "observations" in kwargs and method  == "nvdb":
            self.attach_observations(kwargs["observations"])
            self.bounding_box = self.observations.utm33_bounding_box
        if all(map(lambda x: x in kwargs, ["xmin", "ymin", "xmax", "ymax"])):
            self.most_likely_edge_sequence = None
            self.bounding_box = (
                kwargs["xmin"],
                kwargs["xmax"],
                kwargs["ymin"],
                kwargs["ymax"],
            )
        if method == "postgis":
            self.state_space = self.create_assosciated_state_space_using_postgis(
                kwargs["database"], kwargs["user"], kwargs["password"], gamma
            )
        elif method == "overpass":
            self.state_space = self.create_assosciated_state_space_using_overpass(gamma)
        elif method == "nvdb":
            self.state_space = self.create_assosciated_state_space_using_nvdb(gamma)
        else:
            raise ValueError(
                'Named argument "method" must be either "overpass", "postgis" or "nvdb"'
            )
        self.transition_probability = self.state_space.transition_probability
        self.hmm = HiddenMarkovModel(
            self.transition_probability,
            self.emission_probability,
            self.initial_probability,
            self.state_space.street_network.edges_df.node_set.values.tolist(),
        )

    def create_assosciated_state_space_using_postgis(
        self, database, user, password, gamma
    ):
        """Creating a state space using the arguments supplied in constructor."""
        data = PostGISQuery(database, user, password, self.epsg, self.bounding_box)
        street_network = StreetNetwork(data)
        return StreetStateSpace(street_network, gamma)

    def create_assosciated_state_space_using_overpass(self, gamma):
        """Creating a state space using the arguments supplied in constructor."""
        data = OverpassAPIQuery(self.epsg, self.bounding_box)
        street_network = StreetNetwork(data)
        return StreetStateSpace(street_network, gamma)
    
    def create_assosciated_state_space_using_nvdb(self, gamma):
        print(self.bounding_box, self.epsg)
        data = NVDBAPIQuery(self.bounding_box, self.epsg)
        street_network = StreetNetwork(data)
        return StreetStateSpace(street_network, gamma)

    def emission_probability(self, x, y):
        """Emission probability, normally distributed around closest point on edge."""
        distance = self.state_space.street_network.distance_from_point_to_edge(x, y)
        return (2 * pi * self.sigma) ** (-1) * exp(
            -distance ** 2 / (2 * self.sigma ** 2)
        )

    def initial_probability(self, x):
        """Same initial probabilities for all states."""
        return 1 / self.state_space.street_network.edges_df.shape[0]

    def attach_observations(self, observations):
        """Work with new observations."""
        self.most_likely_edge_sequence = None
        self.observations = observations

    def calculate_most_likely_edge_sequence(self):
        """Run Viterbi to infer most likely path."""
        assert self.observations is not None
        self.most_likely_edge_sequence = self.hmm.most_likely_path(
            self.observations.df.point.values
        )

    @property
    def most_likely_edges(self):
        """The edges returned by Viterbi. Not (necessarily connected)."""
        if self.most_likely_edge_sequence is None:
            self.calculate_most_likely_edge_sequence()
        df = self.state_space.street_network.edges_df
        df_list = []
        for edge in self.most_likely_edge_sequence:
            df_list.append(df[df.node_set.map(lambda x: x == edge)])
        return pd.concat(df_list)

    @property
    def most_likely_path(self):
        """Connect the edges returned by Viterbi by finding the shortest path between subsequent edges."""
        if self.most_likely_edge_sequence is None:
            self.calculate_most_likely_edge_sequence()
        path = self.most_likely_edge_sequence
        graph = self.state_space.street_network.graph
        new_path = []
        in_original_path = []
        for prv, nxt in zip(path[:-1], path[1:]):
            if graph[prv[0]][prv[1]]["line"].equals(graph[nxt[1]][nxt[0]]["line"]):
                new_path.append(prv)
                in_original_path.append(True)
            elif any(map(lambda x: x[0] == x[1], product(prv, nxt))):
                new_path.append(prv)
                in_original_path.append(True)
            else:
                new_path.append(prv)
                in_original_path.append(True)
                path_candidates = [
                    shortest_path(graph, s, t, weight="length")
                    for s, t in product(prv, nxt)
                ]
                path_lengths = []
                for path_candidate in path_candidates:
                    path_length = sum(
                        (
                            graph[s][t]["length"]
                            for s, t in zip(path_candidate[:-1], path_candidate[1:])
                        )
                    )
                    path_lengths.append(path_length)
                shortest_path_candidate = path_candidates[np.argmin(path_lengths)]
                edge_sequence = [
                    tuple(sorted([s, t]))
                    for s, t in zip(
                        shortest_path_candidate[:-1], shortest_path_candidate[1:]
                    )
                ]
                for edge in edge_sequence:
                    new_path.append(edge)
                    in_original_path.append(False)

        df = self.state_space.street_network.edges_df
        df_list = []
        for edge in new_path:
            df_list.append(df[df.node_set.map(lambda x: x == edge)])
        return pd.concat(df_list), np.array(in_original_path)

    def plot_street_network(self, margin):
        """Plot the street network."""
        assert self.observations is not None
        xmin, xmax, ymin, ymax = self.observations.bounding_box
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)
        _ = self.state_space.street_network.edges_df.plot(ax=ax)
        _ = self.state_space.street_network.nodes_df.plot(ax=ax)
        fig.show()

    def plot_observations(self, margin):
        """Plot the observations, together with the underlying street network."""
        assert self.observations is not None
        xmin, xmax, ymin, ymax = self.observations.bounding_box
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)
        _ = self.state_space.street_network.edges_df.plot(ax=ax)
        _ = self.state_space.street_network.nodes_df.plot(ax=ax)
        _ = self.observations.df.plot(ax=ax, color="orange")
        fig.show()

    def plot_most_likely_edges(self, margin):
        """Plot the edges, together with observations and underlying street network."""
        assert self.observations is not None
        xmin, xmax, ymin, ymax = self.observations.bounding_box
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_aspect("equal")
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)
        _ = self.state_space.street_network.edges_df.plot(
            ax=ax, linewidth=0.5, alpha=0.3
        )
        _ = self.state_space.street_network.nodes_df.plot(
            ax=ax, alpha=0.1, color="red", markersize=0.2
        )
        _ = self.observations.df.plot(ax=ax, color="orange")
        self.most_likely_edges.plot(ax=ax, color="green", linewidth=3)
        fig.show()

    def plot_most_likely_path(self, margin):
        """Plot the edges, connected by shortest paths, together with observations and street network."""
        assert self.observations is not None
        xmin, xmax, ymin, ymax = self.observations.bounding_box
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_aspect("equal")
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)
        _ = self.state_space.street_network.edges_df.plot(
            ax=ax, linewidth=0.5, alpha=0.3
        )
        _ = self.state_space.street_network.nodes_df.plot(
            ax=ax, alpha=0.1, color="red", markersize=0.2
        )
        _ = self.observations.df.plot(ax=ax, color="orange")
        path_df, in_edges = self.most_likely_path
        _ = path_df[in_edges].plot(ax=ax, color="green", linewidth=3)
        _ = path_df[~in_edges].plot(ax=ax, color="red", linewidth=3)
        fig.show()

    @property
    def gamma(self):
        """Gamma getter."""
        return self.state_space._gamma

    @gamma.setter
    def gamma(self, value):
        """Gamma setter."""
        self.state_space._gamma = value

    @property
    def sigma(self):
        """Sigma getter."""
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        """Sigma setter."""
        self._sigma = value
