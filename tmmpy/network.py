import pandas as pd
import geopandas as gpd
import osmnx as ox
import numpy as np
import networkx as nx

from shapely import geometry
from osmapi import OsmApi
from itertools import product
from itertools import chain

import matplotlib.pyplot as plt

from shapely.geometry import LineString
from shapely.geometry import Point

from networkx.drawing.nx_pylab import draw_networkx_edges


class Network:
    def __init__(self, nodes_df, ways_df):
        self.nodes_df = nodes_df
        self.ways_df = ways_df
        self.edges_df = self.create_edges_df(ways_df, nodes_df)
        self.graph = self.create_graph()

    def plot_graph(self, xmin, xmax, ymin, ymax):
        fig, ax = plt.subplots()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        draw_networkx_edges(self.graph, self.node_positions, ax=ax, width=0.1, arrowsize=1, color="blue", alpha=0.2)

    def create_edges_df(self, ways_df, nodes_df):
        gdf_list = list()
        for _, row in ways_df.iterrows():
            osmid = row["osmid"]
            us = row["nodes"][:-1]
            vs = row["nodes"][1:]
            ls = row["linestring"]
            coords = list(ls.coords)
            lines = [LineString([i, j]) for i, j in zip(coords[:-1], coords[1:])]
            gdf = gpd.GeoDataFrame(
                {
                    "u": pd.Series(us, dtype=np.int64),
                    "v": pd.Series(vs, dtype=np.int64),
                    "osmid": osmid,
                    "line": lines,
                },
                geometry="line",
            )
            gdf = gdf.assign(
                node_set=lambda x: pd.Series(map(tuple, sorted(zip(x.u, x.v))))
            )
            gdf_list.append(gdf)

        edges_df = gpd.GeoDataFrame(pd.concat(gdf_list), geometry="line")
        edges_df = edges_df.drop_duplicates(["node_set"])

        edges_df.drop("osmid", axis=1, inplace=True)
        return (
            edges_df.merge(
                nodes_df[["x", "y", "point", "osmid"]], left_on="u", right_on="osmid"
            )
            .rename(columns={"x": "ux", "y": "uy", "point": "u_point"})
            .drop("osmid", axis=1)
            .merge(
                nodes_df[["x", "y", "point", "osmid"]], left_on="v", right_on="osmid"
            )
            .drop("osmid", axis=1)
            .rename(columns={"x": "vx", "y": "vy", "point": "v_point"})
        )

    def _filter_gdf_by_tag(self, gdf, required_keys, required_values):
        # Fix this. Need pairs to match. Will probably work though.
        has_keys = gdf.tags.map(
            lambda x: all(map(lambda key: key in x.keys(), required_keys))
        )
        has_values = gdf.tags.map(
            lambda x: all(map(lambda val: val in x.values(), required_values))
        )
        return gdf[has_keys & has_values]

    def filter_nodes_by_tag(self, required_keys=[], required_values=[]):
        return self._filter_gdf_by_tag(self.nodes_df, required_keys, required_values)

    def filter_ways_by_tag(self, required_keys=[], required_values=[]):
        return self._filter_gdf_by_tag(self.ways_df, required_keys, required_values)

    def create_graph(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes_df.osmid)
        graph.add_edges_from(self.edges_df.node_set)
        return graph

    @property
    def node_positions(self):
        return {
            row["osmid"]: [row["x"], row["y"]] for _, row in self.nodes_df.iterrows()
        }
