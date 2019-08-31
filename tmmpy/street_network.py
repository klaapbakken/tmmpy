import pandas as pd
import geopandas as gpd
import osmnx as ox
import numpy as np
import networkx as nx

from shapely import geometry
from shapely.ops import nearest_points
from itertools import product
from itertools import chain

import matplotlib.pyplot as plt

from shapely.geometry import LineString
from shapely.geometry import Point

from networkx.drawing.nx_pylab import draw_networkx_edges


class StreetNetwork:
    def __init__(self, data_source, crs):
        self.nodes_df = data_source.nodes_df
        self.ways_df = data_source.ways_df
        self.edges_df = self.create_edges_df(self.ways_df, self.nodes_df, crs=crs)
        self.graph = self.create_graph()

    def create_edges_df(self, ways_df, nodes_df, crs):
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
            gdf["node_set"] = pd.Series(
                [tuple(sorted([u, v])) for u, v in zip(gdf.u, gdf.v)]
            )
            gdf_list.append(gdf)

        edges_df = gpd.GeoDataFrame(pd.concat(gdf_list), geometry="line", crs=crs)
        edges_df = edges_df.drop_duplicates(["node_set"])

        edges_df.drop("osmid", axis=1, inplace=True)
        edges_df = (
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

        edges_df["length"] = edges_df.line.map(lambda x: x.length)

        return edges_df

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
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes_df.osmid)
        graph.add_edges_from(
            [
                (edge[0], edge[1], {"length": length, "line": line})
                for edge, length, line in zip(
                    self.edges_df.node_set, self.edges_df.length, self.edges_df.line
                )
            ]
        )
        return graph

    def shortest_path_length_between_nodes(self, l_node, r_node):
        return nx.shortest_path_length(
            self.graph, source=l_node, target=r_node, weight="length"
        )

    def shortest_path_length_between_edges(self, l_edge, r_edge):
        return min(
            (
                self.shortest_path_length_between_nodes(nodes[0], nodes[1])
                for nodes in product(l_edge, r_edge)
            )
        )

    def distance_from_point_to_edge(self, point, edge):
        edge_line = self.edges_df[self.edges_df.node_set == tuple(sorted(edge))].line.iloc[0]
        closest_points = nearest_points(point, edge_line)
        ls = LineString([(p.x, p.y) for p in closest_points])
        return ls.length

    @property
    def node_positions(self):
        return {
            row["osmid"]: [row["x"], row["y"]] for _, row in self.nodes_df.iterrows()
        }