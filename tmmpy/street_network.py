import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx

from shapely import geometry
from shapely.ops import nearest_points
from itertools import product
from itertools import chain

import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.geometry import Point

from networkx import connected_component_subgraphs, weakly_connected_component_subgraphs

import fiona

from abc import ABC, abstractmethod


class StreetNetwork(ABC):
    def __init__(self, data, restrict=True):
        assert data.nodes_df.crs == data.ways_df.crs
        self.crs = data.nodes_df.crs
        self.nodes_df = data.nodes_df
        self.ways_df = data.ways_df
        self.bounding_polygon = data.bounding_box.as_polygon

        self.create_edges_df()
        if restrict:
            self.create_restricted_graph()
        else:
            self.create_graph()

        self.trim_graph()
        self.trim_edges_df()
        self.create_linestring_lookup()
        self.create_point_lookup()

    @abstractmethod
    def create_edges_df(self):
        pass

    @abstractmethod
    def create_linestring_lookup(self):
        pass

    @abstractmethod
    def create_point_lookup(self):
        pass

    @abstractmethod
    def create_graph(self):
        pass

    @abstractmethod
    def trim_graph(self):
        pass

    @abstractmethod
    def trim_edges_df(self):
        pass

    @abstractmethod
    def create_restricted_graph(self):
        pass


class UndirectedStreetNetwork(StreetNetwork):
    """Class for representing various data derived from the data source.  
    Intended to support using the data to easily work with street networks as state spaces.
    """

    def create_edges_df(self):
        """Creates a dataframe consisting of the individual segments that make out the ways in the data source."""
        gdf_list = list()
        for _, row in self.ways_df.iterrows():
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
                    "linestring": lines,
                },
                geometry="linestring",
            )
            gdf["node_set"] = pd.Series(
                [tuple(sorted([u, v])) for u, v in zip(gdf.u, gdf.v)]
            )
            gdf_list.append(gdf)

        edges_df = gpd.GeoDataFrame(pd.concat(gdf_list), geometry="linestring", crs=self.crs)
        edges_df = edges_df.drop_duplicates("node_set")

        edges_df.drop("osmid", axis=1, inplace=True)
        edges_df = (
            edges_df.merge(
                self.nodes_df[["x", "y", "point", "osmid"]],
                left_on="u",
                right_on="osmid",
            )
            .rename(columns={"x": "ux", "y": "uy", "point": "u_point"})
            .drop("osmid", axis=1)
            .merge(
                self.nodes_df[["x", "y", "point", "osmid"]],
                left_on="v",
                right_on="osmid",
            )
            .drop("osmid", axis=1)
            .rename(columns={"x": "vx", "y": "vy", "point": "v_point"})
        )

        edges_df["length"] = edges_df.linestring.map(lambda x: x.length)

        self.edges_df = edges_df

    def trim_graph(self):
        self.graph = max(connected_component_subgraphs(self.graph), key=len)

    def create_linestring_lookup(self):
        self.linestring_lookup = {
            tuple(sorted(edge)): line
            for edge, line in zip(self.edges_df.node_set, self.edges_df.linestring)
        }

    def create_point_lookup(self):
        self.point_lookup = {node : point for node, point in zip(self.nodes_df.osmid, self.nodes_df.point)}

    def create_graph(self):
        """Creates a graph, with each node being a node from the data source and each edge being an individual segment
        from the segment that makes out the ways."""
        graph = nx.Graph()
        nodes = pd.concat((self.edges_df.u, self.edges_df.v))
        points = pd.concat((self.edges_df.u_point, self.edges_df.v_point))
        node_df = pd.DataFrame({"nodes" : nodes, "points" : points}).drop_duplicates("nodes")
        graph.add_nodes_from(((node, {"point" : point}) for node, point in zip(node_df.nodes, node_df.points)))
        graph.add_edges_from(
            [
                (edge[0], edge[1], {"length": length, "line": line})
                for edge, length, line in zip(
                    self.edges_df.node_set, self.edges_df.length, self.edges_df.linestring
                )
            ]
        )
        self.graph = graph

    def create_restricted_graph(self):
        bounding_polygon = self.bounding_polygon
        edges_df = self.edges_df
        edges_df = edges_df.assign(u_contained=edges_df.u_point.map(lambda x: bounding_polygon.contains(x)))
        edges_df = edges_df.assign(v_contained=edges_df.v_point.map(lambda x: bounding_polygon.contains(x)))
        edges_df = edges_df.assign(both_contained=edges_df.u_contained & edges_df.v_contained)
        restricted_edges_df = edges_df[edges_df.both_contained == True]

        nodes = pd.concat((restricted_edges_df.u, restricted_edges_df.v))
        points = pd.concat((restricted_edges_df.u_point, restricted_edges_df.v_point))
        node_df = pd.DataFrame({"nodes" : nodes, "points" : points}).drop_duplicates("nodes")

        g = nx.Graph()
        g.add_nodes_from(((node, {"point" : point}) for node, point in zip(node_df.nodes, node_df.points)))
        g.add_edges_from(((x["u"], x["v"], {"line" : x["linestring"], "length" : x["length"]}) for _, x in restricted_edges_df.iterrows()))
        self.graph = g
        self.edges_df = restricted_edges_df

    def trim_edges_df(self):
        """Remove rows in edges_df that are not part of the largest connected component."""
        edges_in_graph = set(map(lambda x: tuple(sorted(x)), self.graph.edges.keys()))
        self.edges_df = self.edges_df[self.edges_df.node_set.isin(edges_in_graph)]

    def distance_from_point_to_edge(self, point: Point, edge: tuple):
        """Find the distance from a point to the nearest point on the edge."""
        line = self.linestring_lookup[tuple(sorted(edge))]
        closest_points = nearest_points(point, line)
        ls = LineString([(p.x, p.y) for p in closest_points])
        return ls.length


class DirectedStreetNetwork(StreetNetwork):
    def create_edges_df(self):
        intermediate_dfs = []
        for _, row in self.ways_df.iterrows():
            point_coords = list(row["linestring"].coords)
            segment = [
                LineString([i, j]) for i, j in zip(point_coords[:-1], point_coords[1:])
            ]
            nodes = row["nodes"]
            edges = [
                (from_node, to_node)
                for from_node, to_node in zip(nodes[:-1], nodes[1:])
            ]
            edges += list(map(lambda x: tuple(reversed(x)), edges))
            segment += segment
            u = list(map(lambda x: x[0], edges))
            v = list(map(lambda x: x[1], edges))
            df = pd.DataFrame(
                {"u": u, "v": v, "node_set": edges, "linestring": segment}
            )
            intermediate_dfs.append(df)
        edges_df = gpd.GeoDataFrame(
            pd.concat(intermediate_dfs), geometry="linestring", crs=self.crs
        )
        edges_df["length"] = edges_df.linestring.map(lambda x: x.length)
        edges_df = edges_df.merge(self.nodes_df[["osmid", "point"]], left_on="u", right_on="osmid").drop("osmid", axis=1).rename({"point" : "u_point"}, axis=1)
        edges_df = edges_df.merge(self.nodes_df[["osmid", "point"]], left_on="v", right_on="osmid").drop("osmid", axis=1).rename({"point" : "v_point"}, axis=1)

        self.edges_df = edges_df

    def create_graph(self):
        graph = nx.DiGraph()
        nodes = pd.concat((self.edges_df.u, self.edges_df.v))
        points = pd.concat((self.edges_df.u_point, self.edges_df.v_point))
        node_df = pd.DataFrame({"nodes" : nodes, "points" : points}).drop_duplicates("nodes")
        graph.add_nodes_from(((node, {"point" : point}) for node, point in zip(node_df.nodes, node_df.points)))
        graph.add_edges_from(
            [
                (edge[0], edge[1], {"length": length, "line" : line})
                for edge, length, line in zip(self.edges_df.node_set, self.edges_df.length, self.edges_df.linestring)
            ]
        )
        self.graph = graph

    def create_linestring_lookup(self):
        self.linestring_lookup = {
            tuple(sorted(edge)): line
            for edge, line in zip(self.edges_df.node_set, self.edges_df.linestring)
        }

    def create_point_lookup(self):
        self.point_lookup = {node : point for node, point in zip(self.nodes_df.osmid, self.nodes_df.point)}

    def trim_graph(self):
        self.graph = max(weakly_connected_component_subgraphs(self.graph), key=len)

    def trim_edges_df(self):
        """Remove rows in edges_df that are not part of the largest connected component."""
        edges_in_graph = list(self.graph.edges.keys())
        self.edges_df = self.edges_df[self.edges_df.node_set.isin(edges_in_graph)]

    def create_restricted_graph(self):
        bounding_polygon = self.bounding_polygon
        edges_df = self.edges_df
        edges_df = edges_df.assign(u_contained=edges_df.u_point.map(lambda x: bounding_polygon.contains(x)))
        edges_df = edges_df.assign(v_contained=edges_df.v_point.map(lambda x: bounding_polygon.contains(x)))
        edges_df = edges_df.assign(both_contained=edges_df.u_contained & edges_df.v_contained)
        restricted_edges_df = edges_df[edges_df.both_contained == True]

        nodes = pd.concat((restricted_edges_df.u, restricted_edges_df.v))
        points = pd.concat((restricted_edges_df.u_point, restricted_edges_df.v_point))
        node_df = pd.DataFrame({"nodes" : nodes, "points" : points}).drop_duplicates("nodes")
        pd.DataFrame({"nodes" : node_df.nodes, "points" : node_df.points}).drop_duplicates("nodes")

        g = nx.DiGraph()
        g.add_nodes_from(((node, {"point" : point}) for node, point in zip(nodes, points)))
        g.add_edges_from(((x["u"], x["v"], {"line" : x["linestring"], "length" : x["length"]}) for _, x in restricted_edges_df.iterrows()))
        self.graph = g
        self.edges_df = restricted_edges_df

    def distance_from_point_to_edge(self, point: Point, edge: tuple):
        """Find the distance from a point to the nearest point on the edge."""
        line = self.linestring_lookup[tuple(sorted(edge[0]))] 
        closest_points = nearest_points(point, line)
        ls = LineString([(p.x, p.y) for p in closest_points])
        return ls.length
