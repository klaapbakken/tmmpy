import psycopg2
import pandas as pd
import geopandas as gpd
import osmnx as ox
import numpy as np
import networkx as nx

from shapely import geometry
from osmapi import OsmApi
from itertools import product
from itertools import chain

from shapely.geometry import LineString


class PostGISQuery:
    def __init__(self, database, user, password):
        self.con = psycopg2.connect(
            database=database, user=user, password=password, host="localhost"
        )
        print("Database connection established.")
        self.nodes_df = None
        self.ways_df = None
        self.edges_df = None
        self.graph = None

    def get_nodes_by_id(self, ids):
        node_ids_string = ", ".join(map(str, ids))
        node_query = f"""
        SELECT *
        FROM nodes
        WHERE nodes.id IN ({node_ids_string});
        """
        self.nodes_df = gpd.GeoDataFrame.from_postgis(
            node_query, self.con, geom_col="geom"
        )
        x_col = self.nodes_df.geometry.map(lambda x: x.x)
        y_col = self.nodes_df.geometry.map(lambda x: x.y)
        self.nodes_df["x"] = x_col
        self.nodes_df["y"] = y_col
        self._parse_tags(self.nodes_df)

    def get_ways_intersecting(self, xmin, xmax, ymin, ymax):
        ways_query = f"""
        SELECT *
        FROM ways 
        WHERE ways.linestring 
            @ 
            ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax});
        """
        self.ways_df = gpd.GeoDataFrame.from_postgis(
            ways_query, self.con, geom_col="linestring"
        )

        self._parse_tags(self.ways_df)

    def _parse_tags(self, gdf):
        assert gdf is not None
        gdf["tags"] = gdf.tags.map(self._parse_tag)

    def _parse_tag(self, tag):
        parsed = list(
            map(
                lambda x: x.strip('"'),
                chain(*map(lambda x: x.split("=>"), tag.split(", "))),
            )
        )
        keys = parsed[::2]
        values = parsed[1::2]
        return {k: v for k, v in zip(keys, values)}

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

    def create_edge_df(self):
        gdf_list = list()
        for _, row in self.ways_df.iterrows():
            osmid = row["id"]
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

        self.edges_df = gpd.GeoDataFrame(pd.concat(gdf_list), geometry="line")
        self.edges_df = self.edges_df.drop_duplicates(["node_set"])

        self.edges_df = (
            self.edges_df.merge(
                self.nodes_df[["x", "y", "geom", "id"]], left_on="u", right_on="id"
            )
            .rename(columns={"x": "ux", "y": "uy", "geom": "u_point"})
            .drop("id", axis=1)
            .merge(self.nodes_df[["x", "y", "geom", "id"]], left_on="v", right_on="id")
            .rename(columns={"x": "vx", "y": "vy", "geom": "v_point"})
            .drop("id", axis=1)
        )

    def create_graph():
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes_df.id)
        self.graph.add_edges_from(self.edges_df.node_set)

    @property
    def node_ids(self):
        return np.array(list(chain(*self.ways_df.nodes.values)))


class OverpassApiQuery:
    def __init__(self):
        self.road_network = None
        self.nodes_df = None
        self.ways_df = None

    def get_highway(self, xmin, xmax, ymin, ymax):
        xs = [xmin, xmax]
        ys = [ymin, ymax]
        points = np.array(list(product(xs, ys)))[np.array([0, 1, 3, 2])]
        bbox = geometry.Polygon(points)
        self.road_network = ox.core.graph_from_polygon(bbox)
        self.nodes_df, self.ways_df = ox.save_load.graph_to_gdfs(self.road_network)
