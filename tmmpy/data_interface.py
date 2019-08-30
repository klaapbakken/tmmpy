import psycopg2
import overpy

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
from shapely.geometry import Point


class PostGISQuery:
    def __init__(self, database, user, password, crs):
        self.con = psycopg2.connect(
            database=database, user=user, password=password, host="localhost"
        )
        print("Database connection established.")
        self.nodes_df = None
        self.ways_df = None
        self.crs = crs

    def get_nodes_by_id(self, ids):
        node_ids_string = ", ".join(map(str, ids))
        node_query = f"""
        SELECT *
        FROM nodes
        WHERE nodes.id IN ({node_ids_string});
        """
        self.nodes_df = gpd.GeoDataFrame.from_postgis(
            node_query, self.con, geom_col="geom", crs={"init": "epsg:4326"}
        )

        self.nodes_df["x"] = self.nodes_df.geom.map(lambda x: x.x)
        self.nodes_df["y"] = self.nodes_df.geom.map(lambda x: x.y)

        self._parse_tags(self.nodes_df)

        self.nodes_df = (
            self.nodes_df.rename(columns={"id": "osmid", "geom": "point"})
            .drop(["version", "user_id", "tstamp", "changeset_id"], axis=1)
            .set_geometry("point")
        )

        self.nodes_df.to_crs(self.crs, inplace=True)
        self.update_node_coordinates()

    def update_node_coordinates(self):
        self.nodes_df["x"] = self.nodes_df.point.map(lambda x: x.x)
        self.nodes_df["y"] = self.nodes_df.point.map(lambda x: x.y)

    def get_ways_intersecting(self, xmin, xmax, ymin, ymax):
        ways_query = f"""
        SELECT *
        FROM ways 
        WHERE ways.linestring 
            && 
            ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax});
        """
        self.ways_df = gpd.GeoDataFrame.from_postgis(
            ways_query, self.con, geom_col="linestring", crs={"init": "epsg:4326"}
        )

        self.ways_df = (
            self.ways_df.rename(columns={"id": "osmid"})
            .drop(["version", "user_id", "tstamp", "changeset_id", "bbox"], axis=1)
            .to_crs(self.crs)
        )

        self._parse_tags(self.ways_df)

        self.get_nodes_by_id(self.node_ids)

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

    @property
    def node_ids(self):
        return np.array(list(chain(*self.ways_df.nodes.values)))


class OverpassApiQuery:
    def __init__(self, crs):
        self.api = overpy.Overpass()

        self.nodes_df = None
        self.ways_df = None
        self.crs = crs

    def get_ways_intersecting(self, xmin, xmax, ymin, ymax):
        way_response = self.api.query(
            f"""
            [out:json];
            way({ymin}, {xmin}, {ymax}, {xmax});
            out body;
        """
        )

        self.ways_df = self.parse_way_response(way_response)

        self.nodes_df = self.get_nodes_in(xmin, xmax, ymin, ymax)
        self.add_missing_nodes()

        self.nodes_df.to_crs(self.crs, inplace=True)
        self.update_node_coordinates()

        lines = []
        for _, row in self.ways_df.iterrows():
            gdf = gpd.GeoDataFrame()
            nodes = row["nodes"]
            gdf["node"] = nodes
            gdf = gdf.merge(self.nodes_df, left_on="node", right_on="osmid")
            line = LineString([(row["x"], row["y"]) for _, row in gdf.iterrows()])
            lines.append(line)

        self.ways_df["linestring"] = lines
        self.ways_df = self.ways_df.set_geometry("linestring")
        self.ways_df.crs = {"init": self.crs}

    def get_nodes_in(self, xmin, xmax, ymin, ymax):
        node_response = self.api.query(
            f"""
            [out:json];
            node({ymin}, {xmin}, {ymax}, {xmax});
            out body;
        """
        )

        return self.parse_node_response(node_response)

    def get_nodes_by_id(self, ids):

        ids_str = ",".join(map(str, ids))

        nodes_response = self.api.query(
            f"""
        node(id:{ids_str});
        out;
        """
        )

        return self.parse_node_response(nodes_response)

    def update_node_coordinates(self):
        self.nodes_df["x"] = self.nodes_df.point.map(lambda x: x.x)
        self.nodes_df["y"] = self.nodes_df.point.map(lambda x: x.y)

    def parse_node_response(self, response):
        osmids = []
        xs = []
        ys = []
        tag_dicts = []
        geoms = []
        for node in response.nodes:
            osmid = node.id
            osmids.append(osmid)
            x = node.lon
            xs.append(x)
            y = node.lat
            ys.append(y)
            tags = node.tags
            tag_dicts.append(tags)
            geom = Point(x, y)
            geoms.append(geom)

        return gpd.GeoDataFrame(
            {
                "osmid": pd.Series(osmids, dtype=np.int64),
                "x": pd.Series(xs, dtype=np.float64),
                "y": pd.Series(ys, dtype=np.float64),
                "tags": pd.Series(tag_dicts),
                "point": pd.Series(geoms),
            },
            geometry="point",
            crs={"init": "epsg:4326"},
        )

    def parse_way_response(self, response):
        osmids = []
        node_lists = []
        tag_dicts = []
        for way in response.ways:
            osmid = way.id
            osmids.append(osmid)
            nodes = way._node_ids
            node_lists.append(nodes)
            tags = way.tags
            tag_dicts.append(tags)

        return gpd.GeoDataFrame(
            {"osmid": osmids, "nodes": node_lists, "tags": tag_dicts}
        )

    def add_missing_nodes(self):
        nodes_in_ways = pd.Series(chain(*self.ways_df.nodes)).unique()
        missing_nodes = []
        for node in nodes_in_ways:
            if node not in self.nodes_df.osmid.values:
                missing_nodes.append(node)

        missing_nodes_df = self.get_nodes_by_id(missing_nodes)

        self.nodes_df = gpd.GeoDataFrame(
            pd.concat([self.nodes_df, missing_nodes_df]),
            geometry="point",
            crs={"init": "epsg:4326"},
        )
