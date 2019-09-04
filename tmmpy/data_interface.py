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
    """
    A class that enables easily obtaining data from a PostgreSQL database (with PostGIS extension)
    containing OSM data. Assumes that database uses EPSG:4326 (longitude-latiude).

    Parameters:
    ---
    database -- Name of the database \n
    user -- Name of user with access to database \n
    password -- Password for user \n
    crs -- Coordinate reference system the data should be converted to \n
    bounding_box -- Tuple of coordinates (xmin, xmax, ymin, ymax) \n
    nodes_table -- Name of the table where nodes are stored \n
    ways_table -- Name of the table where ways are stored \n
    filter_dictionary -- Dictionary that specifies which keys and values are allowed in tags. Ways that do not match any of the key-value pairs are removed
    """

    def __init__(
        self,
        database: str,
        user: str,
        password: str,
        crs: str,
        bounding_box: tuple,
        nodes_table: str = "nodes",
        ways_table: str = "ways",
        filter_dictionary: dict = {},
    ):
        """See class documentation."""
        self.con = psycopg2.connect(
            database=database, user=user, password=password, host="localhost"
        )
        self.nodes_df = None
        self.ways_df = None
        self.crs = crs
        self.filter_dictionary = filter_dictionary
        self.nodes_table = nodes_table
        self.ways_table = ways_table
        self.xmin, self.xmax, self.ymin, self.ymax = bounding_box
        self.query_bounding_box()

    def get_nodes_by_id(self, ids: list):
        """Retrieve nodes by their ID."""
        node_ids_string = ", ".join(map(str, ids))
        node_query = f"""
        SELECT *
        FROM {self.nodes_table}
        WHERE {self.nodes_table}.id IN ({node_ids_string});
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
        """Update x- and y-columns to match x- and y-attributes of point-column."""
        self.nodes_df["x"] = self.nodes_df.point.map(lambda x: x.x)
        self.nodes_df["y"] = self.nodes_df.point.map(lambda x: x.y)

    def query_bounding_box(self):
        """Get ways intersecting a polygon bounded by the bounding box. 
        Also gets all nodes contained by the returned ways.
        """
        ways_query = f"""
        SELECT *
        FROM {self.ways_table}
        WHERE ST_Intersects(
            {self.ways_table}.linestring,
            ST_MakeEnvelope({self.xmin}, {self.ymin}, {self.xmax}, {self.ymax}, 4326)
            );
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

        self.filter_ways_by_tags()

        self.get_nodes_by_id(self.node_ids)

    def _parse_tags(self, gdf: gpd.GeoDataFrame):
        """Reformatting the tags-column."""
        assert gdf is not None
        gdf["tags"] = gdf.tags.map(self._parse_tag)

    def _parse_tag(self, tag: str):
        """Reformatting a tag-string."""
        parsed = list(
            map(
                lambda x: x.strip('"'),
                chain(*map(lambda x: x.split("=>"), tag.split(", "))),
            )
        )
        keys = parsed[::2]
        values = parsed[1::2]
        return {k: v for k, v in zip(keys, values)}

    def filter_ways_by_tags(self):
        """Remove ways that do not contain one of the key-value pairs in the filter dictionary."""
        keep = self.ways_df.tags.map(
            lambda x: self.filter_tags(self.filter_dictionary, x)
        )
        self.ways_df = self.ways_df[keep]

    def filter_tags(self, filter_dictionary: dict, tags: dict):
        """Check if a tag-dictionary contains either of the key-value pairs in filter_dictionary."""
        bool_set = set()
        for key, values in filter_dictionary.items():
            has_key = key in tags
            if has_key:
                has_value = len(set(tags[key]).intersection(set(values))) > 0
            else:
                has_value = False
            bool_set.add(has_key and has_value)
        return any(bool_set) or len(bool_set) == 0

    @property
    def node_ids(self):
        """Get the ID of the all the nodes in the ways dataframe."""
        return np.array(list(chain(*self.ways_df.nodes.values)))


class OverpassAPIQuery:
    """
    A class that enables easily obtaining data from the Overpass API.

    Parameters:
    ---
    crs -- Coordinate reference system the data should be converted to
    bounding_box -- Tuple of coordinates (xmin, xmax, ymin, ymax)
    filter_dictionary -- Dictionary that specifies which keys and values are allowed in tags. Ways that do not match any of the key-value pairs are removed
    """

    def __init__(
        self,
        crs: str,
        bounding_box: tuple,
        nodes_table: str = "nodes",
        ways_table: str = "ways",
        filter_dictionary: dict = {},
    ):
        """See class documentation."""
        self.api = overpy.Overpass()
        self.nodes_df = None
        self.ways_df = None
        self.crs = crs
        self.filter_dictionary = filter_dictionary
        self.xmin, self.xmax, self.ymin, self.ymax = bounding_box
        self.query_bounding_box()

    def query_bounding_box(self):
        """Get ways without bounding box, as well as nodes within said ways."""
        way_response = self.api.query(
            f"""
            [out:json];
            way({self.ymin}, {self.xmin}, {self.ymax}, {self.xmax});
            out body;
        """
        )

        self.ways_df = self.parse_way_response(way_response)
        self.filter_ways_by_tags()

        self.nodes_df = self.get_nodes_in()
        self.add_missing_nodes()
        lines = []
        for _, row in self.ways_df.iterrows():
            df = pd.DataFrame()
            nodes = row["nodes"]
            df["node"] = nodes
            df = df.merge(self.nodes_df, left_on="node", right_on="osmid")
            line = LineString([(row["x"], row["y"]) for _, row in df.iterrows()])
            lines.append(line)

        self.nodes_df.to_crs(self.crs, inplace=True)
        self.update_node_coordinates()

        self.ways_df["linestring"] = pd.Series(lines)
        self.ways_df = gpd.GeoDataFrame(
            self.ways_df, geometry="linestring", crs={"init": "epsg:4326"}
        )
        self.ways_df.to_crs(self.crs, inplace=True)

    def get_nodes_in(self):
        """Get nodes in bounding box."""
        node_response = self.api.query(
            f"""
            [out:json];
            node({self.ymin}, {self.xmin}, {self.ymax}, {self.xmax});
            out body;
        """
        )

        return self.parse_node_response(node_response)

    def get_nodes_by_id(self, ids):
        """Get nodes by their ID."""

        ids_str = ",".join(map(str, ids))

        nodes_response = self.api.query(
            f"""
        node(id:{ids_str});
        out;
        """
        )

        return self.parse_node_response(nodes_response)

    def update_node_coordinates(self):
        """Update x- and y-columns to match x- and y-attributes of point-column."""
        self.nodes_df["x"] = self.nodes_df.point.map(lambda x: x.x)
        self.nodes_df["y"] = self.nodes_df.point.map(lambda x: x.y)

    def parse_node_response(self, response):
        """Parsing the response obtained from the Overpass API when requesting nodes."""
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
        """Parsing the response obtained from the Overpass API when requesting ways."""
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

        return pd.DataFrame({"osmid": osmids, "nodes": node_lists, "tags": tag_dicts})

    def add_missing_nodes(self):
        """If a node in a way is not obtained by getting nodes within bounding box, get it by querying by ID."""
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

    def filter_ways_by_tags(self):
        """Remove ways that do not contain one of the key-value pairs in the filter dictionary."""
        keep = self.ways_df.tags.map(
            lambda x: self.filter_tags(self.filter_dictionary, x)
        )
        self.ways_df = self.ways_df[keep]

    def filter_tags(self, filter_dictionary: dict, tags: dict):
        """Check if a tag-dictionary contains either of the key-value pairs in filter_dictionary."""
        bool_set = set()
        for key, values in filter_dictionary.items():
            has_key = key in tags
            if has_key:
                has_value = len(set(tags[key]).intersection(set(values))) > 0
            else:
                has_value = False
            bool_set.add(has_key and has_value)
        return any(bool_set) or len(bool_set) == 0
