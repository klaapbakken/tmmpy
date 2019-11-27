import psycopg2
import overpy

import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx

from shapely import geometry
from itertools import product
from itertools import chain

from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.wkt import loads

import fiona
import requests

from fiona.transform import transform
from shapely.geometry import Polygon

class BoundingBox:
    def __init__(self, bounding_box, in_epsg, out_epsg):
        self.in_crs = fiona.crs.from_epsg(in_epsg)
        self.out_crs = fiona.crs.from_epsg(out_epsg)
        xmin, xmax, ymin, ymax = bounding_box
        in_polygon = Polygon([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
        self.out_polygon = self._transform_polygon(in_polygon, self.in_crs, self.out_crs)
    
    def _transform_polygon(self, polygon, in_crs, out_crs):
        xs = np.array(list(polygon.boundary.coords))[:, 0]
        ys = np.array(list(polygon.boundary.coords))[:, 1]
        nxs, nys = transform(in_crs, out_crs, xs, ys)
        return Polygon(list(zip(nxs, nys)))
    
    def polygon_to_epsg(self, out_epsg):
        out_crs = fiona.crs.from_epsg(out_epsg)
        return self._transform_polygon(self.out_polygon, self.out_crs, out_crs)
        
    @property
    def as_polygon(self):
        return self.out_polygon
        
    @property
    def as_tuple(self):
        bounding_box = self.out_polygon.bounds
        return (bounding_box[0], bounding_box[2], bounding_box[1], bounding_box[3])


class PostGISQuery:
    """
    A class that enables easily obtaining data from a PostgreSQL database (with PostGIS extension)
    containing OSM data. Assumes that database uses EPSG:4326 (longitude-latiude).

    Parameters:
    ---
    database -- Name of the database
    user -- Name of user with access to database
    password -- Password for user
    epsg -- Coordinate reference system the data should be converted to
    bounding_box -- Tuple of coordinates (xmin, xmax, ymin, ymax)
    nodes_table -- Name of the table where nodes are stored
    ways_table -- Name of the table where ways are stored
    filter_dictionary -- Dictionary that specifies which keys and values are allowed in tags. Ways that do not match any of the key-value pairs are removed
    """

    def __init__(
        self,
        database: str,
        user: str,
        password: str,
        epsg: str,
        bounding_box: tuple,
        nodes_table: str = "nodes",
        ways_table: str = "ways",
        filter_dictionary: dict = {},
    ):
        """See class documentation."""
        self.LONLAT_CRS = fiona.crs.from_epsg(4326)
        self.LONLAT_EPSG = 4326
        self.con = psycopg2.connect(
            database=database, user=user, password=password, host="localhost"
        )
        self.nodes_df = None
        self.ways_df = None
        self.epsg = epsg
        self.crs = fiona.crs.from_epsg(self.epsg)
        self.filter_dictionary = filter_dictionary
        self.nodes_table = nodes_table
        self.ways_table = ways_table
        self.xmin, self.xmax, self.ymin, self.ymax = bounding_box
        self.query_bounding_box()
        self.bounding_box = BoundingBox(bounding_box, self.LONLAT_EPSG, epsg)

    def get_nodes_by_id(self, ids: list):
        """Retrieve nodes by their ID."""
        node_ids_string = ", ".join(map(str, ids))
        node_query = f"""
        SELECT *
        FROM {self.nodes_table}
        WHERE {self.nodes_table}.id IN ({node_ids_string});
        """
        self.nodes_df = gpd.GeoDataFrame.from_postgis(
            node_query, self.con, geom_col="geom", crs=self.LONLAT_CRS
        )

        self.nodes_df["x"] = self.nodes_df.geom.map(lambda x: x.x)
        self.nodes_df["y"] = self.nodes_df.geom.map(lambda x: x.y)

        self._parse_tags(self.nodes_df)

        self.nodes_df = (
            self.nodes_df.rename(columns={"id": "osmid", "geom": "point"})
            .drop(["version", "user_id", "tstamp", "changeset_id"], axis=1)
            .set_geometry("point")
        )

        self.nodes_df.to_crs(epsg=self.epsg, inplace=True)
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
            ways_query, self.con, geom_col="linestring", crs=self.LONLAT_CRS
        )

        self.ways_df = (
            self.ways_df.rename(columns={"id": "osmid"})
            .drop(["version", "user_id", "tstamp", "changeset_id", "bbox"], axis=1)
            .to_crs(epsg=self.epsg)
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

    def transform(self, epsg):
        self.ways_df.to_crs(crs=fiona.crs.from_epsg(epsg), inplace=True)
        self.nodes_df.to_crs(crs=fiona.crs.from_epsg(epsg), inplace=True)
        self.nodes_df["x"] = self.nodes_df.point.map(lambda x: x.x)
        self.nodes_df["y"] = self.nodes_df.point.map(lambda x: x.y)
        return self

    @property
    def node_ids(self):
        """Get the ID of the all the nodes in the ways dataframe."""
        return np.array(list(chain(*self.ways_df.nodes.values)))


class OverpassAPIQuery:
    """
    A class that enables easily obtaining data from the Overpass API.

    Parameters:
    ---
    epsg -- Coordinate reference system the data should be converted to
    bounding_box -- Tuple of coordinates (xmin, xmax, ymin, ymax)
    filter_dictionary -- Dictionary that specifies which keys and values are allowed in tags. Ways that do not match any of the key-value pairs are removed
    """

    def __init__(
        self,
        epsg: int,
        bounding_box: tuple,
        nodes_table: str = "nodes",
        ways_table: str = "ways",
        filter_dictionary: dict = {},
    ):
        """See class documentation."""
        self.LONLAT_CRS = fiona.crs.from_epsg(4326)
        self.LONLAT_EPSG = 4326
        self.api = overpy.Overpass()
        self.nodes_df = None
        self.ways_df = None
        self.epsg = epsg
        self.crs = fiona.crs.from_epsg(self.epsg)
        self.filter_dictionary = filter_dictionary
        self.xmin, self.xmax, self.ymin, self.ymax = bounding_box
        self.query_bounding_box()
        self.bounding_box = BoundingBox(bounding_box, self.LONLAT_EPSG, epsg)

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

        self.nodes_df.to_crs(epsg=self.epsg, inplace=True)
        self.update_node_coordinates()

        self.ways_df["linestring"] = pd.Series(lines)
        self.ways_df = gpd.GeoDataFrame(
            self.ways_df, geometry="linestring", crs=self.LONLAT_CRS
        )
        self.ways_df.to_crs(epsg=self.epsg, inplace=True)

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
            crs=self.LONLAT_CRS,
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
            crs=self.LONLAT_CRS,
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

    def transform(self, epsg):
        self.ways_df.to_crs(crs=fiona.crs.from_epsg(epsg), inplace=True)
        self.nodes_df.to_crs(crs=fiona.crs.from_epsg(epsg), inplace=True)
        self.nodes_df["x"] = self.nodes_df.point.map(lambda x: x.x)
        self.nodes_df["y"] = self.nodes_df.point.map(lambda x: x.y)
        return self


class NVDBAPIQuery:
    def __init__(self, utm33_bounding_box, epsg):
        self.endpoint = "https://www.vegvesen.no/nvdb/api/v2/vegnett/lenker"
        self.headers = {
            "X-Client": "tmmpy",
            "X-Kontaktperson": "oyvind.klaapbakken@gmail.com",
            "Accept": "application/vnd.vegvesen.nvdb-v2+json",
        }
        self.xmin, self.xmax, self.ymin, self.ymax = utm33_bounding_box
        self.epsg = epsg
        self.responses = []
        self.initial_query()
        self.parse_responses()
        self.create_dfs()
        self.transform(epsg=self.epsg)
        self.bounding_box = BoundingBox(utm33_bounding_box, 32633, epsg)

    def initial_query(self):
        params = {
            "kartutsnitt": f"{self.xmin},{self.ymin},{self.xmax},{self.ymax}",
            "srid": 32633,
        }
        self.initial_params = params
        response = requests.get(self.endpoint, params=params, headers=self.headers)
        self.responses.append(response.json())
        self.continue_query()

    def continue_query(self):
        previous_response = self.responses[-1]
        if type(previous_response) != dict:
            print(previous_response)
            raise ValueError
        new_params = self.initial_params
        new_params["start"] = previous_response["metadata"]["neste"]["start"]
        response = requests.get(self.endpoint, params=new_params, headers=self.headers)
        self.responses.append(response.json())
        if self.responses[-1]["metadata"]["returnert"] > 0:
            self.continue_query()

    def parse_responses(self):
        geoms = []
        ids = []
        us = []
        vs = []
        for response in self.responses:
            for objekt in response["objekter"]:
                geoms.append(objekt["geometri"])
                us.append(objekt["startnode"])
                vs.append(objekt["sluttnode"])
                ids.append(objekt["veglenkeid"])
        self.raw_df = gpd.GeoDataFrame(
            {
                "linestring": pd.Series(
                    list(
                        map(
                            lambda x: LineString([xy[:2] for xy in x.coords]),
                            map(lambda x: loads(x["wkt"]), geoms),
                        )
                    )
                ),
                "osmid": pd.Series(ids, dtype=np.int64),
                "u": pd.Series(us, dtype=np.int64),
                "v": pd.Series(vs, dtype=np.int64),
            },
            geometry="linestring",
            crs=fiona.crs.from_epsg(32633),
        )

    def create_dfs(self):
        def id_generator(ids_in_use):
            i = 0
            while True:
                if i not in ids_in_use:
                    yield i
                i += 1

        points = []
        ids = []
        ids_in_use = set(self.raw_df.u.values.tolist()).union(
            set(self.raw_df.v.values.tolist())
        )
        id_gen = id_generator(ids_in_use)
        nodes = []
        for _, row in self.raw_df.iterrows():
            linestring_coords = list(row["linestring"].coords)
            row_ids = []
            for i, xy in enumerate(linestring_coords):
                points.append(Point(xy))
                if i == 0:
                    ids.append(row["u"])
                elif i == len(linestring_coords) - 1:
                    ids.append(row["v"])
                else:
                    ids.append(next(id_gen))
                row_ids.append(ids[-1])
            nodes.append(row_ids)

        ways_df = self.raw_df[["linestring", "osmid"]].copy()
        ways_df["nodes"] = pd.Series(nodes)
        ways_df["osmid"] = pd.Series(list(range(ways_df.shape[0])), dtype=np.int64)
        ways_df.crs = fiona.crs.from_epsg(32633)
        ways_df.set_geometry("linestring", inplace=True)

        nodes_df = gpd.GeoDataFrame(
            {"point": pd.Series(points), "osmid": pd.Series(ids, dtype=np.int64)},
            geometry="point",
            crs=fiona.crs.from_epsg(32633),
        )

        nodes_df["x"] = nodes_df.point.map(lambda x: x.x)
        nodes_df["y"] = nodes_df.point.map(lambda x: x.y)

        self.ways_df = ways_df
        self.nodes_df = nodes_df.drop_duplicates("osmid").reset_index(drop=True)

    def transform(self, epsg):
        self.ways_df = self.ways_df.to_crs(crs=fiona.crs.from_epsg(epsg))
        #Error occurs in the following line.
        self.nodes_df = self.nodes_df.to_crs(crs=fiona.crs.from_epsg(epsg))
        self.nodes_df["x"] = self.nodes_df.point.map(lambda x: x.x)
        self.nodes_df["y"] = self.nodes_df.point.map(lambda x: x.y)
        return self
