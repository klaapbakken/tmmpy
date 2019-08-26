import psycopg2
import geopandas as gpd
import osmnx as osmnx
import numpy as np

from shapely import geometry
from itertools import product


class PostGISQuery():
    def __init__(self, database, user, password):
        self.con = psycopg2.connect(
            database=database,
            user=user,
            password=password,
            host="localhost"
        )
        print("Database connection established.")
        self.nodes_df = None
        self.ways_df = None

    def get_nodes_by_id(self, ids):
        node_ids_string = ", ".join(map(str, ids))
        node_query = f"""
        SELECT *
        FROM nodes
        WHERE nodes.id IN ({node_ids_string});
        """
        self.nodes_df = gpd.GeoDataFrame.from_postgis(
            node_query,
            self.con,
            geom_col="geom"
            )

    def get_ways_intersecting(self, xmin, xmax, ymin, ymax):
        ways_query = f"""
        SELECT *
        FROM ways 
        WHERE ways.linestring 
            && 
            ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax});
        """
        self.ways_df = gpd.GeoDataFrame.from_postgis(
            ways_query,
            self.con,
            geom_col="linestring"
            )
        
    def get_ways_containing(self, xmin, xmax, ymin, ymax):
        ways_query = f"""
        SELECT *
        FROM ways 
        WHERE ways.linestring 
            @ 
            ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax});
        """
        self.ways_df = gpd.GeoDataFrame.from_postgis(
            ways_query,
            self.con,
            geom_col="linestring"
            )

class OSMApiQuery():
    def __init__():
        self.nodes_df = None
        self.ways_df = None
    
    def get_ways_containing(self, xmin, xmax, ymin, ymax):
        xs = [xmin, xmax]
        ys = [ymin, ymax]
        points = np.array(list(product(xs, ys)))[np.array([0,1,3, 2])]
        bbox = geometry.Polygon(points)
        road_network = ox.core.graph_from_polygon(bbox)
        self.nodes_df, self.ways_df = ox.save_load.graph_to_gdfs(road_network)


