import psycopg2
import geopandas as gpd


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
