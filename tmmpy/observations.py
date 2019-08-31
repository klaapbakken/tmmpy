from data_interface import PostGISQuery
from street_network import StreetNetwork
from state_space import StreetStateSpace

import geopandas as gpd
from shapely.geometry import Point

class GPSObservations():
    def __init__(self, observations_df, longitude_col, latitude_col, time_col, crs):
        self.crs = crs
        self.observations_df = observations_df.rename(
            columns={longitude_col : "x",
                 latitude_col : "y",
                 time_col : "time"})
        df = self.observations_df
        self.bounding_box = (
            df.x.min(),
            df.x.max(),
            df.y.min(),
            df.y.max()
        )
        self.observations_df["point"] = self.observations_df.apply(lambda x: Point(x["x"], x["y"]), axis=1)
        self.observations_df = gpd.GeoDataFrame(
            self.observations_df,
            geometry="point",
            crs={"init" : "epsg:4326"}
        ).to_crs(crs)

        self.observations_df["x"] = self.observations_df.point.map(lambda x: x.x)
        self.observations_df["y"] = self.observations_df.point.map(lambda x: x.y)

    def get_bounding_box(self):
        df = self.observations_df
        return (
            df.x.min(),
            df.x.max(),
            df.y.min(),
            df.y.max()
        )