from data_interface import PostGISQuery
from street_network import StreetNetwork
from state_space import StreetStateSpace

import geopandas as gpd
from shapely.geometry import Point

class GPSObservations():
    def __init__(self, observations_df, longitude_col, latitude_col, time_col, crs):
        self.crs = crs
        self._lonlat_df = observations_df.rename(
            columns={longitude_col : "x",
                 latitude_col : "y",
                 time_col : "time"})
        self._lonlat_df["point"] = self._lonlat_df.apply(lambda x: Point(x["x"], x["y"]), axis=1)
        self._lonlat_df = gpd.GeoDataFrame(self._lonlat_df, geometry="point", crs={"init" : "epsg:4326"})

        self.df = self._lonlat_df.to_crs(crs)

        self.df["x"] = self.df.point.map(lambda x: x.x)
        self.df["y"] = self.df.point.map(lambda x: x.y)

    @property
    def lonlat_bounding_box(self):
        return (
            self._lonlat_df.x.min(),
            self._lonlat_df.x.max(),
            self._lonlat_df.y.min(),
            self._lonlat_df.y.max()
        )

    @property
    def bounding_box(self):
        return (
            self.df.x.min(),
            self.df.x.max(),
            self.df.y.min(),
            self.df.y.max()
        )