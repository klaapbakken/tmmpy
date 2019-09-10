from data_interface import PostGISQuery
from street_network import StreetNetwork
from state_space import StreetStateSpace

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import fiona

class GPSObservations:
    """A class for representing GPS-measurements. 
    
    Parameters:
    ---
    observations_df -- Dataframe containing at least x- and y-coordinates \n
    longitude_col -- Name of column containing longitude \n
    latitude_col -- Name of column containing latitude \n
    time_col -- Name of column containing time of measurement \n
    crs -- The desired output coordinate reference system
    """

    def __init__(
        self,
        observations_df: gpd.GeoDataFrame,
        longitude_col: str,
        latitude_col: str,
        time_col: str,
        epsg: int,
    ):
        """See class documentation."""
        self.LONLAT_CRS = fiona.crs.from_epsg(4326)
        self.epsg = epsg
        self._lonlat_df = observations_df.rename(
            columns={longitude_col: "x", latitude_col: "y", time_col: "time"}
        )
        self._lonlat_df["time"] = pd.to_datetime(self._lonlat_df.time)
        self._lonlat_df["point"] = self._lonlat_df.apply(
            lambda x: Point(x["x"], x["y"]), axis=1
        )
        self._lonlat_df = gpd.GeoDataFrame(
            self._lonlat_df, geometry="point", crs=self.LONLAT_CRS
        )
        self._lonlat_df.sort_values(by="time", ascending=True)

        self._utm33_df = self._lonlat_df.to_crs(epsg=32633)
        self._utm33_df["x"] = self._utm33_df.point.map(lambda x: x.x)
        self._utm33_df["y"] = self._utm33_df.point.map(lambda x: x.y)

        self.df = self._lonlat_df.to_crs(epsg=self.epsg)
        self.df["x"] = self.df.point.map(lambda x: x.x)
        self.df["y"] = self.df.point.map(lambda x: x.y)

    @property
    def maximum_subsequent_distance(self):
        """Find the maximum distance between subsequent GPS observations."""
        return np.max(
            np.linalg.norm(
                self.df[["x", "y"]].iloc[:-1].values
                - self.df[["x", "y"]].iloc[1:].values,
                axis=1,
            )
        )

    @property
    def lonlat_bounding_box(self):
        """Get the bounding box in original, longitude-latitude CRS."""
        return (
            self._lonlat_df.x.min(),
            self._lonlat_df.x.max(),
            self._lonlat_df.y.min(),
            self._lonlat_df.y.max(),
        )

    @property
    def utm33_bounding_box(self):
                return (
            self._utm33_df.x.min(),
            self._utm33_df.x.max(),
            self._utm33_df.y.min(),
            self._utm33_df.y.max(),
        )

    @property
    def bounding_box(self):
        """Get the bounding box of the dataframe in current CRS (specified during construction)."""
        return (self.df.x.min(), self.df.x.max(), self.df.y.min(), self.df.y.max())
