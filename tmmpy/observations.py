from data_interface import PostGISQuery
from street_network import StreetNetwork
from state_space import SegmentStateSpace

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import fiona


class GPSObservations:
    """A class for representing GPS-measurements. 
    
    Parameters:
    ---
    observations_df -- Dataframe containing at x- and y-coordinates and time \n
    longitude_col -- Name of column containing longitude \n
    latitude_col -- Name of column containing latitude \n
    time_col -- Name of column containing time of measurement \n
    crs -- The desired output coordinate reference system
    """

    def __init__(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        time_col: str,
        input_epsg: int,
        output_epsg: int,
    ):
        """See class documentation."""
        INPUT_CRS = fiona.crs.from_epsg(input_epsg)
        OUTPUT_CRS = fiona.crs.from_epsg(output_epsg)

        self.input_df = df.rename(columns={x_col: "x", y_col: "y", time_col: "time"})

        self.input_df = (
            self.input_df.assign(
                point=self.input_df.apply(lambda x: Point(x["x"], x["y"]), axis=1)
            )
            .assign(time=pd.to_datetime(self.input_df.time))
            .sort_values(by="time", ascending=True)
            .set_index("time")
        )
        self.input_df = gpd.GeoDataFrame(self.input_df, geometry="point", crs=INPUT_CRS)
        self.output_df = self.input_df.to_crs(crs=OUTPUT_CRS)
        self.update_coordinate_columns()

    def update_coordinate_columns(self):
        self.output_df["x"] = self.output_df.point.map(lambda x: x.x)
        self.output_df["y"] = self.output_df.point.map(lambda x: x.y)

    def transform(self, epsg):
        self.output_df.to_crs(crs=fiona.crs.from_epsg(epsg), inplace=True)
        self.update_coordinate_columns()
        return self

    @property
    def maximum_subsequent_distance(self):
        """Find the maximum distance between subsequent GPS observations."""
        return np.max(
            np.linalg.norm(
                self.output_df[["x", "y"]].iloc[:-1].values
                - self.output_df[["x", "y"]].iloc[1:].values,
                axis=1,
            )
        )

    @property
    def bounding_box(self):
        """Get the bounding box of the dataframe in current CRS."""
        return (
            self.output_df.x.min(),
            self.output_df.x.max(),
            self.output_df.y.min(),
            self.output_df.y.max(),
        )


class TimelessGPSObservations(GPSObservations):
    def __init__(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        input_crs: dict,
        output_crs: dict,
    ):
        """See class documentation."""
        INPUT_CRS = input_crs
        OUTPUT_CRS = output_crs

        self.input_df = df.rename(columns={x_col: "x", y_col: "y"})

        self.input_df = self.input_df.assign(
            point=self.input_df.apply(lambda x: Point(x["x"], x["y"]), axis=1)
        ).sort_index()
        self.input_df = gpd.GeoDataFrame(self.input_df, geometry="point", crs=INPUT_CRS)
        self.output_df = self.input_df.to_crs(crs=OUTPUT_CRS)
        self.update_coordinate_columns()
