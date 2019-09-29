import numpy as np
import os
import pandas as pd


import geopandas as gpd
import fiona
from shapely.geometry import Point, Polygon


def validate_track(gdf, length_requirement=50, quantiles=[0.1, 0.9], slack=100):
    long_enough = gdf.shape[0] > length_requirement
    def within_bounds(column_name):
        lower = gdf[column_name].quantile(quantiles[0])
        median = gdf[column_name].quantile(0.5)
        upper = gdf[column_name].quantile(quantiles[1])
        return (lower + slack) < median < (upper - slack)
    return all(map(within_bounds, ["x", "y"])) and long_enough

def tracks_intersecting_densest_region(gdf, region_width=1000):
    x_range = [gdf.x.min(), gdf.x.max()]
    y_range = [gdf.y.min(), gdf.y.max()]
    x_span = np.max(np.diff(x_range))
    y_span = np.max(np.diff(y_range))
    x_bins = x_span // region_width
    y_bins = y_span // region_width
    counts, x_grid, y_grid = np.histogram2d(
        x = gdf.x.to_numpy(),
        y = gdf.y.to_numpy(),
        bins=[x_bins, y_bins],
        range=[x_range, y_range]
        )
    x_i, y_i = np.unravel_index(np.argmax(counts), shape=counts.shape)
    region_xs = x_grid[x_i], x_grid[x_i + 1]
    region_ys = y_grid[y_i], y_grid[y_i + 1]
    track_ids = gdf[
        (gdf.x > region_xs[0]) &
        (gdf.x < region_xs[1]) &
        (gdf.y > region_ys[0]) &
        (gdf.y < region_ys[1])
    ].track_id.unique()

    x_min, x_max = region_xs
    y_min, y_max = region_ys

    polygon = Polygon([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])
    
    return gdf[gdf.track_id.isin(track_ids)], polygon

def load_observation_gdf(data_path):

    data = list(map(pd.read_csv, map(lambda x: os.path.join(data_path, x), os.listdir(data_path))))

    df = pd.DataFrame(
        {"point" : data[1].apply(lambda x: Point([x["longitude"], x["latitude"]]), axis=1),
        "track_id" : data[1]["track_id"],
        "time" : data[1]["point_time"]
        }
    )

    gdf = gpd.GeoDataFrame(df, crs=fiona.crs.from_epsg(4326)).set_geometry("point").to_crs(epsg=32632)
    gdf["x"] = gdf.point.map(lambda x: x.x)
    gdf["y"] = gdf.point.map(lambda x: x.y)

    return gdf

def track_gdf_preprocessing(track_gdf, region_polygon, seconds=60):
    track_gdf.loc[:, "time"] = track_gdf.time.map(lambda x: pd.to_datetime(x))
    track_gdf = track_gdf.sort_values("time")
    track_gdf = track_gdf.set_index("time")
    
    coord_cols = ["x", "y"]
    rolling_track_gdf = track_gdf[coord_cols].rolling(f"{seconds}s")

    for col in coord_cols:
        track_gdf[f"rolling_median_{col}"] = rolling_track_gdf.median()[col]
        track_gdf[f"rolling_min_{col}"] = rolling_track_gdf.min()[col]
        track_gdf[f"rolling_max_{col}"] = rolling_track_gdf.max()[col]
    
    track_gdf["in_region"] = track_gdf.point.map(lambda x: x.intersects(region_polygon))

    return track_gdf

def extract_longest_sequence_without_stationarity(track_gdf, required_movement):
    track_gdf["sufficient_movement_x"] = ((track_gdf.rolling_min_x + required_movement) < track_gdf.rolling_median_x) & (track_gdf.rolling_median_x < (track_gdf.rolling_max_x - required_movement))
    track_gdf["sufficient_movement_y"] = ((track_gdf.rolling_min_y + required_movement) < track_gdf.rolling_median_y) & (track_gdf.rolling_median_y < (track_gdf.rolling_max_y - required_movement))
    track_gdf["sufficient_movement"] = track_gdf.sufficient_movement_x | track_gdf.sufficient_movement_y

    longest_run, end_index = longest_run_of_value(track_gdf.sufficient_movement.values, True)
    return track_gdf.iloc[(end_index - longest_run):end_index]

def extract_longest_sequence_in_region(track_gdf):
    longest_run, end_index = longest_run_of_value(track_gdf.in_region.values, True)
    return track_gdf.iloc[(end_index - longest_run):end_index]

def longest_run_of_value(array, value):
    if len(array) == 0:
        return 0, 0 
    bool_array = array == value
    end_indices = []
    run_lengths = []
    run_length = 0
    for index, element in enumerate(bool_array):
        if element == True and index < len(bool_array) - 1:
            run_length += 1
        elif (element == False) or (element == True and index == len(bool_array) - 1):
            run_lengths.append(run_length)
            run_length = 0
            end_indices.append(index)

    longest_run = np.max(np.array(run_lengths))
    longest_run_end_index = end_indices[np.argmax(run_lengths)]
    
    return longest_run, longest_run_end_index  