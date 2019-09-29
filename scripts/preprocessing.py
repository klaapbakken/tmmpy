import sys
import fiona

import numpy as np
import geopandas as gpd
import pandas as pd
import os

from shapely.geometry import Point

from utils import load_observation_gdf
from utils import tracks_intersecting_densest_region
from utils import validate_track
from utils import track_gdf_preprocessing
from utils import extract_longest_sequence_in_region
from utils import extract_longest_sequence_without_stationarity

from gps_map_matching import GPSMapMatcher
from data_interface import OverpassAPIQuery
from street_network import StreetNetwork
from state_space import StreetStateSpace

from tqdm import tqdm

input_dir_path = sys.argv[1]
print(f"Input path: {os.path.abspath(input_dir_path)}")
output_path = sys.argv[2]
print(f"Output path: {os.path.abspath(output_path)}")
raw_gdf = load_observation_gdf(os.path.abspath(input_dir_path))

track_ids = raw_gdf.track_id.unique()
valid_track_ids = []
print("Validating tracks:")
for track_id in tqdm(track_ids):
    track_gdf = raw_gdf[raw_gdf.track_id == track_id]
    if validate_track(track_gdf):
        valid_track_ids.append(track_id)
valid_gdf = raw_gdf[raw_gdf.track_id.isin(valid_track_ids)]

dense_region_gdf, polygon = tracks_intersecting_densest_region(valid_gdf)

x_min = np.min(np.array(list(polygon.boundary.coords))[:, 0])
x_max = np.max(np.array(list(polygon.boundary.coords))[:, 0])
y_min = np.min(np.array(list(polygon.boundary.coords))[:, 1])
y_max = np.max(np.array(list(polygon.boundary.coords))[:, 1])
tgdf =  gpd.GeoSeries([Point([x_min, y_min]), Point([x_max, y_max])], crs=fiona.crs.from_epsg(32632)).to_crs(epsg=4326)
xmin, xmax = tgdf.map(lambda x: x.x)
ymin, ymax = tgdf.map(lambda x: x.y)

bounding_box = (xmin, xmax, ymin, ymax)
osm_data_source = OverpassAPIQuery(32632, bounding_box)
street = StreetNetwork(osm_data_source)
street_state_space = StreetStateSpace(street, 1)
gps_map_matcher = GPSMapMatcher(street_state_space, "exponential", "projection", sigma=1)

print("Processing selected tracks:")
track_gdfs = []
for track_id in tqdm(dense_region_gdf.track_id.unique()):
    track_gdf = dense_region_gdf[dense_region_gdf.track_id == track_id].copy()
    track_gdf = track_gdf_preprocessing(track_gdf, polygon)
    track_gdf = extract_longest_sequence_in_region(track_gdf)
    track_gdf = extract_longest_sequence_without_stationarity(track_gdf, 50)
    track_gdfs.append(track_gdf)
processed_gdf = gpd.GeoDataFrame(pd.concat(track_gdfs), crs=fiona.crs.from_epsg(32632)).set_geometry("point")

processed_gdf.to_csv(output_path)