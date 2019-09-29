from observations import GPSObservations
from data_interface import OverpassAPIQuery
from street_network import StreetNetwork
from state_space import StreetStateSpace
from gps_map_matching import GPSMapMatcher

import pandas as pd
import geopandas as gpd

import sys
import fiona

from shapely.wkt import loads
from tqdm import tqdm

import time

start_time = time.time()

path = sys.argv[1]
df = pd.read_csv(path)
df["point"] = pd.Series(list(map(loads, df.point.values)))
gdf = gpd.GeoDataFrame(df, geometry="point", crs=fiona.crs.from_epsg(32632))

lonlat_gdf = gdf.to_crs(crs=fiona.crs.from_epsg(4326))
lonlat_gdf["x"] = lonlat_gdf.point.map(lambda x: x.x)
lonlat_gdf["y"] = lonlat_gdf.point.map(lambda x: x.y)

bounding_box = (
    lonlat_gdf.x.min(),
    lonlat_gdf.x.max(),
    lonlat_gdf.y.min(),
    lonlat_gdf.y.max()
)

osm_data_source = OverpassAPIQuery(32632, bounding_box)

track_ids = df.track_id.unique()

observation_objects = []
print("Creating GPSObservations objects:")
for track_id in tqdm(track_ids):
    track_gdf = gdf[gdf.track_id == track_id]
    gps_observation = GPSObservations(track_gdf, "x", "y", "time", 32632, 32632)
    observation_objects.append(gps_observation)

street = StreetNetwork(osm_data_source)
street_state_space = StreetStateSpace(street, 1)
gps_map_matcher = GPSMapMatcher(street_state_space, "exponential", "projection", sigma=4)

print("Attaching observations:")
for gps_observation in tqdm(observation_objects):
    gps_map_matcher.attach_observations(gps_observation)
    gps_map_matcher.add_observations()

baum_welch_start_time = time.time()

gps_map_matcher.hidden_markov_model.baum_welch(gps_map_matcher.zs)

end_time = time.time()

print(f"Time elapsed during Baum-Welch: {end_time - baum_welch_start_time}")
print(f"Total time elapsed: {end_time - start_time}")
