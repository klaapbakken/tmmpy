import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import sys

from observations import GPSObservations
from map_matching import GPSMapMatcher

from math import ceil

path = sys.argv[1]

df = pd.read_csv(os.path.abspath(path))

track_ids = df.track_id.unique()

no_of_cols = 9
no_of_rows = ceil(len(track_ids)/no_of_cols)

def position_generator(no_of_rows, no_of_cols):
    for i in range(no_of_cols):
        for j in range(no_of_rows):
            yield j, i

fig, ax = plt.subplots(no_of_rows, no_of_cols, figsize=(no_of_cols, no_of_rows))
UTM32 = "epsg:32632"
pos = position_generator(no_of_rows, no_of_cols)

for track_id in track_ids:
    row, col = next(pos)
    gps_observations = GPSObservations(df[df.track_id == track_id], "longitude", "latitude", "point_time", UTM32)
    ax[row, col].set_aspect("equal")
    ax[row, col].axis("off")
    gps_observations.df.plot(ax=ax[row, col])
fig.savefig("img.png")