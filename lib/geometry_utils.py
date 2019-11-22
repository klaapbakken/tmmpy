import fiona
import geopandas as gpd
from shapely.geometry import Polygon


def bounding_box_to_polygon(bounding_box):
    xmin, xmax, ymin, ymax = bounding_box
    return Polygon([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])

def transform_geometry(geometry, in_epsg, out_epsg):
    transformed = gpd.GeoSeries([geometry], crs=fiona.crs.from_epsg(in_epsg)).to_crs(
        fiona.crs.from_epsg(out_epsg)
    )
    return transformed[0]

def polygon_to_bounding_box(polygon):
    bounding_box = polygon.bounds
    return (bounding_box[0], bounding_box[2], bounding_box[1], bounding_box[3])