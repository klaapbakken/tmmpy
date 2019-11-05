from itertools import product
from shapely.geometry import MultiLineString, LineString
from shapely.ops import linemerge
from scipy.spatial.distance import directed_hausdorff
import numpy as np

def to_simple_sequence(sequence):
    return list(map(lambda x: tuple(sorted(x[0])), sequence))

def evaluate_position_accuracy(estimate, truth):
    correct = []
    for e, t in zip(estimate, truth):
        correct.append(e == t)
    return sum(correct)/len(correct)

def remove_repeated_segments(sequence):
    mod_sequence = []
    mod_sequence.append(sequence[0])
    for segment in sequence:
        if segment != mod_sequence[-1]:
            mod_sequence.append(segment)
    return mod_sequence

def connectedness(sequence):
    for previous, current in zip(sequence[:-1], sequence[1:]):
        if len(set(previous).intersection(set(current))) != 1:
            print(previous, current)
            return False
    return True

def curve_to_linestring(curve, linestring_lookup):
    linestrings = list(map(lambda x: linestring_lookup[x], curve))
    mls = MultiLineString(linestrings)
    return linemerge(mls)

def curve_to_coords(curve, linestring_lookup):
    ls = curve_to_linestring(curve, linestring_lookup)
    if type(ls) is MultiLineString:
        coords = np.concatenate(list(map(lambda x: np.array(list(x.coords)), ls)))
    elif type(ls) is LineString:
        coords = np.concatenate(list(map(lambda x: np.array(list(x.coords)), [ls])))
    else:
        raise ValueError
    return coords

def hausdorff_distance(estimate, truth, linestring_lookup):
    l_coords = curve_to_coords(estimate, linestring_lookup)
    r_coords = curve_to_coords(truth, linestring_lookup)
    
    return max(directed_hausdorff(l_coords, r_coords), directed_hausdorff(r_coords, l_coords), key=lambda x: x[0])[0]