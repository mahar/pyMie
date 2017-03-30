# util.py 
import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
from scipy.signal import argrelextrema
def local_maxima(array, min_distance = 1, periodic=False, edges_allowed=True):
    """Find all local maxima of the array, separated by at least min_distance."""
    array = np.asarray(array)
    cval = 0
    if periodic:
        mode = 'wrap'
    elif edges_allowed:
        mode = 'nearest'
    else:
        mode = 'constant'
    cval = array.max()+1
    max_points = array == ndimage.maximum_filter(array, 1+2*min_distance, mode=mode, cval=cval)
    return [indices[max_points] for indices in np.indices(array.shape)]