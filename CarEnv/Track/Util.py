import numpy as np
from shapely.geometry import LinearRing


def track_from_tum_file(path):
    data = np.loadtxt(path, comments='#', delimiter=',')
    track = data[:, :2]
    bounds = np.stack([-data[:, 3], data[:, 2]], axis=-1)
    return track, bounds


def discretize2(track: LinearRing, step=10.):
    dist = 0

    result = []

    while dist < track.length:
        result.append(np.asarray(track.interpolate(dist).coords[0]))
        dist += step

    result = np.stack(result)
    return result


def shapely_safe_buffer(geom, *args, **kwargs):
    """
    Workaround for shapely rarely generating incorrect spurious polygons together with the desired result when buffering
    a line
    :param geom: Geometry to buffer
    :param args: args forwarded to .buffer()
    :param kwargs: kwargs forwarded to .buffer()
    :return: The main buffered geometry
    """
    import sys
    from shapely.geometry import MultiPolygon

    buffered = geom.buffer(*args, **kwargs)

    # Work-around for a bug with shapely sometimes returning multiple geometries when buffering a LineString/LinearRing
    if isinstance(buffered, MultiPolygon):
        max_area = max([g.area for g in buffered.geoms])
        filtered = [g for g in buffered.geoms if g.area >= .001 * max_area]

        if len(filtered) == 1:
            print("shapely returned spurious geometries on buffering which were dropped", file=sys.stderr)
            return filtered[0]
        else:
            raise RuntimeError("Multiple non-negligible geometries")
    else:
        return buffered
