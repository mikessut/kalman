
import numpy as np


def latlong2dist(lat1, long1, lat2, long2):
    """
    https://en.wikipedia.org/wiki/Haversine_formula#:~:text=The%20haversine%20formula%20determines%20the,given%20their%20longitudes%20and%20latitudes.&text=The%20term%20haversine%20was%20coined,sin2(%CE%B82).

    :param lat1:
    :param long1:
    :param lat2:
    :param long2:
    :return:
    """
    R = 6357000.0
    dLat = (lat2 - lat1)*np.pi/180
    dLong = (long2 - long1)*np.pi/180
    a = np.sin(dLat/2) * np.sin(dLat/2) \
        + np.cos(lat1*np.pi/180) * np.cos(lat2*np.pi/180) * \
          np.sin(dLong/2) * np.sin(dLong/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c  # Distance in m
    return d


def latlong2bearing(lat1, long1, lat2, long2):
    """
    # https://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/
    # θ = atan2(sin(Δlong)*cos(lat2), cos(lat1)*sin(lat2) − sin(lat1)*cos(lat2)*cos(Δlong))

    Definition of direction is the traditional North 0 deg, east 90 deg, etc.
    """

    dLong = (long2 - long1)*np.pi/180
    return np.arctan2(np.sin(dLong)*np.cos(lat2*np.pi/180),
                      np.cos(lat1*np.pi/180)*np.sin(lat2*np.pi/180) - np.sin(lat1*np.pi/180)*np.cos(lat2*np.pi/180)*np.cos(dLong))

