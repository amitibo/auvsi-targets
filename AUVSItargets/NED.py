#!/usr/bin/env python
"""
Created on Mon Nov  5 17:24:01 2012

@author: narcis
"""
import math
import numpy as np

NORTH = 0
EAST = 1
DEPTH = 2

class NED:
    def __init__(self, lat, lon, height):
        """ Create a NED plane centered at lat, lon, altitude """
        # Constants defined by the World Geodetic System 1984 (WGS84)
        self.a = 6378137
        self.b = 6356752.3142
        self.esq = 6.69437999014 * 0.001
        self.e1sq = 6.73949674228 * 0.001
        self.f = 1 / 298.257223563

        # Save NED origin
        self.init_lat = math.radians(lat)
        self.init_lon = math.radians(lon)
        self.init_h = height
        self.init_ecef = self.geodetic2ecef([lat, lon, height])
        phiP = math.atan2(self.init_ecef[2],
                          math.sqrt(self.init_ecef[0] ** 2 +
                                    self.init_ecef[1] ** 2))
        self.ecef_to_ned_matrix = __nRe__(phiP, self.init_lon)
        self.ned_to_ecef_matrix = __nRe__(self.init_lat, self.init_lon).T


    def geodetic2ecef(self, coord):
        """Convert geodetic coordinates to ECEF."""
        #coord = [lat(degrees), lon(degress), h(meters)]
        # http://code.google.com/p/pysatel/source/browse/trunk/coord.py?r=22
        lat = math.radians(coord[0])
        lon = math.radians(coord[1])

        xi = math.sqrt(1 - self.esq * math.sin(lat)**2)
        x = (self.a / xi + coord[2]) * math.cos(lat) * math.cos(lon)
        y = (self.a / xi + coord[2]) * math.cos(lat) * math.sin(lon)
        z = (self.a / xi * (1 - self.esq) + coord[2]) * math.sin(lat)
        return np.array([x, y, z])


    def ecef2geodetic(self, ecef):
        """Convert ECEF coordinates to geodetic.
        J. Zhu, "Conversion of Earth-centered Earth-fixed coordinates \
        to geodetic coordinates," IEEE Transactions on Aerospace and \
        Electronic Systems, vol. 30, pp. 957-961, 1994."""
        x = ecef[0]
        y = ecef[1]
        z = ecef[2]

        r = math.sqrt(x * x + y * y)
        Esq = self.a * self.a - self.b * self.b
        F = 54 * self.b * self.b * z * z
        G = r * r + (1 - self.esq) * z * z - self.esq * Esq
        C = (self.esq * self.esq * F * r * r) / (pow(G, 3))
        S = __cbrt__(1 + C + math.sqrt(C * C + 2 * C))
        P = F / (3 * pow((S + 1 / S + 1), 2) * G * G)
        Q = math.sqrt(1 + 2 * self.esq * self.esq * P)
        r_0 =  -(P * self.esq * r) / (1 + Q) + math.sqrt(0.5 * self.a * self.a*(1 + 1.0 / Q) - \
            P * (1 - self.esq) * z * z / (Q * (1 + Q)) - 0.5 * P * r * r)
        U = math.sqrt(pow((r - self.esq * r_0), 2) + z * z)
        V = math.sqrt(pow((r - self.esq * r_0), 2) + (1 - self.esq) * z * z)
        Z_0 = self.b * self.b * z / (self.a * V)
        h = U * (1 - self.b * self.b / (self.a * V))
        lat = math.atan((z + self.e1sq * Z_0) / r)
        lon = math.atan2(y, x)
        return math.degrees(lat), math.degrees(lon), h


    def ecef2ned(self, ecef):
        """Converts ECEF coordinate pos into local-tangent-plane ENU
        coordinates relative to another ECEF coordinate ref. Returns a tuple
        (East, North, Up).
        """
        p = ecef - self.init_ecef
        ned = np.dot(self.ecef_to_ned_matrix, p)
        ned[2] = -ned[2]
        return ned


    def ned2ecef(self, ned):
        """NED (north/east/down) to ECEF coordinate system conversion."""
        ned = np.array([ned[0], ned[1], -ned[2]])
        res = np.dot(self.ned_to_ecef_matrix, ned) + self.init_ecef
        return res


    def geodetic2ned(self, coord):
        """ Geodetic position to a local NED system """
        ecef = self.geodetic2ecef(coord)
        return self.ecef2ned(ecef)


    def ned2geodetic(self, ned):
        """ Local NED position to geodetic """
        ecef = self.ned2ecef(ned)
        return self.ecef2geodetic(ecef)


def degree2DegreeMinute(lat, lon):
    lat_degree = __degree2DegreeMinuteAux__(lat)
    lon_degree = __degree2DegreeMinuteAux__(lon)
    return lat_degree, lon_degree


def degreeMinute2Degree(lat, lon):
        """ Transforms latitude and longitude in the format
            DDDMM.MM to the format DDD.DD """
        lat_deg, lat_min = __splitDegreeMinutes__(lat)
        lon_deg, lon_min = __splitDegreeMinutes__(lon)
        return lat_deg + lat_min/60.0, lon_deg + lon_min/60.0


def __cbrt__(x):
    if x >= 0:
        return pow(x, 1.0/3.0)
    else:
        return -pow(abs(x), 1.0/3.0)


def __nRe__(lat, lon):
    sinLat = math.sin(lat)
    sinLon = math.sin(lon)
    cosLat = math.cos(lat)
    cosLon = math.cos(lon)

    mx = np.array([-sinLat*cosLon,   -sinLat*sinLon,    cosLat,
                   -sinLon,           cosLon,           .0,
                   cosLat*cosLon,   cosLat*sinLon,      sinLat]).reshape(3,3)
    return mx


def __degree2DegreeMinuteAux__(value):
    val = str(value).split('.')
    minute = float('0.' + val[1]) * 60.0
    if minute < 10.0:
        return float(val[0] + '0' + str(minute))
    else:
        return float(val[0] + str(minute))



def __splitDegreeMinutes__(value):
    """ Transform DDDMM.MM to DDD, MM.MM """
    val = str(value).split('.')
    val_min = val[0][-2] + val[0][-1] + '.' + val[1]
    val_deg = ''
    for i in range(len(val[0])-2):
        val_deg = val_deg + val[0][i]

    return int(val_deg), float(val_min)


if __name__ == '__main__':
    """ Test example """
    lat0 = 4303.4739        #DDDMM.MMMM
    lon0 = 600.5378         #DDDMM.MMMM
    lat1 = 43.05760         #DDD.DDDDD
    lon1 = 6.007583         #DDD.DDDDD
    lat2 = 43.057417        #DDD.DDDDD
    lon2 = 6.007667         #DDD.DDDDD
    lat3 = 39.997417        #DDD.DDDDD
    lon3 = 6.007667         #DDD.DDDDD
    lat4 = 40.017417        #DDD.DDDDD
    lon4 = 6.00667         #DDD.DDDDD
    lat5 = 39.9717417        #DDD.DDDDD
    lon5 = 6.01667         #DDD.DDDDD

    # lat_0, lon_0 = degreeMinute2Degree(lat0, lon0)
    # ned = NED(lat0, lon0, 0.0)
    # print ned.llh2Eecef([lat1, lon1, 0.0])

    ned = NED(lat3, lon3, 0.0)
    ned_1 = ned.geodetic2ned([lat4, lon4, 0.0])
    print ned_1
    ned_2 = ned.geodetic2ned([lat5, lon5, 0.0])
    print ned_2
    lat, lon, h = ned.ned2geodetic(ned_2)
    print lat, lon, h

    print np.sqrt((ned_1[0] - ned_2[0])**2 + (ned_1[1] - ned_2[1])**2)
