"""SkyObject class to compute the position of the sun"""
from datetime import datetime

import numpy as np
from skyfield.api import load, N, S, E, W, wgs84
from skyfield import almanac

from pymira import logger


class SkyObject:
    """Class to compute the position of the sun and its rise/set times.

    This class uses the Skyfield library to calculate the sun's position based on the location of the radar.
    The class provides methods to compute the sun's elevation and azimuth at a given time, as well as the sunrise and 
    sunset times for a given date.

    Args:
        lat (float): Latitude of the radar location in degrees.
        lon (float): Longitude of the radar location in degrees.
        elevation (float): Elevation of the radar location in meters.
        refraction_correction (bool): Whether to apply atmospheric refraction correction. Defaults to True.
        humidity (float): Humidity level for refraction correction. Defaults to 0.5.

    Attributes:
        sun (Skyfield object): The sun object.
        location (Skyfield object): The location of the radar.
        ts (Skyfield object): The timescale object.
        refraction (bool): Whether to apply refraction correction.
        humidity (float): Humidity level for refraction correction.

    """

    def __init__(self, lat, lon, elevation, refraction_correction=True, humidity=0.5):
        eph = load('de421.bsp')
        earth = eph['earth']
        self.sun = eph['sun']
        self.location = self.get_radar_location(lat, lon, elevation, earth)
        self.ts = load.timescale()
        self.refraction = refraction_correction
        self.humidity = humidity

    def get_radar_location(self, lat, lon, elevation, earth):
        """Get the radar location as a Skyfield object.

        Args:
            lat (float): Latitude of the radar location in degrees.
            lon (float): Longitude of the radar location in degrees.
            elevation (float): Elevation of the radar location in meters.
            earth (Skyfield object): The Earth object from Skyfield.

        Returns:
            Skyfield object: The location of the radar as a Skyfield object.

        """
        if lat > 0:
            lat = lat * N
        else:
            lat = lat * S
        if lon > 0:
            lon = lon * E
        else:
            lon = lon * W
        return earth + wgs84.latlon(lat, lon, elevation_m=elevation)

    def compute_sun_location(self, t='now'):
        """Compute the position of the sun at a given time.

        Args:
            t (datetime or list of datetime or str): The time(s) to compute the sun position for.
                If 'now', computes for the current time.
                If datetime, computes for that specific time.
                If list of datetime, computes for each time in the list.

        Raises:
            ValueError: If t is not a datetime object, list of datetime objects, or 'now'.

        Returns:
            tuple: The elevation and azimuth of the sun at the specified time(s).

        """
        if isinstance(t, str) and t == 'now':
            times = [self.ts.now()]
        elif isinstance(t, datetime):
            times = [self.ts.utc(t.year, t.month, t.day, t.hour, t.minute, t.second)]
        elif isinstance(t, list):
            times = [self.ts.utc(time.year, time.month, time.day, time.hour, time.minute, time.second) for time in t]
        else:
            raise ValueError('t must be a datetime object or a list of datetime objects or "now"')

        sun_elvs, sun_azis = [], []
        for time in times:
            astrometric = self.location.at(time).observe(self.sun)
            sun_elv, sun_azi, _ = astrometric.apparent().altaz()
            sun_elvs.append(sun_elv.degrees)
            sun_azis.append(sun_azi.degrees)

        if self.refraction:
            sun_elvs = self.add_refraction(sun_elvs)

        if isinstance(t, datetime) or t == 'now':
            return sun_elvs[0], sun_azis[0]

        return sun_elvs, sun_azis

    def add_refraction(self, elevation):
        """Correct the true elevation angle with atmospheric refraction.

        The formulas used for the refraction correction are based on the publication of 
        Huuskonen and Holleman (2007): https://doi.org/10.1175/JTECH1978.1 

        The refraction correction is based on the formula:
            refraction = alpha / tan(elevation + beta / (elevation + gamma))

        where alpha, beta, and gamma are constants that depend on the atmospheric humidity:
            alpha = 0.0155 + 0.0054 * humidity
            beta = 8
            gamma = 4.23

        The formula is applied to the true elevation angle and returns the corrected elevation angle, after
        atmospheric refractivity was taken into account.

        The 'apparent' elevation angle is then:
            apparent_elevation = true_elevation + refraction

        Args:
            elevation (float or list of float): The true elevation angle(s) in degrees.

        Returns:
            numpy.array: The refractivity corrected "apparent" elevation angle(s) in degrees.

        """
        logger.info('Applying refraction correction to sun elevation angles with humidity: %s', self.humidity)
        elevation = np.asarray(elevation, dtype=float)
        alpha = 0.0155 + 0.0054*self.humidity
        beta = 8
        gamma = 4.23
        refraction = alpha/np.tan(np.deg2rad(elevation + beta / (elevation + gamma)))
        el_apparent = elevation + refraction
        return el_apparent

    def get_sunrise(self, date: datetime):
        """Get the sunrise time for a given date.

        Args:
            date (datetime): The date to get the sunrise time for.

        Returns:
            datetime: The sunrise time for the given date.

        """
        start = self.ts.utc(date.replace(hour=0, minute=0, second=0))
        end = self.ts.utc(date.replace(hour=23, minute=59, second=59))
        times, _ = almanac.find_risings(self.location, self.sun, start, end)
        return times[0].utc_datetime()

    def get_sunset(self, date: datetime):
        """Get the sunset time for a given date.

        Args:
            date (datetime): The date to get the sunset time for.

        Returns:
            datetime: The sunset time for the given date.

        """
        start = self.ts.utc(date.replace(hour=0, minute=0, second=0))
        end = self.ts.utc(date.replace(hour=23, minute=59, second=59))
        times, _ = almanac.find_settings(self.location, self.sun, start, end)
        return times[0].utc_datetime()
