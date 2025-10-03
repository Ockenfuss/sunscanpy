"""SkyObject class to compute the position of the sun"""
from datetime import datetime

import numpy as np
import pandas as pd
from skyfield.api import load, wgs84
from skyfield import almanac

from sunscan.utils import logger


class SunObject:
    """Class to compute the position of the sun and its rise/set times.

    This class uses the Skyfield library to calculate the sun's position based on the location of the radar.
    The class provides methods to compute the sun's elevation and azimuth at a given time, as well as the sunrise and 
    sunset times for a given date.

    Args:
        lat (float): Latitude of the radar location in degrees. WGS84 reference system, negative values on southern hemisphere.
        lon (float): Longitude of the radar location in degrees. WGS84 reference system, negative values on western hemisphere.
        altitude (float): Altitude of the radar location in meters.
        refraction_correction (bool): Whether to apply atmospheric refraction correction. Defaults to True.
        humidity (float): Humidity level for refraction correction. Defaults to 0.5.

    Attributes:
        sun (Skyfield object): The sun object.
        location (Skyfield object): The location of the radar.
        ts (Skyfield object): The timescale object.
        refraction (bool): Whether to apply refraction correction.
        humidity (float): Humidity level for refraction correction.

    """

    def __init__(self, lat, lon, altitude, refraction_correction=True, humidity=0.5):
        eph = load('de421.bsp')
        earth = eph['earth']
        self.sun = eph['sun']
        self.location = earth + wgs84.latlon(lat, lon, elevation_m=altitude)
        self.ts = load.timescale()
        self.refraction = refraction_correction
        self.humidity = humidity
        logger.info(f'Initialized SunObject with location lat: {lat}, lon: {lon}, altitude: {altitude}, refraction_correction: {refraction_correction}, humidity: {humidity}')

    def convert_times(self, t):
        """Convert various time formats to Skyfield times."""
        if isinstance(t, str) and t == 'now':
            times = [self.ts.now()]
        elif isinstance(t, datetime):
            times = [self.ts.utc(t.year, t.month, t.day, t.hour, t.minute, t.second)]
        elif isinstance(t, np.datetime64):
            # Handle single numpy datetime64
            timestamp = pd.to_datetime(t)
            times = [self.ts.utc(timestamp.year, timestamp.month, timestamp.day,
                                 timestamp.hour, timestamp.minute, timestamp.second)]
        elif (isinstance(t, np.ndarray) and np.issubdtype(t.dtype, np.datetime64)) or isinstance(t, list):
            # Handle numpy array of datetime64 or list of datetime objects
            time_series = pd.to_datetime(t)
            times = [self.ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) 
                    for dt in time_series.to_pydatetime()]
        else:
            raise ValueError('t must be a datetime object, list of datetime objects, numpy array, or "now"')
        return times

    def compute_sun_location(self, t='now'):
        """Compute the position of the sun at a given time.

        Args:
            t (datetime or list of datetime or str or numpy.ndarray): The UTC time(s) to compute the sun position for.
                If 'now', computes for the current time.
                If datetime, computes for that specific time.
                If list of datetime, computes for each time in the list.
                If numpy.ndarray, converts datetime64 values to datetime objects.

        Raises:
            ValueError: If t is not a datetime object, list of datetime objects, numpy array, or 'now'.

        Returns:
            tuple: The elevation and azimuth of the sun at the specified time(s) in degrees.

        """
        times= self.convert_times(t)

        sun_elvs, sun_azis = [], []
        for time in times:
            astrometric = self.location.at(time).observe(self.sun)
            sun_elv, sun_azi, _ = astrometric.apparent().altaz()
            sun_elvs.append(sun_elv.degrees)
            sun_azis.append(sun_azi.degrees)

        if self.refraction:
            sun_elvs = self.add_refraction(sun_elvs)

        if isinstance(t, (datetime, np.datetime64)) or (isinstance(t, str) and t == 'now'):
            return sun_azis[0], sun_elvs[0]

        return np.array(sun_azis), np.array(sun_elvs)

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
    
    def get_sun_diameter(self, t='now'):
        """Calculate the apparent diameter of the sun at a given time.

        Returns:
            float or list: The apparent diameter of the sun in degrees at the specified time(s).
        """
        sun_radius_km=695660 # solar radius according to https://en.wikipedia.org/wiki/Solar_radius (Haberreiter, Schmutz & Kosovichev (2008))
        times= self.convert_times(t)

        angles=[]
        for time in times:
            astrometric = self.location.at(time).observe(self.sun)
            ra, dec, distance = astrometric.apparent().radec()
            apparent_diameter = np.rad2deg(np.arcsin(sun_radius_km / distance.km) * 2.0)
            angles.append(apparent_diameter)

        if len(angles) == 1:
            angles = angles[0]
        return angles
    
    def __repr__(self):
        return f'SkyObject(location: '+str(self.location)+f'\nrefraction: {self.refraction}, humidity: {self.humidity})'
