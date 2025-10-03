import numpy as np
import pytest
from datetime import datetime, timezone
from sunscan.sun import SunObject

class TestSunObject:
    
    def test_init_basic(self):
        """Test basic initialization of SunObject."""
        sun_obj = SunObject(lat=52.5, lon=13.4, altitude=100)
        assert sun_obj.refraction is True
        assert sun_obj.humidity == 0.5
        
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        sun_obj = SunObject(lat=40.7, lon=-74.0, altitude=50, 
                           refraction_correction=False, humidity=0.8)
        assert sun_obj.refraction is False
        assert sun_obj.humidity == 0.8
    
    def test_northern_hemisphere_sun_position(self):
        """Test that sun is in the south at solar noon in northern hemisphere."""
        # Berlin, Germany
        sun_obj = SunObject(lat=52.5, lon=13.4, altitude=100)
        
        # Summer day - June 21, 2023 at approximate solar noon (12:00 UTC + longitude correction)
        test_time = datetime(2023, 6, 21, 11, 6, 0)  # Approximate solar noon for Berlin
        azimuth, elevation = sun_obj.compute_sun_location(test_time)
        
        # Sun should be roughly in the south (azimuth around 180°) at solar noon
        assert 160 < azimuth < 200, f"Expected azimuth around 180°, got {azimuth}°"
        assert elevation > 0, f"Sun should be above horizon, got {elevation}°"
    
    def test_southern_hemisphere_sun_position(self):
        """Test that sun is in the north at solar noon in southern hemisphere."""
        # Sydney, Australia
        sun_obj = SunObject(lat=-33.9, lon=151.2, altitude=50)
        
        # Winter day in southern hemisphere - June 21, 2023
        test_time = datetime(2023, 6, 21, 2, 5, 0)  # Approximate solar noon for Sydney
        azimuth, elevation = sun_obj.compute_sun_location(test_time)
        
        # Sun should be roughly in the north (azimuth around 0° or 360°) at solar noon
        assert azimuth < 30 or azimuth > 330, f"Expected azimuth around 0°/360°, got {azimuth}°"
        assert elevation > 0, f"Sun should be above horizon, got {elevation}°"
    
    def test_summer_solstice_tropic_of_cancer(self):
        """Test that sun is directly overhead at noon on June 21 at Tropic of Cancer."""
        # Location at Tropic of Cancer (23.5°N)
        sun_obj = SunObject(lat=23.5, lon=0.0, altitude=0)
        
        # June 21, 2023 at solar noon
        test_time = datetime(2023, 6, 21, 12, 0, 0)
        azimuth, elevation = sun_obj.compute_sun_location(test_time)
        
        # Sun should be very close to directly overhead (elevation near 90°)
        assert elevation > 85, f"Expected elevation near 90°, got {elevation}°"
    
    def test_winter_solstice_tropic_of_capricorn(self):
        """Test sun position at Tropic of Capricorn during winter solstice."""
        # Location at Tropic of Capricorn (23.5°S)
        sun_obj = SunObject(lat=-23.5, lon=0.0, altitude=0)
        
        # December 21, 2023 at solar noon
        test_time = datetime(2023, 12, 21, 12, 0, 0)
        azimuth, elevation = sun_obj.compute_sun_location(test_time)
        
        # Sun should be very close to directly overhead
        assert elevation > 85, f"Expected elevation near 90°, got {elevation}°"
    
    def test_longitude_effects(self):
        """Test that longitude affects azimuth position."""
        # Same latitude, different longitudes
        sun_obj_east = SunObject(lat=50.0, lon=120.0, altitude=0)  # Eastern location
        sun_obj_west = SunObject(lat=50.0, lon=-120.0, altitude=0)  # Western location
        
        # Same UTC time
        test_time = datetime(2023, 6, 21, 12, 0, 0)
        
        azi_east, elv_east = sun_obj_east.compute_sun_location(test_time)
        azi_west, elv_west = sun_obj_west.compute_sun_location(test_time)
        
        # At the same UTC time, the sun should be at different positions
        assert abs(azi_east - azi_west) > 10, "Azimuth should differ significantly between distant longitudes"
    
    def test_equator_equinox(self):
        """Test sun position at equator during equinox."""
        sun_obj = SunObject(lat=0.0, lon=0.0, altitude=0)
        
        # March 20, 2023 (approximate spring equinox) at solar noon
        test_time = datetime(2023, 3, 20, 12, 0, 0)
        azimuth, elevation = sun_obj.compute_sun_location(test_time)
        
        # At equinox, sun should be very high at equator
        assert elevation > 85, f"Expected elevation near 90° at equinox, got {elevation}°"
    
    def test_multiple_times(self):
        """Test computing sun position for multiple times."""
        sun_obj = SunObject(lat=40.0, lon=0.0, altitude=0)
        
        times = [
            datetime(2023, 6, 21, 6, 0, 0),
            datetime(2023, 6, 21, 12, 0, 0),
            datetime(2023, 6, 21, 18, 0, 0)
        ]
        
        azimuths, elevations = sun_obj.compute_sun_location(times)
        
        assert len(azimuths) == 3
        assert len(elevations) == 3
        assert isinstance(azimuths, np.ndarray)
        assert isinstance(elevations, np.ndarray)
        
        # At solar noon (12:00), elevation should be highest
        max_elv_idx = np.argmax(elevations)
        assert max_elv_idx == 1, "Highest elevation should be at noon"
    
    def test_longitude_time_synchronization_tropic_cancer(self):
        """Test that sun stays near zenith when longitude and time are varied proportionally."""
        # Test at Tropic of Cancer (23.5°N) on June 21 (summer solstice)
        elevations = []
        
        # Vary longitude in 15-degree steps and increment hour by 1 each time
        for i in range(24):  # Test 24 different longitude/time combinations
            longitude = i * 15.0  # 0°, 15°, 30°, ..., 345°
            if longitude > 180:
                longitude -= 360  # Convert to -180 to +180 range
            
            hour_offset = i  # 0, 1, 2, ..., 23 hours
            test_time = datetime(2023, 6, 21, (12 - hour_offset) % 24, 0, 0)
            
            sun_obj = SunObject(lat=23.5, lon=longitude, altitude=0)
            _, elevation = sun_obj.compute_sun_location(test_time)
            
            elevations.append(elevation)
            
            # Each location should have sun very close to zenith (90°) at its local solar noon
            assert elevation > 85, f"At longitude {longitude}°, hour {(12 + hour_offset) % 24}, expected elevation near 90°, got {elevation}°"
        
        # Also test the same concept for Tropic of Capricorn on December 21
        elevations_south = []
        for i in range(24):
            longitude = i * 15.0
            if longitude > 180:
                longitude -= 360
            
            hour_offset = i
            test_time = datetime(2023, 12, 21, (12 - hour_offset) % 24, 0, 0)
            
            sun_obj = SunObject(lat=-23.5, lon=longitude, altitude=0)
            _, elevation = sun_obj.compute_sun_location(test_time)
            
            elevations_south.append(elevation)
            
            # At Tropic of Capricorn on winter solstice, sun should also be near zenith
            assert elevation > 85, f"At longitude {longitude}° (south), hour {(12 + hour_offset) % 24}, expected elevation near 90°, got {elevation}°"
        
        # Verify consistency - all elevations should be very similar (within a few degrees)
        max_elevation_diff = max(elevations) - min(elevations)
        assert max_elevation_diff < 1, f"Elevation variation too large: {max_elevation_diff}° (should be < 1º)"
        
        max_elevation_diff_south = max(elevations_south) - min(elevations_south)
        assert max_elevation_diff_south < 1, f"Southern elevation variation too large: {max_elevation_diff_south}° (should be < 1º)"
