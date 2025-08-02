# General helper functions
import numpy as np
import xarray as xr

def cartesian_to_spherical(unit_vectors):
    """Convert unit vectors to spherical coordinates (azimuth, elevation).
    
    Args:
        unit_vectors (np.ndarray): Array of shape (N, 3) with unit vectors.
        
    Returns:
        tuple: Two arrays containing azimuth and elevation in radians.
    """
    x, y, z = unit_vectors[:, 0], unit_vectors[:, 1], unit_vectors[:, 2]
    elv = np.arcsin(z)
    azi = np.arctan2(y, x)
    return azi, elv

def spherical_to_xyz(azi, elv):
    x = np.cos(elv) * np.cos(azi)
    y = np.cos(elv) * np.sin(azi)
    z = np.sin(elv)
    return x, y, z

def spherical_to_cartesian(azi, elv):
    x,y,z= spherical_to_xyz(azi, elv)
    # Stack as (N, 3) array of unit vectors
    unit_vectors = np.stack((x, y, z), axis=-1)
    return unit_vectors

