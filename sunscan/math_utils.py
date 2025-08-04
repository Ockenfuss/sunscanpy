
import numpy as np
import xarray as xr

def rmse(values):
    """
    Calculate the root mean square error (RMSE) between two sets of values.
    """
    return np.sqrt(np.mean(values ** 2))

def difference_angles(vec1, vec2):
    """
    Calculate the difference in angles between two vectors.
    Returns the angle in radians.
    """
    dot_product = (vec1 * vec2).sum(axis=-1)
    dot_product=np.clip(dot_product, -1.0, 1.0)
    angle = np.arccos(dot_product)
    return np.rad2deg(angle)



def cartesian_to_spherical(unit_vectors):
    """Convert unit vectors to spherical coordinates (azimuth, elevation).
    
    Args:
        unit_vectors (np.ndarray): Array of shape (N, 3) with unit vectors.
        
    Returns:
        tuple: Two arrays containing azimuth and elevation in degrees.
    """
    x, y, z = unit_vectors[:, 0], unit_vectors[:, 1], unit_vectors[:, 2]
    elv = np.arcsin(z)
    azi = np.arctan2(y, x)
    return np.rad2deg(azi), np.rad2deg(elv)

def spherical_to_xyz(azi, elv):
    azi_rad= np.deg2rad(azi)
    elv_rad= np.deg2rad(elv)
    x = np.cos(elv_rad) * np.cos(azi_rad)
    y = np.cos(elv_rad) * np.sin(azi_rad)
    z = np.sin(elv_rad)
    return x, y, z

def spherical_to_cartesian(azi, elv):
    x,y,z= spherical_to_xyz(azi, elv)
    # Stack as (N, 3) array of unit vectors
    unit_vectors = np.stack((x, y, z), axis=-1)
    return unit_vectors
