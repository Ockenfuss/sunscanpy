# General helper functions
import numpy as np
import xarray as xr
from sunscan.scanner import IdentityScanner
import logging

def guess_offsets(gamma, omega, azi_b, elv_b):
    reverse = omega > 90
    gamma_id, omega_id = IdentityScanner().inverse(azi_b, elv_b, reverse=reverse)
    gamoff_guess = gamma_id-gamma
    gamoff_guess = np.atleast_1d(gamoff_guess)
    gamoff_guess = np.sort(gamoff_guess)[len(gamoff_guess)//2]  # median element
    gamoff_guess = gamoff_guess % 360
    omoff = omega_id-omega
    omoff = np.atleast_1d(omoff)
    omoff = np.sort(omoff)[len(omoff)//2]
    return gamoff_guess, omoff

def format_input_xarray(arr):
    if isinstance(arr, xr.DataArray):
        return arr
    # numeric scalar → 1D DataArray of length 1
    elif np.isscalar(arr) and np.issubdtype(np.asarray(arr).dtype, np.number) and not isinstance(arr, (bool, np.bool_)):
        return xr.DataArray(np.asarray([arr]), dims='sample')
    elif isinstance(arr, np.ndarray):
        if arr.ndim != 1:
            raise ValueError('Input array must be 1D')
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError('Input array must be numeric')
        return xr.DataArray(arr, dims='sample')
    elif isinstance(arr, (list, tuple)):
        arr = np.asarray(arr)
        if arr.ndim != 1:
            raise ValueError('Input list or tuple must be 1D')
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError('Input list or tuple must be numeric')
        return xr.DataArray(arr, dims='sample')
    else:
        raise ValueError(f'Input must be a 1D numeric array, list/tuple, or xarray DataArray. Got {type(arr)} instead.')

def db_to_linear(signal_db):
    """Convert a signal in dB to linear scale."""
    return 10**(signal_db/10)

def linear_to_db(signal_linear):
    """Convert a signal in linear scale to dB."""
    return 10 * np.log10(signal_linear)


# Create a logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logging level

stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - sunscan - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)