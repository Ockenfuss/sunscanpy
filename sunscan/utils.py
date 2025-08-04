# General helper functions
import numpy as np
import xarray as xr
from sunscan.scanner import IdentityScanner

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
    elif isinstance(arr, np.ndarray):
        if arr.ndim != 1:
            raise ValueError('Input array must be 1D')
        return xr.DataArray(arr, dims='sample')
    elif isinstance(arr, (list, tuple)):
        arr = np.array(arr)
        if arr.ndim != 1:
            raise ValueError('Input list or tuple must be 1D')
        return xr.DataArray(arr, dims='sample')
    else:
        raise ValueError(f'Input must be a 1D numpy array or xarray DataArray. Got {type(arr)} instead.')

