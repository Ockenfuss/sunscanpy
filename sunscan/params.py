from importlib import resources

def _get_default_lut_path():
    """Get the default path for the lookup table."""
    traversable = resources.files('sunscan') / 'data' / 'lut.nc'
    with resources.as_file(traversable) as path:
        return str(path)

SCANNER_PARAMETER_MAP = {
    'gamma_offset': 0,
    'omega_offset': 1,
    'alpha': 2,
    'delta': 3,
    'beta': 4,
    'epsilon': 5,
    'flex': 6
}
SUNSIM_PARAMETER_MAP = {
    "dgamma": 0,
    "domega": 1,
    "fwhm_x": 2,
    "fwhm_y": 3,
    "dtime": 4,
    "backlash_gamma": 5,
    "limb_darkening": 6
}

sc_params={
    "lutpath": _get_default_lut_path(),
    "lut_dgamma_range": 3,
    "lut_domega_range": 3,
    "lut_fwhm_x": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "lut_fwhm_y": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "lut_limb_darkening": [1.0],#[0.95, 0.975, 1.0],
    "sunsim_params_optimize": ['dgamma', 'domega', 'fwhm_x', 'fwhm_y', 'dtime', 'backlash_gamma'],
    "sunsim_params_guess": {
        'dgamma': None,
        'domega': None,
        'fwhm_x': 0.6,
        'fwhm_y': 0.6,
        'dtime': 0.0,
        'backlash_gamma': 0.0,
        'limb_darkening': 1.0
    },
    "sunsim_params_bounds": {
        'dgamma': (-0.5, 0.5),
        'domega': (-0.5, 0.5),
        'fwhm_x': (-0.3, 0.3),
        'fwhm_y': (-0.3, 0.3),
        'dtime': (-1.0, 1.0),
        'backlash_gamma': (-0.2, 0.2),
        'limb_darkening': (-0.05, 0.00001)
    },
    "scanner_params_optimize": ['gamma_offset', 'omega_offset', 'alpha', 'delta', 'beta', 'epsilon'],
    "scanner_params_guess": {
        'gamma_offset': None,
        'omega_offset': None,
        'alpha': 0.0,
        'delta': 0.0,
        'beta': 0.0,
        'epsilon': 0.0,
        'flex': 0.0
    }, 
    "scanner_params_bounds": {
        'gamma_offset': (-5.0, 5.0),
        'omega_offset': (-5.0, 5.0),
        'alpha': (-1.0, 1.0),
        'delta': (-1.0, 1.0),
        'beta': (-0.1, 0.1),
        'epsilon': (-2.0, 2.0),
        'flex': (-0.1, 0.0)
    }
}
