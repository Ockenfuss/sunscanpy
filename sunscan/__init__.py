"""
Sunscan - A Python module for performing and evaluating radar sun scans.
"""
import logging
from .sky import SkyObject

__version__ = "0.1.0"
__author__ = "Paul Ockenfuss, Gregor Köcher"
__email__ = "paul.ockenfuss@physik.uni-muenchen.de"

__all__ = [
    "SkyObject"
]

sc_params={
    "lutpath": "sunscan/data/lut.nc",
    "lut_dgamma_range": 2,
    "lut_domega_range": 2,
    "lut_fwhm_x": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "lut_fwhm_y": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "lut_limb_darkening": [1.0],#[0.95, 0.975, 1.0],
    "sunsim_params_optimize": ['dgamma', 'domega', 'fwhm_x', 'fwhm_y', 'dt', 'backlash'],
    "sunsim_params_guess": {
        'dgamma': None,
        'domega': None,
        'fwhm_x': 0.6,
        'fwhm_y': 0.6,
        'dt': 0.0,
        'backlash': 0.0
    },
    "sunsim_params_bounds": {
        'dgamma': (-0.5, 0.5),
        'domega': (-0.5, 0.5),
        'fwhm_x': (0.3, 0.9),
        'fwhm_y': (0.3, 0.9),
        'dt': (-1.0, 1.0),
        'backlash': (-0.2, 0.2),
        'limb_darkening': (0.95, 1.0)
    },
    "scanner_params_optimize": ['azi_offset', 'elv_offset', 'alpha', 'delta', 'beta', 'epsilon'],
    "scanner_params_guess": {
        'azi_offset': None,
        'elv_offset': None,
        'alpha': 0.0,
        'delta': 0.0,
        'beta': 0.0,
        'epsilon': 0.0
    }, 
    "scanner_params_bounds": {
        'azi_offset': (-5.0, 5.0),
        'elv_offset': (-5.0, 5.0),
        'alpha': (-1.0, 1.0),
        'delta': (-1.0, 1.0),
        'beta': (-0.1, 0.1),
        'epsilon': (-2.0, 2.0)
    }
}

# Create a logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logging level

stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
