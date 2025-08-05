"""
Sunscan - A Python module for performing and evaluating radar sun scans.
"""
import logging
from .sky import SkyObject
from .params import sc_params
from .sun_simulation import SunSimulator, SunSimulationEstimator

__version__ = "0.1.0"
__author__ = "Paul Ockenfuss, Gregor Köcher"
__email__ = "paul.ockenfuss@physik.uni-muenchen.de"

__all__ = [
    "SkyObject",
    "sc_params",
]

