"""
Sunscan - A Python module for performing and evaluating radar sun scans.
"""
from .sky import SunObject, MoonObject
from .params import sc_params
from .signal_simulation import SignalSimulator, SignalSimulationEstimator
from .scanner_estimation import ScannerEstimator

__version__ = "0.1.3"
__author__ = "Paul Ockenfuss, Gregor Köcher"
__email__ = "paul.ockenfuss@physik.uni-muenchen.de"

__all__ = [
    "SunObject",
    "MoonObject",
    "sc_params",
    "SignalSimulator",
    "SignalSimulationEstimator",
    "ScannerEstimator",
]
