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

# Create a logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logging level

stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
