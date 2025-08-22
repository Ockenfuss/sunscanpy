import pytest
from sunscan.math_utils import bessel, gaussian
import numpy as np

def test_gaussian():
    """Test that gaussian function returns 0.5 at FWHM/2 points."""
    # Test various FWHM values
    fwhm_values = [1.0, 2.0, 5.0, 10.0, 0.5]
    
    for fwhm_x in fwhm_values:
        for fwhm_y in fwhm_values:
            # Test at (±fwhm_x/2, 0)
            result1 = gaussian(fwhm_x/2, 0, fwhm_x, fwhm_y)
            result2 = gaussian(-fwhm_x/2, 0, fwhm_x, fwhm_y)
            
            # Test at (0, ±fwhm_y/2)
            result3 = gaussian(0, fwhm_y/2, fwhm_x, fwhm_y)
            result4 = gaussian(0, -fwhm_y/2, fwhm_x, fwhm_y)
            
            # All should be approximately 0.5
            np.testing.assert_allclose(result1, 0.5, rtol=1e-10, 
                                     err_msg=f"Failed for fwhm_x={fwhm_x}, fwhm_y={fwhm_y} at (+fwhm_x/2, 0)")
            np.testing.assert_allclose(result2, 0.5, rtol=1e-10, 
                                     err_msg=f"Failed for fwhm_x={fwhm_x}, fwhm_y={fwhm_y} at (-fwhm_x/2, 0)")
            np.testing.assert_allclose(result3, 0.5, rtol=1e-10, 
                                     err_msg=f"Failed for fwhm_x={fwhm_x}, fwhm_y={fwhm_y} at (0, +fwhm_y/2)")
            np.testing.assert_allclose(result4, 0.5, rtol=1e-10, 
                                     err_msg=f"Failed for fwhm_x={fwhm_x}, fwhm_y={fwhm_y} at (0, -fwhm_y/2)")

def test_bessel():
    """Test that bessel function returns 0.5 at FWHM/2 points."""
    # Test various FWHM values
    fwhm_values = [1.0, 2.0, 5.0, 10.0, 0.5]
    
    for fwhm_x in fwhm_values:
        for fwhm_y in fwhm_values:
            # Test at (±fwhm_x/2, 0)
            result1 = bessel(fwhm_x/2, 0, fwhm_x, fwhm_y)
            result2 = bessel(-fwhm_x/2, 0, fwhm_x, fwhm_y)
            
            # Test at (0, ±fwhm_y/2)
            result3 = bessel(0, fwhm_y/2, fwhm_x, fwhm_y)
            result4 = bessel(0, -fwhm_y/2, fwhm_x, fwhm_y)
            
            # All should be approximately 0.5
            np.testing.assert_allclose(result1, 0.5, rtol=1e-10, 
                                     err_msg=f"Failed for fwhm_x={fwhm_x}, fwhm_y={fwhm_y} at (+fwhm_x/2, 0)")
            np.testing.assert_allclose(result2, 0.5, rtol=1e-10, 
                                     err_msg=f"Failed for fwhm_x={fwhm_x}, fwhm_y={fwhm_y} at (-fwhm_x/2, 0)")
            np.testing.assert_allclose(result3, 0.5, rtol=1e-10, 
                                     err_msg=f"Failed for fwhm_x={fwhm_x}, fwhm_y={fwhm_y} at (0, +fwhm_y/2)")
            np.testing.assert_allclose(result4, 0.5, rtol=1e-10, 
                                     err_msg=f"Failed for fwhm_x={fwhm_x}, fwhm_y={fwhm_y} at (0, -fwhm_y/2)")
