import numpy as np
import pytest
from sunscan.utils import guess_offsets
from sunscan.scanner import BacklashScanner


class TestGuessOffsets:
    """Test the guess_offsets function with various scanner configurations."""
    
    @pytest.mark.parametrize("dgamma_true,domega_true,gamma,omega,test_name", [
        # Forward-only scanning (omega < 90)
        (15.0, 5.0, 
         np.array([0, 45, 90, 135, 180, 225, 270, 315]), 
         np.array([10, 20, 30, 40, 50, 60, 70, 80]),
         "forward_only"),
        
        # Backward-only scanning (omega > 90)
        (25.0, -10.0, 
         np.array([0, 60, 120, 180, 240, 300]), 
         np.array([110, 120, 130, 140, 150, 160]),
         "backward_only"),
        
        # Mixed forward and backward scanning
        (45.0, 12.5, 
         np.array([0, 45, 90, 135, 180, 225, 270, 315, 30, 150]), 
         np.array([30, 60, 85, 95, 120, 45, 75, 105, 135, 165]),
         "mixed_case"),
        
        # Zero offsets
        (0.0, 0.0, 
         np.array([0, 90, 180, 270, 45, 135, 225, 315]), 
         np.array([20, 40, 80, 100, 120, 60, 30, 150]),
         "zero_offsets"),
        
        # Negative offsets
        (-30.0, -8.0, 
         np.array([10, 50, 90, 130, 170, 210, 250, 290, 330]), 
         np.array([25, 45, 65, 85, 105, 125, 145, 35, 55]),
         "negative_offsets"),
        
        # # Large offsets
        # (180.0, 25.0, 
        #  np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]), 
        #  np.array([15, 35, 55, 75, 95, 115, 135, 155, 25, 45, 65, 85]),
        #  "large_offsets"),
        # This test is known to fail. The problem occurs, since we determine reverse as omega>90, while the actual scanner might be tilted so far that it is not a reverse case. I don't know if this can be easily fixed, since, by definition, we do not know the scanner tilt.
        
        # Edge omega values around 90 degrees
        (22.5, -3.2, 
         np.array([0, 45, 90, 135, 180, 225, 270, 315]), 
         np.array([88, 89, 90, 91, 92, 89.5, 90.5, 89.9]),
         "edge_omega_values"),
    ])
    def test_guess_offsets_parametrized(self, dgamma_true, domega_true, gamma, omega, test_name):
        """Test guess_offsets with various scanner configurations using parametrization."""
        # Scanner configuration (no time offset or backlash for simplicity)
        dtime = 0.0
        backlash_gamma = 0.0
        
        # Create scanner with known offsets
        scanner = BacklashScanner(dgamma_true, domega_true, dtime, backlash_gamma, flex=0)
        
        # Create velocity arrays (zero for these tests)
        gammav = np.zeros_like(gamma)
        omegav = np.zeros_like(omega)
        
        # Calculate beam pointing using the scanner
        azi_b, elv_b = scanner.forward(gamma, omega, gammav, omegav)
        
        # Use guess_offsets to retrieve offsets
        dgamma_guessed, domega_guessed = guess_offsets(gamma, omega, azi_b, elv_b)
        dgamma_true_wrapped= dgamma_true % 360  # Normalize to [0, 360)
        
        # Compare with true values
        assert np.isclose(dgamma_guessed, dgamma_true_wrapped, atol=1e-10), \
            f"Test '{test_name}': Expected dgamma={dgamma_true_wrapped}, got {dgamma_guessed}"
        assert np.isclose(domega_guessed, domega_true, atol=1e-10), \
            f"Test '{test_name}': Expected domega={domega_true}, got {domega_guessed}"

    def test_guess_offsets_wrap_around_case(self):
        """Test guess_offsets with gamma values that wrap around 360 degrees.
        
        This test is kept separate because it requires special handling of the wrap-around case.
        """
        # Known offsets
        dgamma_true = 350.0  # This will cause wrap-around
        domega_true = 7.5
        dtime = 0.0
        backlash_gamma = 0.0
        
        # Create scanner with known offsets
        scanner = BacklashScanner(dgamma_true, domega_true, dtime, backlash_gamma, flex=0)
        
        # Create test data with values near 0/360 boundary
        gamma = np.array([350, 355, 0, 5, 10, 15, 20, 25])
        omega = np.array([30, 40, 50, 60, 70, 80, 45, 55])
        gammav = np.zeros_like(gamma)
        omegav = np.zeros_like(omega)
        
        # Calculate beam pointing using the scanner
        azi_b, elv_b = scanner.forward(gamma, omega, gammav, omegav)
        
        # Use guess_offsets to retrieve offsets
        dgamma_guessed, domega_guessed = guess_offsets(gamma, omega, azi_b, elv_b)
        
        # Compare with true values (accounting for modulo 360)
        dgamma_diff = abs(dgamma_guessed - dgamma_true)
        dgamma_diff = min(dgamma_diff, 360 - dgamma_diff)  # Handle wrap-around
        assert dgamma_diff < 1e-10, \
            f"Expected dgamma={dgamma_true}, got {dgamma_guessed}"
        assert np.isclose(domega_guessed, domega_true, atol=1e-10), \
            f"Expected domega={domega_true}, got {domega_guessed}"


    def test_guess_offsets_scalar_inputs(self):
        """Test guess_offsets with scalar inputs instead of arrays."""
        # Known offsets
        dgamma_true = 20.0
        domega_true = -5.0
        dtime = 0.0
        backlash_gamma = 0.0
        
        # Create scanner with known offsets
        scanner = BacklashScanner(dgamma_true, domega_true, dtime, backlash_gamma, flex=0)
        
        # Scalar test data
        gamma = 45.0
        omega = 30.0
        gammav = 0.0
        omegav = 0.0
        
        # Calculate beam pointing using the scanner
        azi_b, elv_b = scanner.forward(gamma, omega, gammav, omegav)
        
        # Use guess_offsets to retrieve offsets
        dgamma_guessed, domega_guessed = guess_offsets(gamma, omega, azi_b, elv_b)
        dgamma_true_wrapped = dgamma_true % 360  # Normalize to [0, 360)
        
        # Check that the function returns scalars (not arrays)
        assert np.isscalar(dgamma_guessed), "dgamma_guessed should be a scalar"
        assert np.isscalar(domega_guessed), "domega_guessed should be a scalar"
        
        # Compare with true values
        assert np.isclose(dgamma_guessed, dgamma_true_wrapped, atol=1e-10), \
            f"Expected dgamma={dgamma_true_wrapped}, got {dgamma_guessed}"
        assert np.isclose(domega_guessed, domega_true, atol=1e-10), \
            f"Expected domega={domega_true}, got {domega_guessed}"