import numpy as np
import pytest
from sunscan.scanner import IdentityScanner, BacklashScanner, GeneralScanner
from sunscan.math_utils import calc_azi_diff


class TestIdentityScanner:
    def setup_method(self):
        """Set up test fixtures."""
        self.scanner = IdentityScanner()
    
    def test_identity_scanner_normal_case(self):
        """Test that forward + inverse is identity for omega < 90 degrees."""
        # Test data for normal case (omega < 90)
        gamma_test = np.array([0, 45, 90, 135, 180, 225, 270, 315, 359])
        omega_test = np.array([0, 15, 30, 45, 60, 75, 89])
        
        # Create all combinations
        gamma_grid, omega_grid = np.meshgrid(gamma_test, omega_test)
        gamma_flat = gamma_grid.flatten()
        omega_flat = omega_grid.flatten()
        
        # Apply forward transformation
        azi, elv = self.scanner.forward(gamma_flat, omega_flat)
        
        # Apply inverse transformation (reverse=False for normal case)
        gamma_result, omega_result = self.scanner.inverse(azi, elv, reverse=False)
        
        # Check that we get back the original values
        np.testing.assert_allclose(gamma_result, gamma_flat, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(omega_result, omega_flat, rtol=1e-10, atol=1e-10)
    
    def test_identity_scanner_reverse_case(self):
        """Test that forward + inverse is identity for omega > 90 degrees."""
        # Test data for reverse case (omega > 90)
        gamma_test = np.array([0, 45, 90, 135, 180, 225, 270, 315, 359])
        omega_test = np.array([91, 105, 120, 135, 150, 165, 180])
        
        # Create all combinations
        gamma_grid, omega_grid = np.meshgrid(gamma_test, omega_test)
        gamma_flat = gamma_grid.flatten()
        omega_flat = omega_grid.flatten()
        
        # Apply forward transformation
        azi, elv = self.scanner.forward(gamma_flat, omega_flat)
        
        # Apply inverse transformation (reverse=True for reverse case)
        gamma_result, omega_result = self.scanner.inverse(azi, elv, reverse=True)
        
        # Check that we get back the original values
        np.testing.assert_allclose(gamma_result, gamma_flat, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(omega_result, omega_flat, rtol=1e-10, atol=1e-10)
    
    def test_identity_scanner_edge_cases(self):
        """Test edge cases like omega = 90 degrees."""
        # Test omega = 90 (boundary case)
        gamma_test = np.array([0, 90, 180, 270])
        omega_test = np.array([90])
        
        gamma_grid, omega_grid = np.meshgrid(gamma_test, omega_test)
        gamma_flat = gamma_grid.flatten()
        omega_flat = omega_grid.flatten()
        
        # Apply forward transformation
        azi, elv = self.scanner.forward(gamma_flat, omega_flat)
        
        # For omega = 90, should use normal case (reverse=False)
        gamma_result, omega_result = self.scanner.inverse(azi, elv, reverse=False)
        
        # Check that we get back the original values
        np.testing.assert_allclose(gamma_result, gamma_flat, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(omega_result, omega_flat, rtol=1e-10, atol=1e-10)


class TestBacklashScanner:
    def setup_method(self):
        """Set up test fixtures."""
        # Test with various parameter combinations
        self.test_params = [
            {'gamma_offset': 0, 'omega_offset': 0, 'dtime': 0, 'backlash_gamma': 0},  # Identity case
            {'gamma_offset': 5, 'omega_offset': 2, 'dtime': 0.1, 'backlash_gamma': 1.5},  # Typical values
            {'gamma_offset': -3, 'omega_offset': -1, 'dtime': 0.05, 'backlash_gamma': 0.8},  # Negative offsets
            {'gamma_offset': 10, 'omega_offset': 5, 'dtime': 0.2, 'backlash_gamma': 2.0},  # Large values
        ]
    
    def test_backlash_scanner(self):
        """Test that forward + inverse is identity."""
        for params in self.test_params:
            scanner = BacklashScanner(**params)
            
            # Test data for normal case (omega < 90)
            gamma_test = np.array([0, 45, 90, 135, 180, 270])
            omega_test = np.array([0, 30, 60, 89, 91, 120, 179])
            gammav_test = np.array([0, 0.5, 2, -2, -0.5])
            omegav_test = np.array([0, 1,-1])
            
            # Create combinations
            for gamma in gamma_test:
                for omega in omega_test:
                    for gammav in gammav_test:
                        for omegav in omegav_test:
                            # Apply forward transformation
                            azi, elv = scanner.forward(gamma, omega, gammav, omegav)
                            
                            # Apply inverse transformation
                            # for the backlash scanner, the forward-reverse transition is not exactly at omega=90 deg, but can be shifted
                            reverse=(omega+scanner.omega_offset+omegav*scanner.dtime)>90
                            gamma_result, omega_result = scanner.inverse(azi, elv, gammav, omegav, reverse=reverse)
                            
                            # Check that we get back the original values
                            np.testing.assert_allclose(omega_result, omega, rtol=1e-10, atol=1e-10,
                                                     err_msg=f"Failed for params {params}, gamma={gamma}, omega={omega}, gammav={gammav}, omegav={omegav}, azi={azi}, elv={elv}, reverse={reverse}")
                            np.testing.assert_allclose(gamma_result, gamma, rtol=1e-10, atol=1e-10,
                                                     err_msg=f"Failed for params {params}, gamma={gamma}, omega={omega}, gammav={gammav}, omegav={omegav}")
    
    def test_backlash_scanner_array_inputs(self):
        """Test BacklashScanner with array inputs."""
        scanner = BacklashScanner(gamma_offset=2, omega_offset=1, dtime=0.1, backlash_gamma=1.0)
        
        # Array inputs
        gamma_array = np.array([0, 45, 90, 135])
        omega_array = np.array([30, 45, 60, 75])
        gammav_array = np.array([1, -1, 0.5, -0.5])
        omegav_array = np.array([0.5, -0.5, 0.2, -0.2])
        
        # Apply forward transformation
        azi, elv = scanner.forward(gamma_array, omega_array, gammav_array, omegav_array)
        
        # Apply inverse transformation
        gamma_result, omega_result = scanner.inverse(azi, elv, gammav_array, omegav_array, reverse=False)
        
        # Check that we get back the original values
        np.testing.assert_allclose(gamma_result, gamma_array, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(omega_result, omega_array, rtol=1e-10, atol=1e-10)
    
    def test_backlash_scanner_edge_cases(self):
        """Test edge cases for BacklashScanner."""
        scanner = BacklashScanner(gamma_offset=1, omega_offset=0.5, dtime=0.05, backlash_gamma=0.5)
        
        # Test with zero velocities
        gamma, omega = 180, 45
        azi, elv = scanner.forward(gamma, omega, 0, 0)
        gamma_result, omega_result = scanner.inverse(azi, elv, 0, 0, reverse=False)
        
        np.testing.assert_allclose(gamma_result, gamma, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(omega_result, omega, rtol=1e-10, atol=1e-10)
        
        # Test boundary case omega = 90
        gamma, omega = 90, 90
        azi, elv = scanner.forward(gamma, omega, 1, 0.5)
        gamma_result, omega_result = scanner.inverse(azi, elv, 1, 0.5, reverse=True)
        
        np.testing.assert_allclose(gamma_result, gamma, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(omega_result, omega, rtol=1e-10, atol=1e-10)

class TestGeneralScanner:
    @pytest.mark.parametrize("azi,elv", [
        (0, 30),
        (90, 45),
        (180, 60),
        (270, 75),
        (359, 10),
        (180, 80),   # >75 deg, should trigger warning
        (90, 89),    # >75 deg, should trigger warning
    ])
    @pytest.mark.parametrize("reverse", [True, False, None])
    def test_inverse_accuracy(self, azi, elv, reverse):
        epsilon = 15.0  # degrees tilt
        scanner = GeneralScanner(
            gamma_offset=0.0,
            omega_offset=0.0,
            alpha=0.0,
            delta=0.0,
            beta=0.0,
            epsilon=epsilon,
            dtime=0.0,
            backlash_gamma=0.0
        )
        max_elv = 90 - epsilon
        import warnings
        # debug
        # azi=0
        # elv=30
        # reverse=False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gamma, omega = scanner.inverse(azi, elv, gammav=0, omegav=0, reverse=reverse)
            azi_check, elv_check = scanner.forward(gamma, omega, gammav=0, omegav=0)

            if reverse is True:
                assert omega >= 90, f"Expected omega >= 90 for reverse=True, got {omega}"
            elif reverse is False:
                assert omega <= 90, f"Expected omega < 90 for reverse=False, got {omega}"

            if elv > max_elv:
                # Should have raised a warning
                assert any("Inversion imperfect" in str(warn.message) for warn in w), \
                    f"Expected warning for elv={elv}, got none"
                # Deviation should be target - max_elv
                np.testing.assert_allclose(elv_check, max_elv, atol=0.2)
                np.testing.assert_allclose(elv_check - elv, max_elv - elv, atol=0.2)
            else:
                # No warning expected, check closeness
                np.testing.assert_allclose(azi_diff(azi_check, azi), 0, atol=0.2)
                np.testing.assert_allclose(elv_check, elv, atol=0.2)
