import numpy as np
import pytest
from sunscan.scanner import IdentityScanner, BacklashScanner


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
            {'dgamma': 0, 'domega': 0, 'dtime': 0, 'backlash_gamma': 0},  # Identity case
            {'dgamma': 5, 'domega': 2, 'dtime': 0.1, 'backlash_gamma': 1.5},  # Typical values
            {'dgamma': -3, 'domega': -1, 'dtime': 0.05, 'backlash_gamma': 0.8},  # Negative offsets
            {'dgamma': 10, 'domega': 5, 'dtime': 0.2, 'backlash_gamma': 2.0},  # Large values
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
                            reverse=(omega+scanner.domega+omegav*scanner.dtime)>90
                            gamma_result, omega_result = scanner.inverse(azi, elv, gammav, omegav, reverse=reverse)
                            
                            # Check that we get back the original values
                            np.testing.assert_allclose(omega_result, omega, rtol=1e-10, atol=1e-10,
                                                     err_msg=f"Failed for params {params}, gamma={gamma}, omega={omega}, gammav={gammav}, omegav={omegav}, azi={azi}, elv={elv}, reverse={reverse}")
                            np.testing.assert_allclose(gamma_result, gamma, rtol=1e-10, atol=1e-10,
                                                     err_msg=f"Failed for params {params}, gamma={gamma}, omega={omega}, gammav={gammav}, omegav={omegav}")
    
    def test_backlash_scanner_array_inputs(self):
        """Test BacklashScanner with array inputs."""
        scanner = BacklashScanner(dgamma=2, domega=1, dtime=0.1, backlash_gamma=1.0)
        
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
        scanner = BacklashScanner(dgamma=1, domega=0.5, dtime=0.05, backlash_gamma=0.5)
        
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
