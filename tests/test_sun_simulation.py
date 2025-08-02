import numpy as np
import xarray as xr
import pytest
from sunscan.sun_simulation import _cart_to_tangential_matrix


def test_cart_to_tangential_matrix_identity():
    """Test that _cart_to_tangential_matrix returns identity matrix for azimuth=0, elevation=0."""
    anchor_azi = np.array([0.0])
    anchor_elv = np.array([0.0])
    
    # Get the transformation matrix
    transform_matrix = _cart_to_tangential_matrix(anchor_azi, anchor_elv)
    
    # Convert to numpy for easier testing (squeeze to remove sample dimension)
    matrix = transform_matrix.squeeze().values
    
    # For azimuth=0, elevation=0, the anchor point is at (1, 0, 0)
    # The local coordinate system at this point should be:
    # - local x (cross-elevation): (0, 1, 0) - tangent to longitude
    # - local y (co-elevation): (0, 0, 1) - tangent to latitude  
    # - local z (radial): (1, 0, 0) - pointing outward
    expected = np.array([[0., 1., 0.],
                        [0., 0., 1.],
                        [1., 0., 0.]])
    
    # Test that the matrix matches the expected tangential coordinate system
    np.testing.assert_allclose(matrix, expected, atol=1e-10)


def test_cart_to_tangential_matrix_orthogonal():
    """Test that the transformation matrix is orthogonal (columns are orthonormal)."""
    anchor_azi = np.array([np.pi/4])  # 45 degrees
    anchor_elv = np.array([np.pi/6])  # 30 degrees
    
    transform_matrix = _cart_to_tangential_matrix(anchor_azi, anchor_elv)
    matrix = transform_matrix.squeeze().transpose('row', 'col', ...).values
    
    # Check that matrix is orthogonal: M @ M.T should be identity
    should_be_identity = matrix @ matrix.T
    np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1e-10)
    
    # Check that each column has unit length
    for i in range(3):
        column_norm = np.linalg.norm(matrix[:, i])
        np.testing.assert_allclose(column_norm, 1.0, atol=1e-10)


def test_cart_to_tangential_matrix_north_pole():
    """Test transformation matrix at the north pole (elevation = 90 degrees)."""
    anchor_azi = np.array([0.0])
    anchor_elv = np.array([np.pi/2])  # 90 degrees
    
    transform_matrix = _cart_to_tangential_matrix(anchor_azi, anchor_elv)
    matrix = transform_matrix.squeeze().values
    
    # At the north pole, the z-axis (world up) should become the local z-axis
    # The local z-axis should point towards (0, 0, 1)
    local_z = matrix[2, :]  # third row
    expected_z = np.array([0, 0, 1])
    np.testing.assert_allclose(local_z, expected_z, atol=1e-10)
    
    # Matrix should still be orthogonal
    should_be_identity = matrix @ matrix.T
    np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1e-10)


def test_cart_to_tangential_matrix_various_positions():
    """Test transformation matrix at various anchor positions."""
    test_positions = [
        (0, 0),           # Origin
        (np.pi/2, 0),     # East
        (np.pi, 0),       # South
        (3*np.pi/2, 0),   # West
        (0, np.pi/4),     # Northeast, 45° elevation
        (np.pi/2, np.pi/4), # Southeast, 45° elevation
    ]
    
    for anchor_azi, anchor_elv in test_positions:
        transform_matrix = _cart_to_tangential_matrix(np.array([anchor_azi]), np.array([anchor_elv]))
        matrix = transform_matrix.squeeze().values
        
        # Check orthogonality
        should_be_identity = matrix @ matrix.T
        np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1e-10, 
                                 err_msg=f"Failed orthogonality test at azi={anchor_azi}, elv={anchor_elv}")
        
        # Check that determinant is 1 (proper rotation, not reflection)
        det = np.linalg.det(matrix)
        np.testing.assert_allclose(det, 1.0, atol=1e-10,
                                 err_msg=f"Failed determinant test at azi={anchor_azi}, elv={anchor_elv}")


def test_cart_to_tangential_matrix_xarray_input():
    """Test that the function works with xarray inputs."""
    anchor_azi = xr.DataArray([0.0, np.pi/4], dims='sample')
    anchor_elv = xr.DataArray([0.0, np.pi/6], dims='sample')
    
    # This should work without raising an error
    transform_matrix = _cart_to_tangential_matrix(anchor_azi, anchor_elv)
    
    # Check that result has the expected dimensions
    assert 'row' in transform_matrix.dims
    assert 'col' in transform_matrix.dims
    assert transform_matrix.sizes['row'] == 3
    assert transform_matrix.sizes['col'] == 3