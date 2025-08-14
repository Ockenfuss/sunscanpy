#%%
import numpy as np
import xarray as xr
import pytest
from sunscan.sun_simulation import get_world_to_beam_matrix, get_beamcentered_unitvectors, LookupTable
#%%

class TestLookupTable:
    def test_lookup_table(self):
        da=LookupTable.calculate_new(2,2, fwhm_x=[0.1], fwhm_y=[0.1])
        da.squeeze().plot()
        lut=LookupTable(da, apparent_sun_diameter=10)
        #%%
        lut.lookup(lx=4, ly=0.0, fwhm_x=0.1, fwhm_y=0.1, limb_darkening=1.0)
        #%%


def test_beam_unitvectors_orientation():
    beam_azi=0.0
    beam_elv=0.0
    bx, by, bz= get_beamcentered_unitvectors(beam_azi, beam_elv)
    bx = bx.squeeze().values
    by = by.squeeze().values
    bz = bz.squeeze().values
    # For azimuth=0, elevation=0, the anchor point is at (1, 0, 0)
    # (remember: in world coordinates, x points north and y points east)
    # The beam coordinate system at this point should be:
    # - beam x (cross-elevation): (0, 1, 0) - pointing east
    # - beam y (co-elevation): (0, 0, 1) - pointing up
    # - beam z (radial): (1, 0, 0) - pointing outward
    np.testing.assert_allclose(bx, [0, 1, 0], atol=1e-10)
    np.testing.assert_allclose(by, [0, 0, 1], atol=1e-10)
    np.testing.assert_allclose(bz, [1, 0, 0], atol=1e-10)



def test_world_to_beam_matrix_identity():
    anchor_azi = np.array([0.0])
    anchor_elv = np.array([0.0])
    
    # Get the transformation matrix
    transform_matrix = get_world_to_beam_matrix(anchor_azi, anchor_elv)
    
    # Convert to numpy for easier testing (squeeze to remove sample dimension)
    matrix = transform_matrix.squeeze().values
    # see test_beam_unitvectors_orientation for the expected values
    expected = np.array([[0., 1., 0.],
                        [0., 0., 1.],
                        [1., 0., 0.]])
    
    # Test that the matrix matches the expected tangential coordinate system
    np.testing.assert_allclose(matrix, expected, atol=1e-10)


def test_world_to_beam_matrix_orthogonal():
    """Test that the transformation matrix is orthogonal (columns are orthonormal)."""
    anchor_azi = np.array([45])  # 45 degrees
    anchor_elv = np.array([30])  # 30 degrees
    
    transform_matrix = get_world_to_beam_matrix(anchor_azi, anchor_elv)
    matrix = transform_matrix.squeeze().transpose('row', 'col', ...).values
    
    # Check that matrix is orthogonal: M @ M.T should be identity
    should_be_identity = matrix @ matrix.T
    np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1e-10)
    
    # Check that each column has unit length
    for i in range(3):
        column_norm = np.linalg.norm(matrix[:, i])
        np.testing.assert_allclose(column_norm, 1.0, atol=1e-10)


def test_world_to_beam_matrix_north_pole():
    """Test transformation matrix at the north pole (elevation = 90 degrees)."""
    anchor_azi = np.array([0.0])
    anchor_elv = np.array([90])
    
    transform_matrix = get_world_to_beam_matrix(anchor_azi, anchor_elv)
    matrix = transform_matrix.squeeze().values
    
    # At the north pole, the z-axis (world up) should become the local z-axis
    # The local z-axis should point towards (0, 0, 1)
    local_z = matrix[2, :]  # third row
    expected_z = np.array([0, 0, 1])
    np.testing.assert_allclose(local_z, expected_z, atol=1e-10)
    
    # Matrix should still be orthogonal
    should_be_identity = matrix @ matrix.T
    np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1e-10)


def test_world_to_beam_matrix_orthogonality_determinant():
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
        transform_matrix = get_world_to_beam_matrix(np.array([anchor_azi]), np.array([anchor_elv]))
        matrix = transform_matrix.squeeze().values
        
        # Check orthogonality
        should_be_identity = matrix @ matrix.T
        np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1e-10, 
                                 err_msg=f"Failed orthogonality test at azi={anchor_azi}, elv={anchor_elv}")
        
        # Check that determinant is 1 (proper rotation, not reflection)
        det = np.linalg.det(matrix)
        np.testing.assert_allclose(det, 1.0, atol=1e-10,
                                 err_msg=f"Failed determinant test at azi={anchor_azi}, elv={anchor_elv}")


def test_world_to_beam_matrix_xarray_input():
    """Test that the function works with xarray inputs."""
    anchor_azi = xr.DataArray([0.0, np.pi/4], dims='sample')
    anchor_elv = xr.DataArray([0.0, np.pi/6], dims='sample')
    
    # This should work without raising an error
    transform_matrix = get_world_to_beam_matrix(anchor_azi, anchor_elv)
    
    # Check that result has the expected dimensions
    assert 'row' in transform_matrix.dims
    assert 'col' in transform_matrix.dims
    assert transform_matrix.sizes['row'] == 3
    assert transform_matrix.sizes['col'] == 3