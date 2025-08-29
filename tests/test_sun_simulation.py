#%%
import numpy as np
import xarray as xr
import pytest
from sunscan.signal_simulation import get_world_to_beam_matrix, get_beamcentered_unitvectors, LookupTable, get_beamcentered_coords
#%%

class TestLookupTable:
    def test_lookup_table(self):
        da=LookupTable.calculate_new(2,2, fwhm_x=[0.53], fwhm_y=[0.53]) #internally, we assume a sun diameter of 0.53, so the beam is exactly the size of the sun disk at fwhm
        # in this case, the value should be close to 0.5, since the 2D gaussian has a volume of 0.5 within the fwhm circle
        np.testing.assert_allclose(da.sel(lx=0, ly=0, method='nearest').item(), 0.5, rtol=1e-1)

        # let's make the beam very narrow, such that we essentially get the sun disk
        da=LookupTable.calculate_new(2,2, fwhm_x=[0.01], fwhm_y=[0.01])
        # for an apparent sun diameter of 10, we would expect 0 outside of the sun disk (lx>5) and 1 inside (lx<5)
        lut=LookupTable(da, apparent_sun_diameter=10)
        np.testing.assert_allclose(lut.lookup(lx=0, ly=0, fwhm_x=0.01, fwhm_y=0.01, limb_darkening=1.0).item(), 1.0, atol=1e-2)
        np.testing.assert_allclose(lut.lookup(lx=4.0, ly=0, fwhm_x=0.01, fwhm_y=0.01, limb_darkening=1.0).item(), 1.0, atol=3e-2)
        np.testing.assert_allclose(lut.lookup(lx=6.0, ly=0, fwhm_x=0.01, fwhm_y=0.01, limb_darkening=1.0).item(), 0.0, atol=3e-2)

def test_beam_unitvectors_orientation():
    beam_azi=0.0
    beam_elv=0.0
    bx, by, bz= get_beamcentered_unitvectors(beam_azi, beam_elv)
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
    anchor_azi = 0.0
    anchor_elv = 0.0
    
    # Get the transformation matrix
    matrix = get_world_to_beam_matrix(anchor_azi, anchor_elv)
    
    # see test_beam_unitvectors_orientation for the expected values
    expected = np.array([[0., 1., 0.],
                        [0., 0., 1.],
                        [1., 0., 0.]])
    
    # Test that the matrix matches the expected tangential coordinate system
    np.testing.assert_allclose(matrix, expected, atol=1e-10)


def test_world_to_beam_matrix_orthogonal():
    """Test that the transformation matrix is orthogonal (columns are orthonormal)."""
    anchor_azi = 45  # 45 degrees
    anchor_elv = 30  # 30 degrees
    
    matrix = get_world_to_beam_matrix(anchor_azi, anchor_elv)
    
    # Check that matrix is orthogonal: M @ M.T should be identity
    should_be_identity = matrix @ matrix.T
    np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1e-10)
    
    # Check that each column has unit length
    for i in range(3):
        column_norm = np.linalg.norm(matrix[:, i])
        np.testing.assert_allclose(column_norm, 1.0, atol=1e-10)


def test_world_to_beam_matrix_north_pole():
    """Test transformation matrix at the north pole (elevation = 90 degrees)."""
    anchor_azi = 0.0
    anchor_elv = 90
    
    matrix = get_world_to_beam_matrix(anchor_azi, anchor_elv)
    
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
        matrix = get_world_to_beam_matrix(anchor_azi, anchor_elv)
        
        # Check orthogonality
        should_be_identity = matrix @ matrix.T
        np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1e-10, 
                                 err_msg=f"Failed orthogonality test at azi={anchor_azi}, elv={anchor_elv}")
        
        # Check that determinant is 1 (proper rotation, not reflection)
        det = np.linalg.det(matrix)
        np.testing.assert_allclose(det, 1.0, atol=1e-10,
                                 err_msg=f"Failed determinant test at azi={anchor_azi}, elv={anchor_elv}")


def test_beam_differences():
    """ Test that one degree difference in azimuth or elevation results in about 1 unit difference in beam coordinates. """
    azi_beam, elv_beam = 123, 0 #1:1 correspondence in azimuth only works at the horizon
    azi_diff = np.linspace(-0.1, 0.1, 10) #use only small differences
    elv_diff = np.linspace(-0.1, 0.1, 10)
    azi_sun, elv_sun = azi_beam + azi_diff, elv_beam + elv_diff
    azi_beam, elv_beam = azi_beam + 0*azi_diff, elv_beam + 0*elv_diff
    azi_bc, elv_bc=get_beamcentered_coords(azi_beam, elv_beam, azi_sun, elv_sun)
    np.testing.assert_allclose(azi_diff, azi_bc, atol=1e-4)
    np.testing.assert_allclose(elv_diff, elv_bc, atol=1e-4)