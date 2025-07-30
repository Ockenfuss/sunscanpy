import xarray as xr
import numpy as np
import ikpy
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import ikpy.utils
import ikpy.utils.plot
from sunscan.utils import spherical_to_cartesian




class Scanner(object):
    """Base class for scanner models."""
    def __init__(self):
        pass
    
    def forward(self, gamma, omega):
        """Forward model: maps scanner angles to azimuth and elevation."""
        raise NotImplementedError("Please implement the forward method in the subclass.")
    def inverse(self, azi, elv):
        """Inverse model: maps azimuth and elevation to scanner angles."""
        raise NotImplementedError("Please implement the inverse method in the subclass.")

class IdentityScanner(Scanner):
    def forward(self, gamma, omega):
        """Identity scanner model M_I(gamma, omega) = (phi, theta)
        This model assumes a perfectly oriented scanner.
        For omega <=90 degrees, it is the identity function: gamma=phi, omega=theta.
        """
        reverse = omega > 90
        azi = xr.where(reverse, (gamma+180) % 360, gamma)
        elv = xr.where(reverse, 180 - omega, omega)
        return azi, elv

    def inverse(self, azi, elv, reverse=False):
        """Invert the identity radar model.

        Since the identity model is bijective, you need to specify whether azi,elv should be mapped in the forward or 
        reverse part of the gamma-omega space.

        """
        if reverse:
            gamma = (azi - 180) % 360
            omega = 180 - elv
        else:
            gamma = azi
            omega = elv
        return gamma, omega

#%% 2D pan tilt system
def generate_pt_chain(azi_offset, elv_offset, alpha, delta, beta, epsilon):
    d=1
    pt_chain= Chain(name='pan_tilt', links=[
        OriginLink(),
        URDFLink(
            name="pan",
            origin_translation=[0, 0, d/3],
            origin_orientation=[alpha, delta, 0],
            rotation=[0, 0, 1],
        ),
        URDFLink(
            name='azimuth_offset',
            origin_translation=[0, 0, d],
            origin_orientation=[0, 0, azi_offset],
            rotation=[0, 0, 1], 
        ),
        URDFLink(
            name="tilt",
            origin_translation=[0,0,d],
            origin_orientation=[beta,0,  0],
            rotation=[0, -1, 0],
        ),
        URDFLink(
            name='elevation_offset',
            origin_translation=[0, 0, d/10],
            origin_orientation=[0, np.pi/2-elv_offset, 0],
            rotation=[0, 1, 0],
        ),
        URDFLink(
            name="dish",
            origin_translation=[0, 0, d],
            origin_orientation=[epsilon, 0, 0],
            rotation=[1, 0, 0],
        )
    ],
        active_links_mask=[False, True, False, True, False, False]
    )
    return pt_chain

def _vector_to_azielv(z_axis, x_axis=None, eps=1e-8):
    """
    Calculates azimuth and elevation from a direction vector.
    If the z axis is vertical (x and y near zero), uses the x axis for azimuth.
    Returns azimuth (radians), elevation (radians)
    """
    z_axis = np.asarray(z_axis)
    # Elevation: angle from xy-plane
    elv = np.arcsin(z_axis[2] / np.linalg.norm(z_axis))
    # Azimuth: angle in xy-plane from x-axis
    if np.abs(z_axis[0]) < eps and np.abs(z_axis[1]) < eps:
        # z axis is vertical, use x axis for azimuth
        if x_axis is None:
            raise ValueError("x_axis must be provided when z_axis is vertical")
        x_axis = np.asarray(x_axis)
        azi = np.arctan2(x_axis[1], x_axis[0])
    else:
        azi = np.arctan2(z_axis[1], z_axis[0])
    return azi, elv

def _gam_om_to_joint_positions(gamma, omega):
    """ gamma (azimuth) and omega (elevation) in radians"""
    return [0, gamma, 0, omega, 0, 0]

def _joint_positions_to_gam_om(positions):
    """ positions in radians"""
    gamma = positions[1]
    omega = positions[3]
    return gamma, omega

class GeneralScanner(Scanner):
    def __init__(self, azi_offset, elv_offset, alpha, delta, beta, epsilon):
        """General scanner model M_G(gamma, omega) = (phi, theta)
        This model assumes a scanner with pan-tilt mechanism and a dish.
        The parameters are:
            azi_offset: azimuth offset of the pan-tilt mechanism
            elv_offset: elevation offset of the pan-tilt mechanism
        """
        super().__init__()
        self.azi_offset = azi_offset
        self.elv_offset = elv_offset
        self.alpha = alpha
        self.delta = delta
        self.beta = beta
        self.epsilon = epsilon
        self.chain = generate_pt_chain(azi_offset, elv_offset, alpha, delta, beta, epsilon)
    
    def forward_pointing(self, gamma, omega):
        """Calculate the pointing of the radar, i.e. the direction of the z-axis of the last link in the chain.

        Returns:
            np.ndarray: Array of shape (N, 3) with the pointing direction vectors.
        """
        gamma= np.atleast_1d(gamma)
        omega= np.atleast_1d(omega)
        radar_pointing=[self.chain.forward_kinematics(_gam_om_to_joint_positions(a, e))[:3, 2] for a, e in zip(gamma, omega)]
        return np.array(radar_pointing)


    def forward(self, gamma, omega):
        azimuth=[]
        elevation=[]
        for a, e in zip(gamma, omega):
            trans_matrix=self.chain.forward_kinematics(_gam_om_to_joint_positions(a,e))[:3, :3]
            z_axis=trans_matrix[:, 2]
            x_axis=trans_matrix[:, 0]
            azi_rad, elv_rad = _vector_to_azielv(z_axis, x_axis)
            azimuth.append(azi_rad)
            elevation.append(elv_rad)
        return np.array(azimuth), np.array(elevation)
    

    def inverse(self, azi, elv):
        azi= np.atleast_1d(azi)
        elv= np.atleast_1d(elv)
        positions=[]
        for a,e in zip(azi, elv):
            # calculate the orientation vector
            unit_vector= spherical_to_cartesian(a,e)
            initial_guess=_gam_om_to_joint_positions(a, e)
            position=self.chain.inverse_kinematics(target_orientation=unit_vector, orientation_mode='Z', initial_position=initial_guess)
            position=_joint_positions_to_gam_om(position)
            positions.append(position)
        return np.array(positions)
    
    def get_params(self):
        """Get the parameters of the scanner as a dictionary."""
        return {
            'azi_offset': self.azi_offset,
            'elv_offset': self.elv_offset,
            'alpha': self.alpha,
            'delta': self.delta,
            'beta': self.beta,
            'epsilon': self.epsilon
        }
    
    def __repr__(self):
        return "General Scanner Model:\n" + \
               f"Azi Offset: {np.rad2deg(self.azi_offset):.2f} º\n" + \
               f"Elv Offset: {np.rad2deg(self.elv_offset):.2f} º\n" + \
               f"Alpha: {np.rad2deg(self.alpha):.2f} º\n" + \
               f"Delta: {np.rad2deg(self.delta):.2f} º\n" + \
               f"Beta: {np.rad2deg(self.beta):.2f} º\n" + \
               f"Epsilon: {np.rad2deg(self.epsilon):.2f} º"

    