import xarray as xr
import numpy as np
import ikpy
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from sunscan.math_utils import spherical_to_cartesian
from sunscan.params import SCANNER_PARAMETER_MAP




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
    
    def get_params(self, complete=False):
        """Get the parameters of the scanner as a dictionary.

        If complete is True, return all parameters, otherwise only the ones that are actually used in this scanner model.
        """
        raise NotImplementedError("Please implement the get_params method in the subclass.")

class IdentityScanner(Scanner):
    def forward(self, gamma, omega):
        """Identity scanner model M_I(gamma, omega) = (phi, theta)
        This model assumes a perfectly oriented scanner.
        For omega <=90 degrees, it is the identity function: gamma=phi, omega=theta.
        """
        reverse = omega > 90
        azi = xr.where(reverse, (gamma+180) , gamma)
        elv = xr.where(reverse, 180 - omega, omega)
        return azi%360, elv
    
    def forward_pointing(self, gamma, omega):
        return spherical_to_cartesian(*self.forward(gamma, omega))

    def inverse(self, azi, elv, reverse=False):
        """Invert the identity radar model.

        Since the identity model is bijective, you need to specify whether azi,elv should be mapped in the forward or 
        reverse part of the gamma-omega space.

        """
        gamma=xr.where(reverse, (azi - 180) , azi)
        omega=xr.where(reverse, 180 - elv, elv)
        return gamma%360, omega
    
    def get_params(self, complete=False):
        params={}
        if complete:
            params['gamma_offset'] = 0.0
            params['omega_offset'] = 0.0
            params['alpha'] = 0.0
            params['delta'] = 0.0
            params['beta'] = 0.0
            params['epsilon'] = 0.0
            params['dtime'] = 0.0
            params['backlash_gamma'] = 0.0
        return params

    
class BacklashScanner(Scanner):
    """Identity Scanner model, but with global offsets and backlash correction.
    
       A time offset will be applied to both axes and create an angle offset depending on the speed of movement.
       A backlash correction is applied individually to the axes and only depends on the direction of movement.
    """
    def __init__(self, gamma_offset, omega_offset, dtime, backlash_gamma):
        self.gamma_offset= gamma_offset
        self.omega_offset= omega_offset
        self.dtime= dtime
        self.backlash_gamma= backlash_gamma
        self.identity_scanner = IdentityScanner()
    
    def apply_offsets(self, gamma, omega, gammav, omegav):
        gamma_corr = gamma+self.gamma_offset
        gamma_corr = gamma_corr+self.backlash_gamma*np.sign(gammav)
        gamma_corr = gamma_corr + self.dtime * gammav
        omega_corr = omega+self.omega_offset
        omega_corr = omega_corr + self.dtime * omegav
        return np.round(gamma_corr, 12)%360, omega_corr
    
    def remove_offsets(self, gamma, omega, gammav, omegav):
        gamma = gamma - self.gamma_offset
        gamma = gamma - self.backlash_gamma * np.sign(gammav)
        gamma = gamma - self.dtime * gammav
        omega = omega - self.omega_offset
        omega = omega - self.dtime * omegav
        return np.round(gamma, 12)%360.0, omega

    
    def forward(self, gamma, omega, gammav, omegav):
        gamma_corr, omega_corr = self.apply_offsets(gamma, omega, gammav, omegav)
        azi, elv = self.identity_scanner.forward(gamma_corr, omega_corr)
        return azi, elv
    
    def inverse(self, azi, elv, gammav, omegav, reverse=False):
        gamma, omega = self.identity_scanner.inverse(azi, elv, reverse=reverse)
        gamma, omega = self.remove_offsets(gamma, omega, gammav, omegav)
        return gamma, omega
    
    def get_params(self, complete=False):
        """Get the parameters of the scanner as a dictionary."""
        params= {
            'gamma_offset': self.gamma_offset,
            'omega_offset': self.omega_offset,
            'dtime': self.dtime,
            'backlash_gamma': self.backlash_gamma
        }
        if complete:
            params['alpha'] = 0.0
            params['delta'] = 0.0
            params['beta'] = 0.0
            params['epsilon'] = 0.0
        return params



#%% 2D pan tilt system
def generate_pt_chain(gamma_offset, omega_offset, alpha, delta, beta, epsilon):
    d=1
    gamma_offset=np.deg2rad(gamma_offset)
    omega_offset=np.deg2rad(omega_offset)
    alpha=np.deg2rad(alpha)
    delta=np.deg2rad(delta)
    beta=np.deg2rad(beta)
    epsilon=np.deg2rad(epsilon)
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
            origin_orientation=[0, 0, gamma_offset],
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
            origin_orientation=[0, np.pi/2-omega_offset, 0],
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
    Returns azimuth (degrees), elevation (degrees)
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
    return np.rad2deg(azi), np.rad2deg(elv)

def _gam_om_to_joint_positions(gamma, omega):
    """ gamma (azimuth) and omega (elevation) in degrees"""
    return [0, np.deg2rad(gamma), 0, np.deg2rad(omega), 0, 0]

def _joint_positions_to_gam_om(positions):
    """ positions in degrees"""
    gamma = np.rad2deg(positions[1])
    omega = np.rad2deg(positions[3])
    return gamma, omega

class GeneralScanner(Scanner):
    def __init__(self, gamma_offset, omega_offset, alpha, delta, beta, epsilon, dtime, backlash_gamma):
        """General scanner model M_G(gamma, omega) = (phi, theta)
        This model assumes a scanner with pan-tilt mechanism and a dish.
        """
        super().__init__()
        self.gamma_offset = gamma_offset
        self.omega_offset = omega_offset
        self.alpha = alpha
        self.delta = delta
        self.beta = beta
        self.epsilon = epsilon
        self.backlash_scanner= BacklashScanner(dtime=dtime, backlash_gamma=backlash_gamma, gamma_offset=0, omega_offset=0) #the constant offsets are handled by the chain
        self.chain = generate_pt_chain(gamma_offset, omega_offset, alpha, delta, beta, epsilon)
    
    def forward_pointing(self, gamma, omega, gammav=0, omegav=0):
        """Calculate the pointing of the radar, i.e. the direction of the z-axis of the last link in the chain.

        Returns:
            np.ndarray: Array of shape (N, 3) with the pointing direction vectors.
        """
        gamma= np.atleast_1d(gamma)
        omega= np.atleast_1d(omega)
        gamma, omega = self.backlash_scanner.apply_offsets(gamma, omega, gammav, omegav)
        radar_pointing=[self.chain.forward_kinematics(_gam_om_to_joint_positions(g, o))[:3, 2] for g, o in zip(gamma, omega)]
        return np.array(radar_pointing)


    def forward(self, gamma, omega, gammav=0, omegav=0):
        azi_list=[]
        elv_list=[]
        gamma, omega = self.backlash_scanner.apply_offsets(gamma, omega, gammav, omegav)
        for g, o in zip(gamma, omega):
            trans_matrix=self.chain.forward_kinematics(_gam_om_to_joint_positions(g,o))[:3, :3]
            z_axis=trans_matrix[:, 2]
            x_axis=trans_matrix[:, 0]
            azi, elv = _vector_to_azielv(z_axis, x_axis)
            azi_list.append(azi)
            elv_list.append(elv)
        return np.array(azi_list), np.array(elv_list)
    

    def inverse(self, azi, elv, gammav=0, omegav=0):
        azi= np.atleast_1d(azi)
        elv= np.atleast_1d(elv)
        gamma, omega=[],[]
        for a,e in zip(azi, elv):
            # calculate the orientation vector
            unit_vector= spherical_to_cartesian(a,e)
            initial_guess=_gam_om_to_joint_positions(a, e)
            position=self.chain.inverse_kinematics(target_orientation=unit_vector, orientation_mode='Z', initial_position=initial_guess)
            g,o=_joint_positions_to_gam_om(position)
            gamma.append(g)
            omega.append(o)
        gamma, omega = np.array(gamma), np.array(omega)
        gamma, omega = self.backlash_scanner.remove_offsets(gamma, omega, gammav, omegav)
        return gamma, omega
    
    def get_params(self, complete=False):
        """Get the parameters of the scanner as a dictionary."""
        backlash_params = self.backlash_scanner.get_params()
        return {
            'gamma_offset': self.gamma_offset,
            'omega_offset': self.omega_offset,
            'alpha': self.alpha,
            'delta': self.delta,
            'beta': self.beta,
            'epsilon': self.epsilon,
            'dtime': backlash_params['dtime'],
            'backlash_gamma': backlash_params['backlash_gamma'],
        }
    
    def __repr__(self):
        return "General Scanner Model:\n" + \
               f"Azi Offset: {self.gamma_offset:.2f} º\n" + \
               f"Elv Offset: {self.omega_offset:.2f} º\n" + \
               f"Alpha: {self.alpha:.2f} º\n" + \
               f"Delta: {self.delta:.2f} º\n" + \
               f"Beta: {self.beta:.2f} º\n" + \
               f"Epsilon: {self.epsilon:.2f} º\n" + \
               f"Time Offset: {self.backlash_scanner.dtime:.2f} s\n" + \
               f"Backlash: {self.backlash_scanner.backlash_gamma:.2f} º"

    