import xarray as xr
import numpy as np
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from sunscan.math_utils import spherical_to_xyz, calc_azi_diff
from sunscan.params import SCANNER_PARAMETER_MAP
import warnings
import yaml

SCANNER_FILE_VERSION="1.0"

class Scanner(object):
    """Base class for scanner models."""
    def __init__(self):
        pass
    
    def forward(self, gamma, omega, gammav, omegav):
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
    def forward(self, gamma, omega, gammav=None, omegav=None):
        """Identity scanner model M_I(gamma, omega) = (phi, theta)
        This model assumes a perfectly oriented scanner.
        For omega <=90 degrees, it is the identity function: gamma=phi, omega=theta.
        """
        reverse = omega > 90
        azi = xr.where(reverse, (gamma+180) , gamma)
        elv = xr.where(reverse, 180 - omega, omega)
        return azi%360, elv
    
    def forward_pointing(self, gamma, omega, gammav=None, omegav=None):
        return spherical_to_xyz(*self.forward(gamma, omega))

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
    def __init__(self, gamma_offset, omega_offset, dtime, backlash_gamma, flex):
        self.gamma_offset= gamma_offset
        self.omega_offset= omega_offset
        self.dtime= dtime
        self.backlash_gamma= backlash_gamma
        self.flex = flex
        self.identity_scanner = IdentityScanner()
    
    def apply_offsets(self, gamma, omega, gammav, omegav):
        gamma_corr = gamma+self.gamma_offset
        gamma_corr = gamma_corr+self.backlash_gamma*np.sign(gammav)
        gamma_corr = gamma_corr + self.dtime * gammav
        omega_corr = omega+self.omega_offset
        omega_corr = omega_corr + self.dtime * omegav
        omega_corr = omega_corr + self.flex * np.cos(np.deg2rad(omega_corr))
        return np.round(gamma_corr, 12)%360, omega_corr
    
    def remove_offsets(self, gamma, omega, gammav, omegav):
        omega = omega - self.flex * np.cos(np.deg2rad(omega))
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
            'backlash_gamma': self.backlash_gamma,
            'flex': self.flex
        }
        if complete:
            params['alpha'] = 0.0
            params['delta'] = 0.0
            params['beta'] = 0.0
            params['epsilon'] = 0.0
        return params



#%% 2D pan tilt system
def generate_pt_chain(alpha, delta, beta, epsilon, omega_bounds=None):
    d=1
    alpha=np.deg2rad(alpha)
    delta=np.deg2rad(delta)
    beta=np.deg2rad(beta)
    epsilon=np.deg2rad(epsilon)
    if omega_bounds is not None:
        j2_bounds=(_gam_om_to_joint(-999, omega_bounds[0])[1], _gam_om_to_joint(-999, omega_bounds[1])[1])
    else:
        j2_bounds=None
    pt_chain= Chain(name='pan_tilt', links=[
        OriginLink(),
        URDFLink(
            name="pan",
            origin_translation=[0, 0, d/3],
            origin_orientation=[alpha, delta, 0],
            rotation=[0, 0, 1],
        ),
        URDFLink(
            name="tilt",
            origin_translation=[0,0,d],
            origin_orientation=[beta,0,  0],
            rotation=[0, -1, 0],
            bounds=j2_bounds
        ),
        URDFLink(
            name="dish",
            origin_translation=[0, 0, d],
            origin_orientation=[epsilon, 0, 0],
            rotation=[1, 0, 0],
        )
    ],
        active_links_mask=[False, True, True, False]
    )
    return pt_chain

def _vector_to_azielv(z_axis, x_axis, eps=1e-8):
    """
    Calculates azimuth and elevation from direction vectors.
    If any z axis is vertical (x and y near zero), uses the corresponding x axis for azimuth.
    
    Args:
        z_axis: array of shape (3, ...) with direction vectors
        x_axis: array of shape (3, ...) with x-axis vectors
        eps: tolerance for determining if z_axis is vertical
    
    Returns:
        azimuth (degrees), elevation (degrees) - arrays matching the input shape
    """
    z_axis = np.asarray(z_axis)
    x_axis = np.asarray(x_axis)
    
    # Calculate norms and elevation
    norms = np.linalg.norm(z_axis, axis=0)
    elv = np.arcsin(z_axis[2] / norms)
    
    # Check which vectors are vertical (x and y components near zero)
    is_vertical = (np.abs(z_axis[0]) < eps) & (np.abs(z_axis[1]) < eps)
    
    # Calculate azimuth from z_axis
    azi = np.arctan2(z_axis[1], z_axis[0])
    
    # For vertical vectors, use x_axis for azimuth
    azi_from_x = np.arctan2(x_axis[1], x_axis[0])
    azi = np.where(is_vertical, azi_from_x, azi)
    
    # Convert to degrees
    return np.rad2deg(azi), np.rad2deg(elv)

def _gam_om_to_joint(gamma, omega):
    j1=np.deg2rad(gamma)
    j2=np.deg2rad(omega-90)
    return j1, j2

def _joint_to_gam_om(j1, j2):
    """ Convert joint angles to gamma and omega in degrees """
    gamma = np.rad2deg(j1)
    omega = np.rad2deg(j2) + 90
    return gamma, omega

def _gam_om_to_joint_list(gamma, omega):
    """ gamma (azimuth) and omega (elevation) in degrees"""
    j1, j2= _gam_om_to_joint(gamma, omega)
    positions=np.zeros((len(gamma), 4)) # shape (N, 4)
    positions[:, 1] = j1
    positions[:, 2] = j2
    return positions

def _joint_list_to_gam_om(positions):
    """ positions in degrees"""
    j1,j2= positions[:, 1], positions[:, 2]
    gamma, omega= _joint_to_gam_om(j1, j2)
    return gamma, omega

class GeneralScanner(Scanner):
    def __init__(self, gamma_offset, omega_offset, alpha, delta, beta, epsilon, dtime, backlash_gamma, flex):
        """General scanner model M_G(gamma, omega) = (phi, theta)
        """
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.beta = beta
        self.epsilon = epsilon
        self.backlash_scanner= BacklashScanner(dtime=dtime, backlash_gamma=backlash_gamma, gamma_offset=gamma_offset, omega_offset=omega_offset, flex=flex)
        self.chain = generate_pt_chain(alpha, delta, beta, epsilon)

    @classmethod
    def load(cls, filename):
        """Load scanner parameters from a YAML file."""
        with open(filename, 'r') as f:
            params = yaml.safe_load(f)
        file_version = params.get('file_version', 'nan')
        if file_version != SCANNER_FILE_VERSION:
            warnings.warn(f"Scanner file version {file_version} does not match expected version {SCANNER_FILE_VERSION}.")
        return cls(gamma_offset=params['gamma_offset'],
                     omega_offset=params['omega_offset'],
                     alpha=params['alpha'],
                     delta=params['delta'],
                     beta=params['beta'],
                     epsilon=params['epsilon'],
                     dtime=params['dtime'],
                     backlash_gamma=params['backlash_gamma'],
                     flex=params['flex']
        )
    
    def _get_joint_positions(self, gamma, omega, gammav=0, omegav=0):
        gamma, omega = self.backlash_scanner.apply_offsets(gamma, omega, gammav, omegav)
        positions=_gam_om_to_joint_list(gamma, omega)
        return positions
    
    def _get_gamma_omega(self, positions, gammav=0, omegav=0):
        gamma, omega= _joint_list_to_gam_om(positions)
        gamma, omega = self.backlash_scanner.remove_offsets(gamma, omega, gammav, omegav)
        return gamma, omega
    
    def forward_unitvectors(self, gamma, omega, gammav=0, omegav=0):
        """Calculate the unit vectors of the system at the end of the chain."""
        gamma= np.atleast_1d(gamma)
        omega= np.atleast_1d(omega)
        positions= self._get_joint_positions(gamma, omega, gammav, omegav)
        radar_pointing=np.array([self.chain.forward_kinematics(pos)[:3, :3] for pos in positions]) # shape (N, 3, 3), where the last axis containes the x,y,z unit vectors
        # move N to the last axis
        radar_pointing=np.moveaxis(radar_pointing, 0, -1)
        x_axis=radar_pointing[:, 0, :]
        y_axis=radar_pointing[:, 1, :]
        z_axis=radar_pointing[:, 2, :]

        # remove singleton dimensions
        if x_axis.shape[1] == 1:
            x_axis=x_axis[:, 0]
            y_axis=y_axis[:, 0]
            z_axis=z_axis[:, 0]
        return x_axis, y_axis, z_axis

    def forward_pointing(self, gamma, omega, gammav=0, omegav=0):
        """Calculate the pointing of the radar, i.e. the direction of the z-axis of the last link in the chain.

        Returns:
            np.ndarray: Array of shape (N, 3) with the pointing direction vectors.
        """
        ex, ey, ez = self.forward_unitvectors(gamma, omega, gammav, omegav)
        return ez
    

    def forward(self, gamma, omega, gammav=0, omegav=0):
        x_axis, y_axis, z_axis = self.forward_unitvectors(gamma, omega, gammav, omegav)
        azi, elv= _vector_to_azielv(z_axis, x_axis)
        return azi%360, elv
    
    def _create_bounded_chain_copy(self, omega_bounds):
        chain=generate_pt_chain(self.alpha, self.delta, self.beta, self.epsilon, omega_bounds=omega_bounds)
        return chain

    def inverse(self, azi, elv, gammav=0, omegav=0, reverse=None):
        azi= np.atleast_1d(azi)
        elv= np.atleast_1d(elv)
        if len(azi) != len(elv):
            raise ValueError("azi and elv must have the same length.")
        if azi.ndim != 1 or elv.ndim != 1:
            raise ValueError("azi and elv must be 1D arrays or scalars.")
        if reverse is None:
            chain = self.chain
        else:
            if reverse == False:
                omega_bounds_offset=np.array([-20, 90])
            else:
                omega_bounds_offset=np.array([90, 200])
            _, omega_bounds = self.backlash_scanner.remove_offsets(0, omega_bounds_offset, 0,0)
            chain = self._create_bounded_chain_copy(omega_bounds=omega_bounds)
        reverse_guess = False if reverse is None else reverse
        positions=[]
        target_vectors= np.array(spherical_to_xyz(azi,elv)).T # shape (3, N) -> (N, 3) after transpose
        gamma_guess, omega_guess = self.backlash_scanner.inverse(azi, elv, gammav=gammav, omegav=omegav, reverse=reverse_guess)
        initial_positions=_gam_om_to_joint_list(gamma_guess, omega_guess)
        for initial_pos, target_vector in zip(initial_positions, target_vectors):
            # calculate the orientation vector
            pos=chain.inverse_kinematics(target_orientation=target_vector, orientation_mode='Z', initial_position=initial_pos)
            positions.append(pos)
        gamma, omega=self._get_gamma_omega(np.array(positions), gammav, omegav)
        if len(gamma) == 1:
            gamma, omega = gamma[0], omega[0]
        # check the quality of the inversion
        azi_check, elv_check = self.forward(gamma, omega, gammav, omegav)
        azi_diff= calc_azi_diff(azi_check, azi)
        if not np.allclose(azi_diff, 0, atol=0.1) or not np.allclose(elv_check, elv, atol=0.1):
            warnings.warn(f"Inversion imperfect: Results deviate from target by up to {np.max(np.abs(azi_check-azi)):.2f} degrees in azimuth and {np.max(np.abs(elv_check-elv)):.2f} degrees in elevation.")
        return gamma, omega
    
    def get_params(self, complete=False):
        """Get the parameters of the scanner as a dictionary."""
        backlash_params = self.backlash_scanner.get_params()
        params = {
            'gamma_offset': self.backlash_scanner.gamma_offset,
            'omega_offset': self.backlash_scanner.omega_offset,
            'alpha': self.alpha,
            'delta': self.delta,
            'beta': self.beta,
            'epsilon': self.epsilon,
            'dtime': backlash_params['dtime'],
            'backlash_gamma': backlash_params['backlash_gamma'],
            'flex': backlash_params['flex']
        }
        params={k:float(v) for k,v in params.items()}
        return params
    
    def __repr__(self):
        return "General Scanner Model:\n" + \
               f"Azimuth Offset: {self.backlash_scanner.gamma_offset:.4f} º\n" + \
               f"Elevation Offset: {self.backlash_scanner.omega_offset:.4f} º\n" + \
               f"Alpha: {self.alpha:.4f} º\n" + \
               f"Delta: {self.delta:.4f} º\n" + \
               f"Beta: {self.beta:.4f} º\n" + \
               f"Epsilon: {self.epsilon:.4f} º\n" + \
               f"Time Offset: {self.backlash_scanner.dtime:.4f} s\n" + \
               f"Azimuth Backlash: {self.backlash_scanner.backlash_gamma:.4f} º\n" + \
               f"Flex: {self.backlash_scanner.flex:.4f} º at 0 elevation"
    
    def save(self, filename):
        """Save the scanner parameters to a YAML file."""
        params = self.get_params(complete=True)
        params['file_version'] = SCANNER_FILE_VERSION
        with open(filename, 'w') as f:
            yaml.dump(params, f)

    