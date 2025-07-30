import numpy as np
import scipy.optimize as opt
from sunscan import sc_params, logger
from sunscan.fit_utils import get_parameter_lists, optimize_brute_force, rmse
from sunscan.scanner import GeneralScanner
from sunscan.utils import spherical_to_cartesian


PARAMETER_MAP = {
    'azi_offset': 0,
    'elv_offset': 1,
    'alpha': 2,
    'delta': 3,
    'beta': 4,
    'epsilon': 5,
}

def difference_angles(vec1, vec2):
    """
    Calculate the difference in angles between two vectors.
    Returns the angle in radians.
    """
    dot_product = (vec1 * vec2).sum(axis=-1)
    dot_product=np.clip(dot_product, -1.0, 1.0)
    angle = np.arccos(dot_product)
    return angle

def forward_model(gamma, omega, params):
    """For a given set of parameters and azimuth/elevation angles, calculate the pointing direction of the radar.
    Args:
        azi (float or np.ndarray): Azimuth angle(s) in radians.
        elv (float or np.ndarray): Elevation angle(s) in radians.
        params (dict): Parameters for the scanner model
    
    Returns:
        np.ndarray: Array of shape (N, 3) with the pointing direction vectors.
    """
    scanner=GeneralScanner(
        azi_offset=params.get("azi_offset", 0.0),
        elv_offset=params.get("elv_offset", 0.0),
        alpha=params.get("alpha", 0.0),
        delta=params.get("delta", 0.0),
        beta=params.get("beta", 0.0),
        epsilon=params.get("epsilon", 0.0),
    )
    radar_pointing = scanner.forward_pointing(gamma, omega)
    return radar_pointing


def objective(params_dict, gamma, omega, target_vectors, return_vectors=False):
    """For a given set of parameters, azimuth and elevation angles, evaluate the forward model and calculate objective values wrt. the target vectors.
    
    Returns:
        np.ndarray: Array of shape (N,) with the objective values (difference angles).
    """
    turned=forward_model(gamma, omega, params_dict)
    angles= difference_angles(turned, target_vectors)
    if return_vectors:
        return angles, turned
    return angles

def optimize_function(params_list, gamma, omega, target_vectors):
    params_dict= {k:params_list[v] for k,v in PARAMETER_MAP.items()}
    return rmse(objective(params_dict, gamma, omega, target_vectors))


class ScannerEstimator(object):
    def __init__(self, params_optimize=None, params_guess=None, params_bounds=None):
        if params_optimize is None:
            params_optimize = sc_params['scanner_params_optimize']
        if params_guess is None:
            params_guess = sc_params['scanner_params_guess']
        if params_bounds is None:
            params_bounds = sc_params['scanner_params_bounds']
        self.params_optimize = params_optimize
        self.params_guess = params_guess
        self.params_bounds = params_bounds
    
    def fit(self, gamma, omega, azi_b, elv_b, brute_force=True, brute_force_points=3):
        logger.info('Starting optimization')
        init_guess_rad= {k:np.deg2rad(v) for k,v in self.params_guess.items()}
        parameter_bounds_rad= {k:(np.deg2rad(v[0]), np.deg2rad(v[1])) for k,v in self.params_bounds.items()}  
        init_guess_list, bounds_list= get_parameter_lists(self.params_optimize, init_guess_rad, parameter_bounds_rad, PARAMETER_MAP)
        pointing_b=spherical_to_cartesian(azi_b, elv_b)
        #%%
        if brute_force
            logger.info(f"Brute force optimization enabled with {brute_force_points} points ({brute_force_points**len(self.params_optimize)} total)")
            brute_force_params, brute_force_rmse = optimize_brute_force(bounds_list, optimize_function, optimize_args=(gamma, omega, pointing_b), points=brute_force_points)
            logger.info(f"Best Parameters: {np.rad2deg(brute_force_params)}")
            logger.info(f"Best RMSE: {brute_force_rmse:.6f}")
            init_rmse=optimize_function(init_guess_list, gamma, omega, pointing_b)
            if init_rmse > brute_force_rmse:
                logger.info(f"Brute force did improve the initial guess from {init_rmse:.6f} to {brute_force_rmse:.6f}")
                init_guess_list = brute_force_params
        #%%
        opt_res = opt.minimize(
            fun=optimize_function,
            x0=init_guess_list,
            method="Nelder-Mead",
            args=(gamma, omega, pointing_b),
            bounds=bounds_list,
        )
        fit_result_list=opt_res.x
        fit_result_dict={k:fit_result_list[v] for k,v in PARAMETER_MAP.items()}
        fitted_scanner= GeneralScanner(**fit_result_dict)
        return fitted_scanner

