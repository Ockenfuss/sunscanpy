import numpy as np
import scipy.optimize as opt
from sunscan import sc_params, logger
from sunscan.fit_utils import get_parameter_lists, optimize_brute_force, rmse
from sunscan.scanner import GeneralScanner, IdentityScanner
from sunscan.math_utils import spherical_to_cartesian
from sunscan.utils import guess_offsets


PARAMETER_MAP = {
    'gamma_offset': 0,
    'omega_offset': 1,
    'alpha': 2,
    'delta': 3,
    'beta': 4,
    'epsilon': 5,
}
identity_scanner = IdentityScanner()

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
        params (dict): Parameters for the scanner model
    
    Returns:
        np.ndarray: Array of shape (N, 3) with the pointing direction vectors.
    """
    scanner=GeneralScanner(
        **params,
        dtime=0.0, # when estimating, we assume the (gamma, omega) (azi,elv) pairs are calculated stationary, i.e. neither the time offset nor the backlash are relevant
        backlash_gamma=0.0
    )
    radar_pointing = scanner.forward_pointing(gamma, omega, gammav=0, omegav=0) #assume stationary pairs.
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
    def __init__(self, params_optimize=None, params_guess=None, params_bounds=None, dtime=0, backlash_gamma=0):
        """
        """
        if params_optimize is None:
            params_optimize = sc_params['scanner_params_optimize']
        if params_guess is None:
            params_guess = sc_params['scanner_params_guess']
        if params_bounds is None:
            params_bounds = sc_params['scanner_params_bounds']
        self.params_optimize = params_optimize
        self.params_guess = params_guess
        self.params_bounds = params_bounds
        self.dtime= dtime #dtime and backlash are not used in the estimation (assumption of stationary calibration pairs). They are only added after fitting, when the scanner is returned.
        self.backlash_gamma= backlash_gamma
    
    def fit(self, gamma, omega, azi_b, elv_b, brute_force=True, brute_force_points=3):

        # gamma_offset and omega_offset can be None, in which case they are determined dynamically based on the mean difference between the scanner coordinates and beam position
        params_guess = self.params_guess.copy()
        params_bounds = self.params_bounds.copy()
        if params_guess['gamma_offset'] is None or params_guess['omega_offset'] is None:
            gamoff_guess, omoff_guess= guess_offsets(gamma, omega, azi_b, elv_b)
            logger.info(f"Estimated gamma_offset: {gamoff_guess:.4f}, omega_offset: {omoff_guess:.4f}")
            if params_guess['gamma_offset'] is None:
                params_guess['gamma_offset'] = gamoff_guess
                params_bounds['gamma_offset'] = (gamoff_guess+params_bounds['gamma_offset'][0], gamoff_guess+params_bounds['gamma_offset'][1]) # in case the guess for go is determined dynamically, the bounds are interpreted as relative to the guess
            if params_guess['omega_offset'] is None:
                params_guess['omega_offset'] = omoff_guess
                params_bounds['omega_offset'] = (omoff_guess+params_bounds['omega_offset'][0], omoff_guess+params_bounds['omega_offset'][1]) 
        logger.info('Starting optimization')
        params_guess_list, bounds_list= get_parameter_lists(self.params_optimize, params_guess, params_bounds, PARAMETER_MAP)
        pointing_b=spherical_to_cartesian(azi_b, elv_b)
        optimize_args = (gamma, omega, pointing_b)
        #%%
        if brute_force:
            logger.info(f"Brute force optimization enabled with {brute_force_points} points ({brute_force_points**len(self.params_optimize)} total)")
            brute_force_params, brute_force_rmse = optimize_brute_force(bounds_list, optimize_function, optimize_args=optimize_args, points=brute_force_points)
            logger.info(f"Best Parameters: " + ", ".join([f"{v:.4f}" for v in brute_force_params]))
            logger.info(f"Best RMSE: {brute_force_rmse:.6f}")
            init_rmse=optimize_function(params_guess_list, gamma, omega, pointing_b)
            if init_rmse > brute_force_rmse:
                logger.info(f"Brute force did improve the initial guess from {init_rmse:.6f} to {brute_force_rmse:.6f}")
                params_guess_list = brute_force_params
        #%%
        opt_res = opt.minimize(
            fun=optimize_function,
            x0=params_guess_list,
            method="Nelder-Mead",
            args=optimize_args,
            bounds=bounds_list,
        )
        fit_result_list=opt_res.x
        fit_result_dict={k:fit_result_list[v] for k,v in PARAMETER_MAP.items()}
        logger.info("Optimization Result:\n" + '\n'.join([f"{k}: {v:.4f}" for k, v in fit_result_dict.items()]))
        init_rmse = optimize_function(params_guess_list, *optimize_args)
        logger.info(f"Initial objective: {init_rmse:.6f}")
        logger.info(f"Optimal objective: {opt_res.fun:.6f}")
        fitted_scanner= GeneralScanner(**fit_result_dict, dtime=self.dtime, backlash_gamma=self.backlash_gamma)
        return fitted_scanner, opt_res.fun

