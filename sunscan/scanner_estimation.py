import numpy as np
import scipy.optimize as opt
from sunscan.utils import logger
from sunscan.fit_utils import get_parameter_lists, optimize_brute_force
from sunscan.scanner import GeneralScanner, IdentityScanner
from sunscan.math_utils import spherical_to_xyz, rmse, difference_angles
from sunscan.utils import guess_offsets
from sunscan.params import SCANNER_PARAMETER_MAP, sc_params


identity_scanner = IdentityScanner()

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
    params_dict= {k:params_list[v] for k,v in SCANNER_PARAMETER_MAP.items()}
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
    
    def _fit(self, gamma, omega, azi_b, elv_b, params_optimize, params_guess, params_bounds, brute_force=True, brute_force_points=3):
        # gamma_offset and omega_offset can be None, in which case they are determined dynamically based on the mean difference between the scanner coordinates and beam position
        if params_guess['gamma_offset'] is None or params_guess['omega_offset'] is None:
            gamoff_guess, omoff_guess= guess_offsets(gamma, omega, azi_b, elv_b)
            logger.info(f"Estimated gamma_offset: {gamoff_guess:.4f}, omega_offset: {omoff_guess:.4f}")
            if params_guess['gamma_offset'] is None:
                params_guess['gamma_offset'] = gamoff_guess
            if params_guess['omega_offset'] is None:
                params_guess['omega_offset'] = omoff_guess
        logger.info(f'Starting to optimize {", ".join(params_optimize)} using {len(gamma)} calibration pairs')
        params_guess_list, bounds_list= get_parameter_lists(params_optimize, params_guess, params_bounds, SCANNER_PARAMETER_MAP)
        pointing_b=np.array(spherical_to_xyz(azi_b, elv_b))
        optimize_args = (gamma, omega, pointing_b)
        #%%
        if brute_force:
            logger.info(f"Brute force optimization enabled with {brute_force_points} points ({brute_force_points**len(params_optimize)} total)")
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
        fit_result_dict={k:fit_result_list[v] for k,v in SCANNER_PARAMETER_MAP.items()}
        logger.info("Optimization Result:\n" + '\n'.join([f"{k}: {v:.4f}" for k, v in fit_result_dict.items()]))
        init_rmse = optimize_function(params_guess_list, *optimize_args)
        logger.info(f"Initial objective: {init_rmse:.6f}")
        logger.info(f"Optimal objective: {opt_res.fun:.6f}")
        fitted_scanner= GeneralScanner(**fit_result_dict, dtime=self.dtime, backlash_gamma=self.backlash_gamma)
        return fitted_scanner, opt_res.fun

    def fit_global(self, gamma, omega, azi_b, elv_b, brute_force=True, brute_force_points=3):
        params_guess = self.params_guess.copy()
        params_bounds = self.params_bounds.copy()
        params_optimize = self.params_optimize.copy()
        return self._fit(gamma, omega, azi_b, elv_b, params_optimize, params_guess, params_bounds, brute_force=brute_force, brute_force_points=brute_force_points)
    
    
    def fit_sequential(self, gamma, omega, azi_b, elv_b, brute_force=True):
        # Step 1: Fit gamma_offset and epsilon close to the horizon
        is_horizon = np.logical_or(elv_b<20, elv_b>160)
        if not np.any(is_horizon):
            raise ValueError("No points close to the horizon found in the data. Sequential fitting not possible")
        gamma_step1, omega_step1 = gamma[is_horizon], omega[is_horizon]
        azi_b_step1, elv_b_step1 = azi_b[is_horizon], elv_b[is_horizon]
        params_optimize_step1=['gamma_offset', 'epsilon']
        params_guess_step1=self.params_guess.copy()
        params_bounds_step1=self.params_bounds.copy()
        logger.info(f"Step 1")
        scanner_step1, rmse_step1 = self._fit(
            gamma_step1, omega_step1, azi_b_step1, elv_b_step1,
            params_optimize=params_optimize_step1,
            params_guess=params_guess_step1,
            params_bounds=params_bounds_step1,
            brute_force=brute_force, brute_force_points=10
        )
        # Step 2: optimize omega_offset, alpha in the west / east
        is_west_east = np.logical_or(np.abs(azi_b-270)<20, np.abs(azi_b-90)<20)
        if not np.any(is_west_east):
            raise ValueError("No points in the west/east found in the data. Sequential fitting not possible")
        gamma_step2, omega_step2 = gamma[is_west_east], omega[is_west_east]
        azi_b_step2, elv_b_step2 = azi_b[is_west_east], elv_b[is_west_east]
        params_optimize_step2=['omega_offset', 'alpha']
        params_guess_step2=scanner_step1.get_params()
        params_bounds_step2=self.params_bounds.copy()
        logger.info(f"Step 2")
        scanner_step2, rmse_step2 = self._fit(
            gamma_step2, omega_step2, azi_b_step2, elv_b_step2,
            params_optimize=params_optimize_step2,
            params_guess=params_guess_step2,
            params_bounds=params_bounds_step2,
            brute_force=brute_force, brute_force_points=10
        )
        # Step 3: Take points in the south high up in the sky to fit delta (causing an elevation offet) and beta (causing an azimuth (mostly) and elevation offset)
        is_north_south = np.logical_or(np.abs(azi_b-180)<20, np.abs(azi_b-0)<20)
        if not np.any(is_north_south):
            raise ValueError("No points in the north/south found in the data. Sequential fitting not possible")
        gamma_step3, omega_step3 = gamma[is_north_south], omega[is_north_south]
        azi_b_step3, elv_b_step3 = azi_b[is_north_south], elv_b[is_north_south]
        params_optimize_step3=['delta', 'beta']
        params_guess_step3=scanner_step2.get_params()
        params_bounds_step3=self.params_bounds.copy()
        logger.info(f"Step 3")
        scanner_step3, rmse_step3 = self._fit(
            gamma_step3, omega_step3, azi_b_step3, elv_b_step3,
            params_optimize=params_optimize_step3,
            params_guess=params_guess_step3,
            params_bounds=params_bounds_step3,
            brute_force=brute_force, brute_force_points=10
        )
        # Final step 4: Fit all parameters again to the full sky
        gamma_step4, omega_step4 = gamma, omega
        azi_b_step4, elv_b_step4 = azi_b, elv_b
        params_optimize_step4=self.params_optimize.copy()
        params_guess_step4=scanner_step3.get_params()
        params_bounds_step4=self.params_bounds.copy()
        logger.info(f"Step 4")
        scanner_step4, rmse_step4 = self._fit(
            gamma_step4, omega_step4, azi_b_step4, elv_b_step4,
            params_optimize=params_optimize_step4,
            params_guess=params_guess_step4,
            params_bounds=params_bounds_step4,
            brute_force=False
        )
        return scanner_step4, rmse_step4
