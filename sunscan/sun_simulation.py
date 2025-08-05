"""Module for simulating sun scans and fitting the simulation to real data."""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import convolve2d

from sunscan.utils import logger
from sunscan.math_utils import spherical_to_xyz, rmse
from sunscan.scanner import IdentityScanner, BacklashScanner
from sunscan.fit_utils import get_parameter_lists, optimize_brute_force
from sunscan.utils import guess_offsets, format_input_xarray
from sunscan.params import SUNSIM_PARAMETER_MAP, sc_params

identity_scanner = IdentityScanner()


def calculate_lut(dgamma_range=None, domega_range=None, resolution=401, fwhm_x=None, fwhm_y=None, limb_darkening=None):
    if dgamma_range is None:
        dgamma_range = sc_params['lut_dgamma_range']
    if domega_range is None:
        domega_range = sc_params['lut_domega_range']
    if fwhm_x is None:
        fwhm_x = sc_params['lut_fwhm_x']
    if fwhm_y is None:
        fwhm_y = sc_params['lut_fwhm_y']
    if limb_darkening is None:
        limb_darkening = sc_params['lut_limb_darkening']
    
    logger.info('Calculating new lookup table...')
    s_to_l_conversion = 1  # this way, units in the tangent space are essentially in degrees
    lookup_range_dl_gamma = dgamma_range*s_to_l_conversion
    lookup_range_dl_omega = domega_range*s_to_l_conversion
    lx = xr.DataArray(np.linspace(-lookup_range_dl_gamma, lookup_range_dl_gamma, resolution), dims='lx')
    lx.coords['lx'] = lx
    ly = xr.DataArray(np.linspace(-lookup_range_dl_omega, lookup_range_dl_omega, resolution), dims='ly')
    ly.coords['ly'] = ly
    # I found that non-zero centered luts can create an offset in contour plots
    assert 0 in lx.values
    assert 0 in ly.values

    limb_darkening = xr.DataArray(limb_darkening, dims='limb_darkening')
    limb_darkening.coords['limb_darkening'] = limb_darkening
    sun_rl = 0.5/2*s_to_l_conversion  # opening angle of the sun
    sundist = np.sqrt(lx**2+ly**2)
    sun = 1.0-(1.0-limb_darkening)*sundist/sun_rl
    sun = xr.where(sundist < sun_rl, sun, np.nan)
    sun = sun.dropna(dim='lx', how='all').dropna(dim='ly', how='all').fillna(0).rename(lx='sx', ly='sy')
    # Gaussian beam
    # FWHM = Full width half maximum
    # FWHM = 2.355 * sigma
    sigma_to_fwhm = 2*np.sqrt(2*np.log(2))
    fwhm_x = xr.DataArray(fwhm_x, dims='fwhm_x')
    fwhm_x.coords['fwhm_x'] = fwhm_x
    sigma_x = fwhm_x*s_to_l_conversion/sigma_to_fwhm
    fwhm_y = xr.DataArray(fwhm_y, dims='fwhm_y')
    fwhm_y.coords['fwhm_y'] = fwhm_y
    sigma_y = fwhm_y*s_to_l_conversion/sigma_to_fwhm
    gaussian = np.exp(-lx**2/(2*sigma_x**2)-ly**2/(2*sigma_y**2))
    # convolve sun and beam
    def convolve_2d(arr, sun):
        return convolve2d(arr, sun, mode='same', boundary='wrap')
    lut = xr.apply_ufunc(convolve_2d, gaussian, sun, input_core_dims=[['lx', 'ly'], [
                         'sx', 'sy']], output_core_dims=[['lx', 'ly']], vectorize=True)
    # normalize
    lut = (lut-lut.min(('lx', 'ly')))/(lut.max(('lx', 'ly'))-lut.min(('lx', 'ly')))
    logger.info('Lookup table size: %.2f GB', lut.nbytes/1024**3)
    return lut



# def _cart_to_tangential_matrix(anchor_azi, anchor_elv):
#     """Matrix to convert from cartesian coordinates to cartesian coordinates in the tangential plane, 
#     anchored at the given position on the unit sphere."""
#     anchor_phi, anchor_theta = _azi_elv_to_theta_phi(np.deg2rad(anchor_azi), np.deg2rad(anchor_elv))
#     loc_ze = xr.concat(_phitheta_to_cartesian(anchor_phi, anchor_theta), dim='row')
#     # y: co-elevation axis
#     anc_theta_y = np.pi/2-anchor_theta
#     anc_phi_y = (anchor_phi+np.pi) % (2*np.pi)
#     loc_ye = xr.concat(_phitheta_to_cartesian(anc_phi_y, anc_theta_y), dim='row')
#     # x: cross-elevation axis
#     anc_phi_x = (anchor_phi-np.pi/2) % (2*np.pi)
#     anc_theta_x = 0*anchor_theta+np.pi/2
#     loc_xe = xr.concat(_phitheta_to_cartesian(anc_phi_x, anc_theta_x), dim='row')

#     conv_loc_to_cart = xr.concat([loc_xe, loc_ye, loc_ze], dim='col')
#     # invert
#     conv_cart_to_loc = xr.apply_ufunc(np.linalg.inv, conv_loc_to_cart, input_core_dims=[
#                                       ['col', 'row']], output_core_dims=[['col', 'row']])
#     return conv_cart_to_loc

def _cart_to_tangential_matrix(anchor_azi, anchor_elv):
    """Matrix to convert from cartesian world coordinates to cartesian coordinates in the tangential plane, 
    anchored at the given position on the unit sphere."""
    anchor_azi = format_input_xarray(anchor_azi)
    anchor_elv = format_input_xarray(anchor_elv)
    loc_ze = xr.concat(spherical_to_xyz(anchor_azi, anchor_elv), dim='row')
    world_ze= xr.zeros_like(loc_ze)
    world_ze[{'row': 2}] = 1.0
    # x: cross-elevation axis
    loc_xe=xr.cross(world_ze, loc_ze, dim='row')
    # normalize
    loc_xe = loc_xe / xr.apply_ufunc(np.linalg.norm, loc_xe, input_core_dims=[['row']], kwargs={'axis': -1})
    # y: co-elevation axis
    loc_ye=xr.cross(loc_ze, loc_xe, dim='row')
    # stacking those vectors in columns would give the local to world transformation matrix
    # stacking them in rows gives the world to local transformation matrix
    # therefore, we need to transpose the matrix. This can be done by renaming row to col for each vector
    world_to_local=xr.concat([l.rename(row='col') for l in [loc_xe, loc_ye, loc_ze]], dim='row')
    return world_to_local

def _get_tangential_coords(anchor_azi, anchor_elv, data_azi, data_elv):
    data_azi = format_input_xarray(data_azi)
    data_elv = format_input_xarray(data_elv)
    conv_cart_to_loc = _cart_to_tangential_matrix(anchor_azi, anchor_elv)
    # phi_sun, theta_sun = _azi_elv_to_theta_phi(data_azi), data_elv))
    sun_distance = 360/(2*np.pi)  # this way, 1deg sun offset is roughly 1 unit in the local coordinate system
    # sun_positions = xr.concat(_phitheta_to_cartesian(phi_sun, theta_sun, sun_distance), dim='col')
    positions = sun_distance* xr.concat(spherical_to_xyz(data_azi, data_elv), dim='col')
    sun_pos_local = (conv_cart_to_loc*positions).sum(dim='col')
    return sun_pos_local



# Test: radar always in one fixed position, sun moves around
# test_elv_radar=xr.DataArray([0], dims='test')
# test_azi_radar=xr.DataArray([0], dims='test')
# test_elv_sun=xr.DataArray(np.arange(-2,2, 0.5), dims='test_x')
# test_elv_sun=test_elv_radar+test_elv_sun
# test_azi_sun=xr.DataArray(np.arange(-2, 2, 0.5), dims='test_y')
# test_elv_radar, test_azi_radar, test_elv_sun, test_azi_sun=xr.broadcast(test_elv_radar, test_azi_radar, test_elv_sun, test_azi_sun)
# test_elv_radar=test_elv_radar.stack(sample=['test', 'test_x', 'test_y']).drop_vars(['test', 'test_x', 'test_y'])
# test_azi_radar=test_azi_radar.stack(sample=['test', 'test_x', 'test_y']).drop_vars(['test', 'test_x', 'test_y'])
# test_elv_sun=test_elv_sun.stack(sample=['test', 'test_x', 'test_y']).drop_vars(['test', 'test_x', 'test_y'])
# test_azi_sun=test_azi_sun.stack(sample=['test', 'test_x', 'test_y']).drop_vars(['test', 'test_x', 'test_y'])
# test_sun_pos_local=get_sunpos_local(test_elv_radar, test_azi_radar, test_elv_sun, test_azi_sun, 0, 0)
# fig, ax=plt.subplots(figsize=(10,10))
# ax.scatter(test_sun_pos_local.sel(row=0).values, test_sun_pos_local.sel(row=1).values)


def norm_signal(signal):
    """Normalize the signal to the range [0, 1]."""
    return (signal-signal.min())/(signal.max()-signal.min())

def process_lut_argument(lut):
    """Process the LUT argument to ensure it is an xarray DataArray."""
    if lut is None:
        lut=Path(sc_params['lutpath'])
    if isinstance(lut, str):
        lut= Path(lut)
    if isinstance(lut, Path):
        lutpath=lut
        if lutpath.exists():
            logger.info('Loading lookup table...')
            lut = xr.open_dataarray(lutpath)
        else:
            lut = calculate_lut()
            lutpath.parent.mkdir(parents=True, exist_ok=True)
            lut.to_netcdf(lutpath)
            logger.info("Lookup table calculated and saved to %s.", lutpath)
    elif not isinstance(lut, xr.DataArray):
        raise ValueError("lut must be an xarray DataArray, string, Pathlib.Path or None, got %s" % type(lut))
    return lut



class SunSimulator(object):
    def __init__(self, dgamma, domega, dtime, fwhm_x, fwhm_y, backlash_gamma, limb_darkening, lut, sky=None):
        self.lut = process_lut_argument(lut)
        self.fwhm_x = fwhm_x
        self.fwhm_y = fwhm_y
        self.limb_darkening = limb_darkening
        self.sky = sky
        self.local_scanner = BacklashScanner(dgamma, domega, dtime, backlash_gamma)
    
    def get_params(self):
        """Get the parameters of the simulator as a dictionary."""
        scanner_params = self.local_scanner.get_params()
        return {
            "dgamma": scanner_params['gamma_offset'],
            "domega": scanner_params['omega_offset'],
            "dtime": scanner_params['dtime'],
            "fwhm_x": self.fwhm_x,
            "fwhm_y": self.fwhm_y,
            "backlash_gamma": scanner_params['backlash_gamma'],
            "limb_darkening": self.limb_darkening
        }

    def _lookup_interp(self, **kwargs):
        """Select scalar dimensions in the lookup table directly and interpolate the rest."""
        sizes={k:self.lut.sizes[k] for k in kwargs.keys()}
        len1=[k for k, v in sizes.items() if v == 1]
        longer= [k for k, v in sizes.items() if v > 1]
        lut=self.lut.sel(**{k: kwargs[k] for k in len1})
        if len(longer) > 0:
            lut = lut.interp(**{k: kwargs[k] for k in longer})
        return lut

    def _lookup(self, tangential_coordinates):
        # sun_sim=lut.sel(lx=sun_pos_local.sel(row=0), ly=sun_pos_local.sel(row=1), fwhm_x=fwhm_x, fwhm_y=fwhm_y, method='nearest')
        # sun_sim = self.lut.interp(lx=tangential_coordinates.sel(row=0), ly=tangential_coordinates.sel(
        #     row=1), fwhm_x=self.fwhm_x, fwhm_y=self.fwhm_y, limb_darkening=self.limb_darkening)
        sun_sim= self._lookup_interp(lx=tangential_coordinates.sel(row=0), ly=tangential_coordinates.sel(row=1), fwhm_x=self.fwhm_x, fwhm_y=self.fwhm_y, limb_darkening=self.limb_darkening)
        # sun_sim=lut.isel(limb_darkening=-1).interp(lx=tangential_coordinates.sel(row=0), ly=tangential_coordinates.sel(row=1), fwhm_x=fwhm_x, fwhm_y=fwhm_y)
        # sun_sim=lut.sel(lx=sun_pos_local.sel(row=0), ly=sun_pos_local.sel(row=1), method='nearest').interp(fwhm_x=fwhm_x, fwhm_y=fwhm_y)
        # sun_sim=lut.interp(lx=sun_pos_local.sel(row=0), ly=sun_pos_local.sel(row=1)).interp(fwhm_x=fwhm_x, fwhm_y=fwhm_y)
        return sun_sim
    
    def get_sunpos_tangential(self, gamma, omega, sun_azi, sun_elv, gammav, omegav):
        beam_azi, beam_elv = self.local_scanner.forward(gamma, omega, gammav=gammav, omegav=omegav)
        sunpos_tangential = _get_tangential_coords(beam_azi, beam_elv, sun_azi, sun_elv)
        return sunpos_tangential

    def check_within_lut(self, gamma, omega, sun_azi, sun_elv, gammav, omegav):
        sun_pos_tangential = self.get_sunpos_tangential(gamma, omega, sun_azi, sun_elv, gammav=gammav, omegav=omegav)
        lxmin, lxmax = self.lut.lx.min().item(), self.lut.lx.max().item()
        lymin, lymax = self.lut.ly.min().item(), self.lut.ly.max().item()
        valid = (sun_pos_tangential.sel(row=0) > lxmin) & (sun_pos_tangential.sel(row=0) < lxmax) & (
            sun_pos_tangential.sel(row=1) > lymin) & (sun_pos_tangential.sel(row=1) < lymax)
        return valid
    
    def forward_sun(self, gamma, omega, sun_azi, sun_elv, gammav, omegav):
        # get the tangential coordinates of the sun position
        # Since we are not using the time in the simulation, it is possible to calcualte the sun positions only once externally and save the expensive calculation in the fit every time.
        # Therefore, this version of forward exists, which takes the sun position as input.
        sunpos_tangential = self.get_sunpos_tangential(gamma, omega, sun_azi, sun_elv, gammav, omegav)
        sun_sim = self._lookup(sunpos_tangential)
        return sun_sim
    
    def forward(self, gamma, omega, time, gammav, omegav):
        sun_azi, sun_elv = self.sky.compute_sun_location(t=time)
        return self.forward_sun(gamma, omega, sun_azi, sun_elv, gammav=gammav, omegav=omegav)
        
    def get_calibrated_pair(self, time, reverse):
        """Given a time, calculate the sun position at this time and the corresponding scanner angles.
        """
        #This function implements the "stationary assumption": We calculate a pair of scanner and celestial positions assuming gammav and omegav = 0. The scanner fit function will then do the same assumption.
        beam_azi, beam_elv = self.sky.compute_sun_location(t=time)
        gamma_s, omega_s=self.local_scanner.inverse(beam_azi, beam_elv, gammav=0.0, omegav=0.0, reverse=reverse)
        return gamma_s, omega_s, beam_azi, beam_elv


def forward_model(params_dict, gamma, omega, sun_azi, sun_elv, gammav, omegav, lut):
    simulator= SunSimulator(**params_dict, lut=lut)
    # for performance reasons, we we use the forward_sun method and calculate the sun position once externally
    sun_sim = simulator.forward_sun(gamma, omega, sun_azi, sun_elv, gammav, omegav)
    return sun_sim
        
def optimize_function(params_list, gamma, omega, sun_azi, sun_elv, signal_norm, gammav, omegav, lut):
    params_dict= {k: params_list[v] for k, v in SUNSIM_PARAMETER_MAP.items()}
    sun_sim = forward_model(params_dict, gamma, omega, sun_azi, sun_elv, gammav, omegav, lut)
    error = sun_sim-signal_norm
    # se= (error**2).sum().item()
    return rmse(error).item()


class SunSimulationEstimator(object):
    def __init__(self, sky, params_optimize=None, params_guess=None, params_bounds=None, lut=None):
        self.lut=process_lut_argument(lut)
        self.sky = sky
        if params_optimize is None:
            params_optimize = sc_params['sunsim_params_optimize'].copy()
        if params_guess is None:
            params_guess = {}
        if params_bounds is None:
            params_bounds = {}
        # check that only valid parameters are provided
        if not set(params_optimize).issubset(SUNSIM_PARAMETER_MAP.keys()):
            raise ValueError(f"Invalid parameters to optimize: {params_optimize}. Valid parameters are: {SUNSIM_PARAMETER_MAP.keys()}")
        if not set(params_guess.keys()).issubset(SUNSIM_PARAMETER_MAP.keys()):
            raise ValueError(f"Invalid parameters to guess: {params_guess.keys()}. Valid parameters are: {SUNSIM_PARAMETER_MAP.keys()}")
        if not set(params_bounds.keys()).issubset(SUNSIM_PARAMETER_MAP.keys()):
            raise ValueError(f"Invalid parameters to bound: {params_bounds.keys()}. Valid parameters are: {SUNSIM_PARAMETER_MAP.keys()}")
        # fill missing parameters with defaults
        params_guess={**sc_params['sunsim_params_guess'], **params_guess}
        params_bounds={**sc_params['sunsim_params_bounds'], **params_bounds}


        self.params_optimize = params_optimize
        self.params_guess = params_guess
        self.params_bounds = params_bounds
    
    def fit(self, gamma, omega, time, signal, gammav, omegav, brute_force=True, brute_force_points=3):
        signal_norm = norm_signal(signal)
        time_max = signal_norm.argmax()

        # dgamma and domega can be None, in which case they are determined dynamically based on the scanner and sun position at the time of maximum signal
        params_guess = self.params_guess.copy()
        params_bounds = self.params_bounds.copy()
        if params_guess['dgamma'] is None or params_guess['domega'] is None:
            gamma_max, omega_max = gamma[time_max], omega[time_max]
            sun_azi, sun_elv = self.sky.compute_sun_location(t=time[time_max])
            dgamma_guess, domega_guess = guess_offsets(gamma_max, omega_max, sun_azi, sun_elv)
            logger.info(f"Estimated dgamma: {dgamma_guess:.4f}, domega: {domega_guess:.4f}")
            if params_guess['dgamma'] is None:
                params_guess['dgamma'] = dgamma_guess
                params_bounds['dgamma'] = (dgamma_guess+params_bounds['dgamma'][0], dgamma_guess+params_bounds['dgamma'][1]) # in case the guess for dgamma is determined dynamically, the bounds are interpreted as relative to the guess
            if params_guess['domega'] is None:
                params_guess['domega'] = domega_guess
                params_bounds['domega'] = (domega_guess+params_bounds['domega'][0], domega_guess+params_bounds['domega'][1]) 

        gamma_xr= xr.DataArray(gamma, dims='sample')
        omega_xr= xr.DataArray(omega, dims='sample')
        time_xr= xr.DataArray(time, dims='sample')
        signal_norm_xr= xr.DataArray(signal_norm, dims='sample')
        gammav_xr= xr.DataArray(gammav, dims='sample')
        omegav_xr= xr.DataArray(omegav, dims='sample')
        sun_azi, sun_elv = xr.apply_ufunc(self.sky.compute_sun_location, time_xr, output_core_dims=[[],[]])
        # check that with the initial guess, the relative difference between sun and scanner is within the lookup table
        init_simulator= SunSimulator(**params_guess, lut=self.lut)
        valid=init_simulator.check_within_lut(gamma_xr, omega_xr, sun_azi, sun_elv, gammav_xr, omegav_xr)
        if not valid.all():
            logger.warning(f'Warning: {(~valid).sum().item()} datapoints are too far away from the sun. They will be ignored.')
            gamma_xr= gamma_xr.where(valid, drop=True)
            omega_xr= omega_xr.where(valid, drop=True)
            signal_norm_xr= signal_norm_xr.where(valid, drop=True)
            gammav_xr= gammav_xr.where(valid, drop=True)
            omegav_xr= omegav_xr.where(valid, drop=True)
            time_xr= time_xr.where(valid, drop=True)
            sun_azi= sun_azi.where(valid, drop=True)
            sun_elv= sun_elv.where(valid, drop=True)
        
        optimize_args = (gamma_xr, omega_xr, sun_azi, sun_elv, signal_norm_xr, gammav_xr, omegav_xr, self.lut)
        params_guess_list, params_bounds_list= get_parameter_lists(self.params_optimize, params_guess, params_bounds, SUNSIM_PARAMETER_MAP)
        init_rmse = optimize_function(params_guess_list, *optimize_args)
        if brute_force:
            logger.info(f"Brute force optimization enabled with {brute_force_points} points ({brute_force_points**len(self.params_optimize)} total)")
            brute_force_params, brute_force_rmse = optimize_brute_force(params_bounds_list, optimize_function, optimize_args=optimize_args, points=brute_force_points)
            logger.info(f"Best Parameters: " + ", ".join([f"{v:.4f}" for v in brute_force_params]))
            logger.info(f"Best RMSE: {brute_force_rmse:.6f}")
            if init_rmse > brute_force_rmse:
                logger.info(f"Brute force did improve the initial guess from {init_rmse:.6f} to {brute_force_rmse:.6f}")
                params_guess_list = brute_force_params
        #
        opt_res = minimize(optimize_function, params_guess_list, args=optimize_args, bounds=params_bounds_list, method='Nelder-Mead')
        # alternative to minimize:
        # from scipy.optimize import differential_evolution
        # res = differential_evolution(objective, bounds, args=(ds,))
        # logger.info(
        #     f"dgamma: {res.x[0]:.2f}, domega: {res.x[1]:.2f}, fwhm_azi: {res.x[2]:.2f}, fwhm_elv: {res.x[3]:.2f}, azi_backlash: {res.x[4]:.2f}, limb_darkening: {res.x[5]:.2f}")
        # logger.info(f'RMSE: {res.fun:.3f}')
        # return res.x, res.fun  # params and rmse
        fit_result_list=opt_res.x
        fit_result_dict={k:fit_result_list[v] for k,v in SUNSIM_PARAMETER_MAP.items()}
        logger.info("Optimization Result:\n" + '\n'.join([f"{k}: {v:.4f}" for k, v in fit_result_dict.items()]))
        init_rmse = optimize_function(params_guess_list, *optimize_args)
        logger.info(f"Initial objective: {init_rmse:.6f}")
        logger.info(f"Optimal objective: {opt_res.fun:.6f}")
        fitted_simulator= SunSimulator(**fit_result_dict, lut=self.lut, sky=self.sky)
        return fitted_simulator, opt_res.fun


