"""Module for simulating sun scans and fitting the simulation to real data."""
import argparse
import datetime as dt
from itertools import product
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import convolve2d

from sunscan import logger, sc_params
from sunscan.utils import _phitheta_to_cartesian, _azi_elv_to_theta_phi
from sunscan.scanner import IdentityScanner
from sunscan.fit_utils import get_parameter_lists, optimize_brute_force, rmse

# import mira_utils.sunscan.processing as sunproc
from sunscan import SkyObject
# from mira_utils.preprocess import open_znc

PARAMETER_MAP = {
    "dgamma": 0,
    "domega": 1,
    "fwhm_x": 2,
    "fwhm_y": 3,
    "dt": 4,
    "backlash": 5,
    "limb_darkening": 6
}
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



def _cart_to_tangential_matrix(anchor_azi, anchor_elv):
    """Matrix to convert from cartesian coordinates to cartesian coordinates in the tangential plane, 
    anchored at the given position on the unit sphere."""
    anchor_phi, anchor_theta = _azi_elv_to_theta_phi(np.deg2rad(anchor_azi), np.deg2rad(anchor_elv))
    loc_ze = xr.concat(_phitheta_to_cartesian(anchor_phi, anchor_theta), dim='row')
    # y: co-elevation axis
    anc_theta_y = np.pi/2-anchor_theta
    anc_phi_y = (anchor_phi+np.pi) % (2*np.pi)
    loc_ye = xr.concat(_phitheta_to_cartesian(anc_phi_y, anc_theta_y), dim='row')
    # x: cross-elevation axis
    anc_phi_x = (anchor_phi-np.pi/2) % (2*np.pi)
    anc_theta_x = 0*anchor_theta+np.pi/2
    loc_xe = xr.concat(_phitheta_to_cartesian(anc_phi_x, anc_theta_x), dim='row')

    conv_loc_to_cart = xr.concat([loc_xe, loc_ye, loc_ze], dim='col')
    # invert
    conv_cart_to_loc = xr.apply_ufunc(np.linalg.inv, conv_loc_to_cart, input_core_dims=[
                                      ['col', 'row']], output_core_dims=[['col', 'row']])
    return conv_cart_to_loc


def _radar_model(gamma, omega, dgamma, domega, backlash=0, gammadir=None):
    gamma = gamma+dgamma
    if backlash != 0:
        gamma = gamma+backlash*np.sign(gammadir)
    omega = omega+domega
    azi, elv = identity_scanner.forward(gamma, omega)
    return azi, elv


def _get_tangential_coords(anchor_azi, anchor_elv, data_azi, data_elv):
    conv_cart_to_loc = _cart_to_tangential_matrix(anchor_azi, anchor_elv)
    phi_sun, theta_sun = _azi_elv_to_theta_phi(np.deg2rad(data_azi), np.deg2rad(data_elv))
    sun_distance = 360/(2*np.pi)  # this way, 1deg sun offset is roughly 1 unit in the local coordinate system
    sun_positions = xr.concat(_phitheta_to_cartesian(phi_sun, theta_sun, sun_distance), dim='col')
    sun_pos_local = (conv_cart_to_loc*sun_positions).sum(dim='col')
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


def _filter_datapoints(ds, lut):
    # check that without correction the sun is close to the radar and within the lookup table
    sun_pos_original = _get_tangential_coords(ds.scanner_azi, ds.scanner_elv, ds.sun_azi, ds.sun_elv)
    lxmin, lxmax = lut.lx.min().item(), lut.lx.max().item()
    lymin, lymax = lut.ly.min().item(), lut.ly.max().item()
    valid = (sun_pos_original.sel(row=0) > lxmin) & (sun_pos_original.sel(row=0) < lxmax) & (
        sun_pos_original.sel(row=1) > lymin) & (sun_pos_original.sel(row=1) < lymax)
    if not valid.all():
        logger.warning(
            f'Warning: {(~valid).sum().item()} datapoints are too far away from the sun. They will be ignored.')
    ds_filtered = ds.where(valid, drop=True)
    return ds_filtered


# def _optimize_brute_force(ds, lut, bounds):
#     steps = np.array([0.25, 0.5, 0.75])
#     values = []
#     for i, (low, high) in enumerate(bounds):
#         if low == high:
#             values.append([low])
#         else:
#             values.append(low+steps*(high-low))
#     param_combinations = list(product(*values))
#     best_rmse, best_params = float('inf'), None
#     for params in param_combinations:
#         rmse = _objective(params, ds, lut)
#         if rmse < best_rmse:
#             best_rmse = rmse
#             best_params = params
#     return best_params, best_rmse


def _preprocess_data_sunscan(file, processor, lut):
    ds_raw = open_znc(file)[['HSDco', 'Zg', 'elv_e', 'azi_e', 'aziv_e', 'elvv_e']]
    # bring aziv_e and elvv_e to the same time coordinate as the rest
    ds_v = ds_raw[['aziv_e', 'elvv_e']].rename(time_e='time_m', aziv_e='aziv_m', elvv_e='elvv_m')
    ds_v.coords['time_m'] = ds_raw.time_m
    reverse = True if 'rev' in file.stem else False
    ds_processed = processor.process(ds_raw)  # , reverse=reverse)
    ds = ds_processed.merge(ds_v)
    sun_signal = ds.sun_signal
    # sun_signal=np.pow(10, ds.sun_signal/10) #try with linear units
    return ds


def _simulate_sunscan(ds, lut, optimize_params=("dgamma", "domega", "fwhm_x", "fwhm_y", "backlash", "limb_darkening")):


def _check_result_exists(csv_path, time):
    csv_file = Path(csv_path)
    if csv_file.exists():
        df = pd.read_csv(csv_file, sep=';', index_col="time")
        if time.strftime('%Y%m%d_%H%M%S') in df.index:
            return True
    return False


def _save_simulation_result(csv_path, time, params, rmse, file_paths, overwrite=False):
    # Prepare the row as a DataFrame
    row_dict = {"time": time.strftime('%Y%m%d_%H%M%S'), "rmse": rmse}
    for name, idx in PARAMETER_MAP.items():
        row_dict[name] = params[idx]
    row_dict['file_paths'] = [str(Path(f).name) for f in file_paths]
    row = pd.DataFrame([row_dict]).set_index("time")

    csv_file = Path(csv_path)
    if csv_file.exists():
        df = pd.read_csv(csv_file, sep=';', index_col="time")
        if row.index[0] in df.index:
            if not overwrite:
                raise ValueError(f"Entry for time {row.index[0]} already exists. Set overwrite=True to replace.")
            else:
                df.loc[row.index[0]] = row.iloc[0]
        else:
            df = pd.concat([df, row])
    else:
        df = row

    df.sort_index(inplace=True)
    df.to_csv(csv_file, sep=';')
    logger.info(f'Simulation results saved to {csv_file}.')


def _plot_points_tangent_plane(sun_pos_plot, sun_signal, ax):
    ax.axvline(x=0, color='k', linestyle='--')
    ax.axhline(y=0, color='k', linestyle='--')
    im = ax.scatter(sun_pos_plot.sel(row=0).values, sun_pos_plot.sel(row=1).values,
                    c=sun_signal.values, vmin=0, vmax=1, cmap='turbo', s=9.0)
    ax.set_xlabel('Cross-elevation [deg]')
    ax.set_ylabel('Co-elevation [deg]')
    ax.set_aspect('equal')
    return im


def _plot_sunscan_simulation(ds, lut, plot_params):
    # plot_params=[0.3, 0.0, 0.4, 0.1, 0.0]
    # plot_params=initial_guess
    starttime = pd.to_datetime(ds.time_m.min().values)
    sun_pos_original = _get_tangential_coords(ds.scanner_azi, ds.scanner_elv, ds.sun_azi, ds.sun_elv)
    sun_pos_corrected = _get_tangential_coords(*_radar_model(ds.gamma, ds.omega,
                                                             dgamma=plot_params[0], domega=plot_params[1], backlash=plot_params[4], gammav=ds.aziv_m), ds.sun_azi, ds.sun_elv)
    sun_sim = _lookup(sun_pos_corrected, lut, fwhm_x=plot_params[2], fwhm_y=plot_params[3])
    plane_full_x = xr.DataArray(np.linspace(sun_pos_corrected.isel(row=0).min().item(),
                                sun_pos_corrected.isel(row=0).max().item(), 100), dims='plane_x')
    plane_full_y = xr.DataArray(np.linspace(sun_pos_corrected.isel(row=1).min().item(),
                                sun_pos_corrected.isel(row=1).max().item(), 100), dims='plane_y')
    plane_full_x, plane_full_y = xr.broadcast(plane_full_x, plane_full_y)
    sim_full = _lookup(xr.concat([plane_full_x, plane_full_y], dim='row'),
                       lut, fwhm_x=plot_params[2], fwhm_y=plot_params[3])

    #
    fig, axs = plt.subplots(3, 2, figsize=(8, 12))  # , layout='tight')
    ax = axs[0, 0]

    def plot_points_gammaomega(azi, elv, c, ax):
        im = ax.scatter(azi, elv, c=c, cmap='viridis', s=13)
        ax.set_xlabel('Gamma ("Azimuth axis") [deg]')
        ax.set_ylabel('Omega ("Elevation axis") [deg]')
        return im
    im = plot_points_gammaomega(ds.gamma, ds.omega, ds.sun_signal, ax)
    fig.colorbar(im, ax=ax, label='Signal strength [dB]')
    ax = axs[0, 1]
    im = plot_points_gammaomega(ds.gamma, ds.omega, sun_sim, ax)
    # remove y tick labels
    ax.set_yticklabels([])
    ax.set_ylabel('')
    fig.colorbar(im, ax=ax, label='Simulated signal [normalized]')

    # Plot measurements and simulation with the uncorrected tangent plane positions
    ax = axs[1, 0]
    im = _plot_points_tangent_plane(sun_pos_original, ds.sun_signal_norm, ax)
    ax = axs[1, 1]
    im = _plot_points_tangent_plane(sun_pos_original, sun_sim, ax)

    scanner_azi_corrected, scanner_elv_corrected = sunproc.identity_radar_model(
        ds.gamma+plot_params[0], ds.omega+plot_params[1])
    # regardless of the anchor point, if we add the correction to the anchor point, we should get the center of the sun, therefore we simply take the mean
    sun_center_corrected = _get_tangential_coords(
        ds.scanner_azi, ds.scanner_elv, scanner_azi_corrected, scanner_elv_corrected).mean('time_m')
    ax.axvline(sun_center_corrected.sel(row=0).item(), color='grey', linestyle='--')
    ax.axhline(sun_center_corrected.sel(row=1).item(), color='grey', linestyle='--')
    ax.set_yticklabels([])
    ax.set_ylabel('')

    # Plot measurements and correction in the tangent plane if taking the fitted correction parameters into account
    # Now, the maximum should be at (0,0) sun-radar offset
    ax = axs[2, 0]
    im = _plot_points_tangent_plane(sun_pos_corrected, ds.sun_signal_norm, ax)
    ax = axs[2, 1]
    ax.pcolormesh(plane_full_x.values, plane_full_y.values, sim_full.values, cmap='turbo', alpha=0.2)
    ax.contour(plane_full_x.values, plane_full_y.values, sim_full.values, levels=[
               0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99], cmap='turbo', linewidths=1)
    im = _plot_points_tangent_plane(sun_pos_corrected, sun_sim, ax)
    ax.set_yticklabels([])
    ax.set_ylabel('')

    # Create a single colorbar for the lower row
    cax = fig.add_axes([0.2, 0.0, 0.7, 0.02])  # Adjust position and size of the colorbar
    fig.colorbar(im, cax=cax, orientation='horizontal', label='Normalized Signal Strength')
    fig.suptitle(
        f"{starttime.strftime('%Y-%m-%d %H:%M')}\ndgamma: {plot_params[0]:.2f}, domega: {plot_params[1]:.2f}, fwhm_azi: {plot_params[2]:.2f}, fwhm_elv: {plot_params[3]:.2f}, backlash: {plot_params[4]:.2f}\nlimb darkening: {plot_params[5]:.2f}", fontsize='x-large')
    ax.set_aspect('equal')

    axs[0, 0].annotate('Scanner Coordinates', xy=(-0.4, 0.5), xycoords='axes fraction',
                       ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    axs[1, 0].annotate('Tangential Cartesian Plane', xy=(-0.35, 0.5), xycoords='axes fraction',
                       ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    axs[2, 0].annotate('Tangential Plane w. Correction', xy=(-0.35, 0.5), xycoords='axes fraction',
                       ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    # add column labels on the top
    axs[0, 0].annotate('Measurement', xy=(0.5, 1.05), xycoords='axes fraction',
                       ha='center', va='center', fontsize=12, fontweight='bold')
    axs[0, 1].annotate('Simulation', xy=(0.5, 1.05), xycoords='axes fraction',
                       ha='center', va='center', fontsize=12, fontweight='bold')
    return fig, axs

class SunSimulator(object):
    def __init__(self, dgamma, domega, dtime, fwhm_x, fwhm_y, backlash, limb_darkening, sky, lut):
        self.dgamma = dgamma
        self.domega = domega
        self.dtime = dtime
        self.fwhm_x = fwhm_x
        self.fwhm_y = fwhm_y
        self.backlash = backlash
        self.limb_darkening = limb_darkening
        self.sky = sky
        self.lut = lut

    def _lookup(self, tangential_coordinates):
        # sun_sim=lut.sel(lx=sun_pos_local.sel(row=0), ly=sun_pos_local.sel(row=1), fwhm_x=fwhm_x, fwhm_y=fwhm_y, method='nearest')
        sun_sim = self.lut.interp(lx=tangential_coordinates.sel(row=0), ly=tangential_coordinates.sel(
            row=1), fwhm_x=self.fwhm_x, fwhm_y=self.fwhm_y, limb_darkening=self.limb_darkening)
        # sun_sim=lut.isel(limb_darkening=-1).interp(lx=tangential_coordinates.sel(row=0), ly=tangential_coordinates.sel(row=1), fwhm_x=fwhm_x, fwhm_y=fwhm_y)
        # sun_sim=lut.sel(lx=sun_pos_local.sel(row=0), ly=sun_pos_local.sel(row=1), method='nearest').interp(fwhm_x=fwhm_x, fwhm_y=fwhm_y)
        # sun_sim=lut.interp(lx=sun_pos_local.sel(row=0), ly=sun_pos_local.sel(row=1)).interp(fwhm_x=fwhm_x, fwhm_y=fwhm_y)
        return sun_sim
    
    def forward(self, gamma, omega, time, gammadir=None):
        sun_elv, sun_azi = self.sky.compute_sun_location(time+self.dtime)
        sun_pos_local = _get_tangential_coords(*_radar_model(gamma, omega,dgamma=self.dgamma, domega=self.domega, backlash=self.backlash, gammadir=gammadir), sun_azi, sun_elv)
        sun_sim = self._lookup(sun_pos_local)
        return sun_sim
        

def forward_model(params_dict, gamma, omega, time, gammadir, lut, sky):
    simulator= SunSimulator(**params_dict, sky=sky, lut=lut)
    sun_sim = simulator.forward(gamma, omega, time, gammadir)
    return sun_sim
        
def optimize_function(params_list, gamma, omega, time, signal_norm, gammadir, lut, sky):
    params_dict= {k: params_list[v] for k, v in PARAMETER_MAP.items()}
    sun_sim = forward_model(params_dict, gamma, omega, time, gammadir, lut, sky)
    error = sun_sim-signal_norm
    # se= (error**2).sum().item()
    return rmse(error)


class SunSimulationEstimator(object):
    def __init__(self, sky, params_optimize=None, params_guess=None, params_bounds=None, lut=None):
        if lut is None:
            if Path(sc_params['lutpath']).exists():
                logger.info('Loading lookup table...')
                lut = xr.open_dataarray(sc_params['lutpath'])
            else:
                lut = calculate_lut()
                lut_file = Path(sc_params['lutpath'])
                lut_file.parent.mkdir(parents=True, exist_ok=True)
                lut.to_netcdf(lut_file)
                logger.info("Lookup table calculated and saved to %s.", lut_file)
        self.lut=lut
        self.sky = sky
        if params_optimize is None:
            params_optimize = sc_params['sun_sim_params_optimize']
        if params_guess is None:
            params_guess = sc_params['sun_sim_params_guess']
        if params_bounds is None:
            params_bounds = sc_params['sun_sim_params_bounds']
        self.params_optimize = params_optimize.copy()
        self.params_guess = params_guess.copy()
        self.params_bounds = params_bounds.copy()
    
    def fit(self, gamma, omega, time, signal, brute_force=True, brute_force_points=3):
        signal_norm = (signal-signal.min())/(signal.max()-signal.min())
        time_max = signal_norm.argmax()

        params_guess = self.params_guess.copy()
        params_bounds = self.params_bounds.copy()
        if params_guess['dgamma'] is None or params_guess['domega'] is None:
            gamma_max, omega_max = gamma[time_max], omega[time_max]
            reverse = omega_max > 90
            sun_azi, sun_elv = self.sky.compute_sun_location(t=time[time_max])
            gamma_sun, omega_sun = identity_scanner.inverse(sun_azi, sun_elv, reverse=reverse)
            dgamma_guess = gamma_sun-gamma_max
            domega_guess = omega_sun-omega_max
            if params_guess['dgamma'] is None:
                params_guess['dgamma'] = dgamma_guess
                params_bounds['dgamma'] = (dgamma_guess+params_bounds['dgamma'][0], dgamma_guess+params_bounds['dgamma'][1]) # in case the guess for dgamma is determined dynamically, the bounds are interpreted as relative to the guess
            if params_guess['domega'] is None:
                params_guess['domega'] = domega_guess
                params_bounds['domega'] = (domega_guess+params_bounds['domega'][0], domega_guess+params_bounds['domega'][1]) 
        
    
        # ds = _filter_datapoints(ds, lut)
        params_guess_list, params_bounds_list= get_parameter_lists(self.params_optimize, params_guess, params_bounds, PARAMETER_MAP)

        optimize_args = (gamma, omega, time, signal_norm, gammadir, self.lut, self.sky)
        if brute_force:
            logger.info(f"Brute force optimization enabled with {brute_force_points} points ({brute_force_points**len(self.params_optimize)} total)")
            brute_force_params, brute_force_rmse = optimize_brute_force(params_bounds_list, optimize_function, optimize_args=optimize_args, points=brute_force_points)
            logger.info(f"Best Parameters: {brute_force_params}")
            logger.info(f"Best RMSE: {brute_force_rmse:.6f}")
            init_rmse = optimize_function(params_guess_list, *optimize_args)
            if init_rmse > brute_force_rmse:
                logger.info(f"Brute force did improve the initial guess from {init_rmse:.6f} to {brute_force_rmse:.6f}")
                params_guess_list = brute_force_params
        #
        res = minimize(optimize_function, params_guess_list, args=optimize_args, bounds=params_bounds_list, method='Nelder-Mead')
        # alternative to minimize:
        # from scipy.optimize import differential_evolution
        # res = differential_evolution(objective, bounds, args=(ds,))
        # logger.info(
        #     f"dgamma: {res.x[0]:.2f}, domega: {res.x[1]:.2f}, fwhm_azi: {res.x[2]:.2f}, fwhm_elv: {res.x[3]:.2f}, azi_backlash: {res.x[4]:.2f}, limb_darkening: {res.x[5]:.2f}")
        # logger.info(f'RMSE: {res.fun:.3f}')
        # return res.x, res.fun  # params and rmse
        fit_result_list=res.x
        fit_result_dict={k:fit_result_list[v] for k,v in PARAMETER_MAP.items()}
        fitted_simulator= SunSimulator(**fit_result_dict, sky=self.sky, lut=self.lut)
        return fitted_simulator



def operational_sunsim(file_paths, processor, sun_config, update_lut=None, overwrite_results='skip'):
    """
    Run the operational sun simulation for a given sunscan file.

    This function processes a sunscan measurement file, simulates the sunscan using a lookup table (LUT),
    fits simulation parameters to the measurement, saves the results, and generates a diagnostic plot.

    Args:
        file_paths (list of str or Path): List containing a single path to the sunscan file to process.
        processor (SunscanProcessor): Processor object for handling sunscan data preprocessing.
        sun_config (dict): Configuration dictionary with paths for LUT, results, and plot output.
        update_lut (bool or None, optional): If True, recalculates and overwrites the LUT. If False, loads existing LUT.
            If None, recalculates only if LUT does not exist. Default is None.
        overwrite_results (str or bool, optional): Controls behavior if results already exist.
            'skip' (default): Skip processing if results exist.
            True: Overwrite existing results.
            False: Raise an error if results exist.

    Raises:
        ValueError: If more than one file is provided in file_paths.

    Returns:
        None. Results are saved to disk and a plot is generated.

    """
    # remember: time needs to be converted to datetime, somehow like this: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ds.time.values.astype('datetime64[s]')+dt).tolist()
    if len(file_paths) != 1:
        raise ValueError(f"Expected one file, got {len(file_paths)}. Multiple files are not yet supported.")
    file = Path(file_paths[0])
    time = dt.datetime.strptime(file.stem[:15], '%Y%m%d_%H%M%S')
    if overwrite_results == 'skip':
        if _check_result_exists(sun_config['simulation_results'], time):
            logger.info(
                f"Simulation result for {time.strftime('%Y-%m-%d %H:%M:%S')} already exists. Skipping simulation.")
            return
    ds = _preprocess_data_sunscan(file, processor, lut)
    params, rmse = _simulate_sunscan(ds, lut)
    _save_simulation_result(sun_config['simulation_results'], time, params,
                            rmse, file_paths, overwrite=overwrite_results)
    fig, axs = _plot_sunscan_simulation(ds, lut, params)
    plot_folder = Path(sun_config['plot_folder'])
    outfile = plot_folder/time.strftime('%Y/%m')/(file.stem+'_simulation.png')
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    logger.info(f"Simulation plot saved to {outfile}.")
    # inp.write_log(outfile)


if __name__ == "__main__":
    __main__()
