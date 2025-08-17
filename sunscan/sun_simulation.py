"""Module for simulating sun scans and fitting the simulation to real data."""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage import convolve

from sunscan.utils import logger
from sunscan.math_utils import spherical_to_xyz, rmse
from sunscan.scanner import IdentityScanner, BacklashScanner
from sunscan.fit_utils import get_parameter_lists, optimize_brute_force
from sunscan.utils import guess_offsets, format_input_xarray, db_to_linear, linear_to_db
from sunscan.params import SUNSIM_PARAMETER_MAP, sc_params

identity_scanner = IdentityScanner()
LUT_VERSION='2.0'

class LookupTable:
    def __init__(self, dataarray, apparent_sun_diameter):
        self.lut=dataarray
        self.apparent_sun_diameter=apparent_sun_diameter

    @staticmethod
    def _from_file(filepath):
        da = xr.open_dataarray(filepath)
        version= da.attrs.get('version', None)
        return da, version
    
    @classmethod
    def from_file(cls, filepath, apparent_sun_diameter):
        logger.info('Loading lookup table...')
        da, version = cls._from_file(filepath)
        if version!= LUT_VERSION:
            logger.warning("Lookup table version mismatch: expected %s, got %s", LUT_VERSION, version)
        return cls(da, apparent_sun_diameter)

    @staticmethod
    def calculate_new(dgamma_range=None, domega_range=None, resolution=401, fwhm_x=None, fwhm_y=None, limb_darkening=None):
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
        # the input values are in degrees, but the lut is in units of the sun diameter
        # to create the lookup table, we assume a sun diameter of 0.532 degrees, which is the yearly average in Germany
        # This value is not too important, it only determines the range of the lookup table
        apparent_sun_diameter = 0.532
        lookup_range_x_su = dgamma_range/apparent_sun_diameter
        lookup_range_y_su = domega_range/apparent_sun_diameter
        lx = xr.DataArray(np.linspace(-lookup_range_x_su, lookup_range_x_su, resolution), dims='lx')
        lx.coords['lx'] = lx
        ly = xr.DataArray(np.linspace(-lookup_range_y_su, lookup_range_y_su, resolution), dims='ly')
        ly.coords['ly'] = ly
        # I found that non-zero centered luts can create an offset in contour plots
        assert 0 in lx.values
        assert 0 in ly.values

        limb_darkening = xr.DataArray(limb_darkening, dims='limb_darkening')
        limb_darkening.coords['limb_darkening'] = limb_darkening
        sun_rl = 1/2  # radius of the sun in units of the sun diameter
        sundist = np.sqrt(lx**2+ly**2)
        sun = 1.0-(1.0-limb_darkening)*sundist/sun_rl
        sun = xr.where(sundist < sun_rl, sun, np.nan)
        sun = sun.dropna(dim='lx', how='all').dropna(dim='ly', how='all').fillna(0).rename(lx='sx', ly='sy')
        # Gaussian beam
        # FWHM = Full width half maximum
        # FWHM = 2.355 * sigma
        sigma_to_fwhm = 2*np.sqrt(2*np.log(2))
        fwhm_x_su = xr.DataArray(fwhm_x, dims='fwhm_x')/apparent_sun_diameter
        fwhm_x_su.coords['fwhm_x'] = fwhm_x
        sigma_x_su = fwhm_x_su/sigma_to_fwhm
        fwhm_y_su = xr.DataArray(fwhm_y, dims='fwhm_y')/apparent_sun_diameter
        fwhm_y_su.coords['fwhm_y'] = fwhm_y
        sigma_y_su = fwhm_y_su/sigma_to_fwhm
        gaussian = np.exp(-lx**2/(2*sigma_x_su**2)-ly**2/(2*sigma_y_su**2))
        gaussian=gaussian/gaussian.sum(('lx', 'ly'))  # normalize to 1
        # convolve sun and beam
        def convolve_2d(gauss, sun):
            return convolve(gauss, sun, mode='constant', cval=0.0)
        lut = xr.apply_ufunc(convolve_2d, gaussian, sun, input_core_dims=[['lx', 'ly'], [
                            'sx', 'sy']], output_core_dims=[['lx', 'ly']], vectorize=True)
        lut.lx.attrs['units'] = 'sun diameter'
        lut.lx.attrs['description'] = 'Cross Elevation distance in beam centered coordinates'
        lut.ly.attrs['units'] = 'sun diameter'
        lut.ly.attrs['description'] = 'Co Elevation distance in beam centered coordinates'
        lut.fwhm_x.attrs['units'] = 'degrees'
        lut.fwhm_x.attrs['description'] = 'FWHM of the beam in cross elevation direction'
        lut.fwhm_y.attrs['units'] = 'degrees'
        lut.fwhm_y.attrs['description'] = 'FWHM of the beam in co elevation direction'
        lut.attrs['version'] = LUT_VERSION
        logger.info('Lookup table size: %.2f GB', lut.nbytes/1024**3)
        return lut


    def save(self, filepath):
        self.lut.to_netcdf(filepath)

    @classmethod
    def load_or_create_if_necessary(cls, lutpath, apparent_sun_diameter):
        """Process the LUT argument to ensure it is an xarray DataArray."""
        if lutpath is None:
            lutpath=Path(sc_params['lutpath'])
        if isinstance(lutpath, str):
            lutpath= Path(lutpath)
        if isinstance(lutpath, Path):
            lutpath=lutpath
            if lutpath.exists():
                logger.info("Loading lookup table from %s.", lutpath)
                da, version= cls._from_file(lutpath)
                if version != LUT_VERSION:
                    logger.warning("Lookup table version mismatch: expected %s, got %s", LUT_VERSION, version)
                    logger.info("Recalculating lookup table...")
                    da= cls.calculate_new()
                    lut= cls(da, apparent_sun_diameter)
                    lut.save(lutpath)
                    logger.info("Lookup table calculated and saved to %s.", lutpath)
                    return lut
                else:
                    return cls(da, apparent_sun_diameter)
            else:
                logger.info("Calculating new lookup table...")
                da = cls.calculate_new()
                lut = cls(da, apparent_sun_diameter)
                lutpath.parent.mkdir(parents=True, exist_ok=True)
                lut.save(lutpath)
                logger.info("Lookup table calculated and saved to %s.", lutpath)
                return lut
        else:
            raise ValueError(f"lutpath must be either None, a string or a Path object. Received: {type(lutpath)}")
    
    def deg_to_su(self, deg):
        """Convert an angle in degrees to an angles in units of the sun diameter"""
        return deg / self.apparent_sun_diameter
    
    def su_to_deg(self, su):
        """Convert an angle in units of the sun diameter to degrees"""
        return su * self.apparent_sun_diameter

    def _lookup_interp(self, **kwargs):
        """Select scalar dimensions in the lookup table directly and interpolate the rest."""
        sizes={k:self.lut.sizes[k] for k in kwargs.keys()}
        len1=[k for k, v in sizes.items() if v == 1]
        longer= [k for k, v in sizes.items() if v > 1]
        lut=self.lut.sel(**{k: kwargs[k] for k in len1})
        if len(longer) > 0:
            lut = lut.interp(**{k: kwargs[k] for k in longer})
        return lut

    

    def lookup(self, lx, ly, fwhm_x, fwhm_y, limb_darkening):
        lx_su=self.deg_to_su(lx)
        ly_su=self.deg_to_su(ly)
        # sun_contribution=lut.sel(lx=sun_pos_local.sel(row=0), ly=sun_pos_local.sel(row=1), fwhm_x=fwhm_x, fwhm_y=fwhm_y, method='nearest')
        # sun_contribution = self.lut.interp(lx=tangential_coordinates.sel(row=0), ly=tangential_coordinates.sel(
        #     row=1), fwhm_x=self.fwhm_x, fwhm_y=self.fwhm_y, limb_darkening=self.limb_darkening)
        sun_contribution= self._lookup_interp(lx=lx_su, ly=ly_su, fwhm_x=fwhm_x, fwhm_y=fwhm_y, limb_darkening=limb_darkening)
        # sun_contribution=lut.isel(limb_darkening=-1).interp(lx=tangential_coordinates.sel(row=0), ly=tangential_coordinates.sel(row=1), fwhm_x=fwhm_x, fwhm_y=fwhm_y)
        # sun_contribution=lut.sel(lx=sun_pos_local.sel(row=0), ly=sun_pos_local.sel(row=1), method='nearest').interp(fwhm_x=fwhm_x, fwhm_y=fwhm_y)
        # sun_contribution=lut.interp(lx=sun_pos_local.sel(row=0), ly=sun_pos_local.sel(row=1)).interp(fwhm_x=fwhm_x, fwhm_y=fwhm_y)
        return sun_contribution
    
    def check_within_lut(self, px, py):
        px_su= self.deg_to_su(px)
        py_su= self.deg_to_su(py)
        lxmin, lxmax = self.lut.lx.min().item(), self.lut.lx.max().item()
        lymin, lymax = self.lut.ly.min().item(), self.lut.ly.max().item()
        valid = (px > lxmin) & (px < lxmax) & (py > lymin) & (py < lymax)
        return valid



def get_beamcentered_unitvectors(azi_beam, elv_beam):
    """Matrix to convert from cartesian world coordinates to cartesian coordinates in the tangential plane, 
    anchored at the given position on the unit sphere."""
    azi_beam = format_input_xarray(azi_beam)
    elv_beam = format_input_xarray(elv_beam)
    bz = xr.concat(spherical_to_xyz(azi_beam, elv_beam), dim='row')
    world_ze= xr.zeros_like(bz)
    world_ze[{'row': 2}] = 1.0
    # x: cross-elevation axis
    bx=xr.cross(world_ze, bz, dim='row')
    # normalize
    bx = bx / xr.apply_ufunc(np.linalg.norm, bx, input_core_dims=[['row']], kwargs={'axis': -1})
    # y: co-elevation axis
    by=xr.cross(bz, bx, dim='row')
    return bx, by, bz

def get_world_to_beam_matrix(azi_beam, elv_beam):
    # stacking those vectors in columns would give the local to world transformation matrix
    # stacking them in rows gives the world to local transformation matrix
    # therefore, we need to transpose the matrix. This can be done by renaming row to col for each vector
    bx, by, bz = get_beamcentered_unitvectors(azi_beam, elv_beam)
    world_to_beam=xr.concat([l.rename(row='col') for l in [bx, by, bz]], dim='row')
    return world_to_beam

def get_beamcentered_coords(azi_beam, elv_beam, azi_sun, elv_sun):
    azi_sun = format_input_xarray(azi_sun)
    elv_sun = format_input_xarray(elv_sun)
    # phi_sun, theta_sun = _azi_elv_to_theta_phi(data_azi), data_elv))
    sun_distance = 360/(2*np.pi)  # this way, 1deg sun offset is roughly 1 unit in the local coordinate system
    # sun_positions = xr.concat(_phitheta_to_cartesian(phi_sun, theta_sun, sun_distance), dim='col')
    positions = sun_distance* xr.concat(spherical_to_xyz(azi_sun, elv_sun), dim='col')
    world_to_beam= get_world_to_beam_matrix(azi_beam, elv_beam)
    sun_pos_beam = (world_to_beam*positions).sum(dim='col')
    return sun_pos_beam

class SunSimulator(object):
    def __init__(self, dgamma, domega, dtime, fwhm_x, fwhm_y, backlash_gamma, limb_darkening, lut: LookupTable, sky_lin, sun_lin, sky=None):
        self.lut = lut
        self.fwhm_x = fwhm_x
        self.fwhm_y = fwhm_y
        self.limb_darkening = limb_darkening
        self.sky = sky
        self.sky_lin = sky_lin
        self.sun_lin = sun_lin
        self.local_scanner = BacklashScanner(dgamma, domega, dtime, backlash_gamma, flex=0)
    
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
            "limb_darkening": self.limb_darkening,
            "sky_lin": self.sky_lin,
            "sun_lin": self.sun_lin,
        }
    
    def __repr__(self):
        return "Sun Simulator Object:\n" +\
            f"Azimuth Offset: {self.local_scanner.gamma_offset:.4f} º\n" + \
            f"Elevation Offset: {self.local_scanner.omega_offset:.4f} º\n" + \
            f"Time Offset: {self.local_scanner.dtime:.4f} º\n" + \
            f"Beamwidth cross-elevation: {self.fwhm_x:.4f} º\n" + \
            f"Beamwidth co-elevation: {self.fwhm_y:.4f} º\n" + \
            f"Azimuth Backlash: {self.local_scanner.backlash_gamma:.4f} º\n" + \
            f"Limb Darkening factor: {self.limb_darkening:.4f}\n" + \
            f"Sky Noise: {self.sky_lin:.4f} [lin. units]\n" + \
            f"Sun Brightness: {self.sun_lin:.4f} [lin. units]\n"
    
    def get_sunpos_tangential(self, gamma, omega, sun_azi, sun_elv, gammav, omegav):
        beam_azi, beam_elv = self.local_scanner.forward(gamma, omega, gammav=gammav, omegav=omegav)
        sunpos_tangential = get_beamcentered_coords(beam_azi, beam_elv, sun_azi, sun_elv)
        return sunpos_tangential

    def check_within_lut(self, gamma, omega, sun_azi, sun_elv, gammav, omegav):
        sun_pos_tangential = self.get_sunpos_tangential(gamma, omega, sun_azi, sun_elv, gammav=gammav, omegav=omegav)
        valid=self.lut.check_within_lut(sun_pos_tangential.sel(row=0), sun_pos_tangential.sel(row=1))
        return valid
    
    def signal_from_bc_coords(self, lx, ly):
        sun_contribution = self.lut.lookup(lx=lx, ly=ly, fwhm_x=self.fwhm_x, fwhm_y=self.fwhm_y, limb_darkening=self.limb_darkening)
        sun_sim_linear= self.sky_lin*(1-sun_contribution) + self.sun_lin*sun_contribution
        return sun_sim_linear
    
    def forward_sun(self, gamma, omega, sun_azi, sun_elv, gammav, omegav):
        # get the tangential coordinates of the sun position
        # Since we are not using the time in the simulation, it is possible to calculate the sun positions only once externally and save the expensive calculation in the fit every time.
        # Therefore, this version of forward exists, which takes the sun position as input.
        sunpos_tangential = self.get_sunpos_tangential(gamma, omega, sun_azi, sun_elv, gammav, omegav)
        lx, ly=sunpos_tangential.sel(row=0), sunpos_tangential.sel(row=1)
        sun_sim_lin = self.signal_from_bc_coords(lx, ly)
        return sun_sim_lin
    
    def forward(self, gamma, omega, time, gammav, omegav):
        sun_azi, sun_elv = self.sky.compute_sun_location(t=time)
        return self.forward_sun(gamma, omega, sun_azi, sun_elv, gammav=gammav, omegav=omegav)
    
    def get_calibrated_pair(self, gamma, omega, time):
        """Same as get_calibrated_pair_time, but takes gamma, omega and time as input.
        The time for the calibrated pair is determined as the time of the middle sample in the time array.
        """
        gamma=np.atleast_1d(gamma)
        omega=np.atleast_1d(omega)
        time=np.atleast_1d(time)
        index_middle=np.argsort(time)[len(time) // 2]
        time_middle= time[index_middle]
        omega_middle= omega[index_middle]
        reverse= omega_middle > 90
        return self.get_calibrated_pair_time(time_middle, reverse=reverse)
            
    def get_calibrated_pair_time(self, time, reverse):
        """Given a time, calculate the sun position at this time and the corresponding scanner angles.
        """
        #This function implements the "stationary assumption": We calculate a pair of scanner and celestial positions assuming gammav and omegav = 0. The scanner fit function will then do the same assumption.
        beam_azi, beam_elv = self.sky.compute_sun_location(t=time)
        gamma_s, omega_s=self.local_scanner.inverse(beam_azi, beam_elv, gammav=0.0, omegav=0.0, reverse=reverse)
        return gamma_s, omega_s, beam_azi, beam_elv

def sun_lin_from_center_signal(lut: LookupTable, center_lin, sky_lin, fwhm_x, fwhm_y, limb_darkening):
    sun_contribution = lut.lookup(0,0, fwhm_x, fwhm_y, limb_darkening)
    sky_contribution = 1- sun_contribution
    sun_linear = (center_lin - sky_lin* sky_contribution) / sun_contribution
    return sun_linear
        


def forward_model(params_dict, gamma, omega, sun_azi, sun_elv, gammav, omegav, lut, sky_lin, sun_lin):
    simulator= SunSimulator(**params_dict, lut=lut, sky_lin=sky_lin, sun_lin=sun_lin)
    # for performance reasons, we we use the forward_sun method and calculate the sun position once externally
    sun_sim_lin = simulator.forward_sun(gamma, omega, sun_azi, sun_elv, gammav, omegav)
    return sun_sim_lin
        
def optimize_function(params_list, gamma, omega, sun_azi, sun_elv, signal_lin, gammav, omegav, lut:LookupTable, sky_lin, sun_lin):
    params_dict= {k: params_list[v] for k, v in SUNSIM_PARAMETER_MAP.items()}
    if sun_lin is None: 
        # assume that the maximum signal is from pointing to the center of the sun-> Determine sun brightness from the max signal and the beam width
        sun_lin=sun_lin_from_center_signal(lut, signal_lin.max(), sky_lin, params_dict['fwhm_x'], params_dict['fwhm_y'], params_dict['limb_darkening'])
    sun_sim_lin = forward_model(params_dict, gamma, omega, sun_azi, sun_elv, gammav, omegav, lut, sky_lin, sun_lin)
    #db error
    error = linear_to_db(sun_sim_lin) - linear_to_db(signal_lin)
    # linear error
    # error = db_to_linear(sun_sim_db) - db_to_linear(signal_db)
    # se= (error**2).sum().item()
    return rmse(error).item()


class SunSimulationEstimator(object):
    def __init__(self, sky, params_optimize=None, params_guess=None, params_bounds=None, lutpath=None, sky_lin=None, sun_lin=None):
        self.lutpath=lutpath
        self.sky = sky
        self.sky_lin = sky_lin
        self.sun_lin = sun_lin

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
    
    def fit(self, gamma, omega, time, signal_db, gammav, omegav, brute_force=True, brute_force_points=3):
        signal_lin= db_to_linear(signal_db)
        sky_lin=self.sky_lin
        if sky_lin is None:
            sky_lin = signal_lin.min() 
            #if sun_lin is also None, it will be determined during the optimization based on the beam width and the maximum signal

        time_max = signal_lin.argmax()
        apparent_sun_diameter = self.sky.get_sun_diameter(t=time[time_max])
        lut=LookupTable.load_or_create_if_necessary(self.lutpath, apparent_sun_diameter)

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
            if params_guess['domega'] is None:
                params_guess['domega'] = domega_guess

        gamma_xr= xr.DataArray(gamma, dims='sample')
        omega_xr= xr.DataArray(omega, dims='sample')
        time_xr= xr.DataArray(time, dims='sample')
        signal_lin_xr= xr.DataArray(signal_lin, dims='sample')
        gammav_xr= xr.DataArray(gammav, dims='sample')
        omegav_xr= xr.DataArray(omegav, dims='sample')
        sun_azi, sun_elv = xr.apply_ufunc(self.sky.compute_sun_location, time_xr, output_core_dims=[[],[]])
        # check that with the initial guess, the relative difference between sun and scanner is within the lookup table
        init_simulator= SunSimulator(**params_guess, lut=lut, sky=self.sky, sky_lin=sky_lin, sun_lin=self.sun_lin)
        valid=init_simulator.check_within_lut(gamma_xr, omega_xr, sun_azi, sun_elv, gammav_xr, omegav_xr)
        if not valid.all():
            logger.warning(f'Warning: {(~valid).sum().item()} datapoints are too far away from the sun. They will be ignored.')
            gamma_xr= gamma_xr.where(valid, drop=True)
            omega_xr= omega_xr.where(valid, drop=True)
            signal_lin_xr= signal_lin_xr.where(valid, drop=True)
            gammav_xr= gammav_xr.where(valid, drop=True)
            omegav_xr= omegav_xr.where(valid, drop=True)
            time_xr= time_xr.where(valid, drop=True)
            sun_azi= sun_azi.where(valid, drop=True)
            sun_elv= sun_elv.where(valid, drop=True)
        
        optimize_args = (gamma_xr, omega_xr, sun_azi, sun_elv, signal_lin_xr, gammav_xr, omegav_xr, lut, sky_lin, self.sun_lin)
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

        if self.sun_lin is None: 
            sun_lin=sun_lin_from_center_signal(lut, signal_lin.max(), sky_lin, fit_result_dict['fwhm_x'], fit_result_dict['fwhm_y'], fit_result_dict['limb_darkening'])
        else:
            sun_lin=self.sun_lin
        fitted_simulator= SunSimulator(**fit_result_dict, lut=lut, sky=self.sky, sky_lin=sky_lin, sun_lin=sun_lin)
        return fitted_simulator, opt_res.fun


