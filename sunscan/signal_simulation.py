"""Module for simulating sun scans and fitting the simulation to real data."""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage import convolve

from sunscan.utils import logger
from sunscan.math_utils import spherical_to_xyz, rmse, bessel, gaussian
from sunscan.scanner import IdentityScanner, BacklashScanner
from sunscan.fit_utils import get_parameter_lists, optimize_brute_force
from sunscan.utils import guess_offsets, db_to_linear, linear_to_db
from sunscan.params import SUNSIM_PARAMETER_MAP, sc_params

identity_scanner = IdentityScanner()
LUT_VERSION='3.2'

class LookupTable:
    def __init__(self, dataarray, apparent_sun_diameter):
        self.lut=dataarray
        self.apparent_sun_diameter=apparent_sun_diameter

    @staticmethod
    def _from_file(filepath):
        da = xr.open_dataarray(filepath)#.load() #somehow, load makes the interpolation much slower?!
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
        # Beam width
        fwhm_x_su = xr.DataArray(fwhm_x, dims='fwhm_x')/apparent_sun_diameter
        fwhm_x_su.coords['fwhm_x'] = fwhm_x
        fwhm_y_su = xr.DataArray(fwhm_y, dims='fwhm_y')/apparent_sun_diameter
        fwhm_y_su.coords['fwhm_y'] = fwhm_y
        # antenna_pattern = xr.apply_ufunc(gaussian, lx, ly, fwhm_x_su, fwhm_y_su)
        antenna_pattern = xr.apply_ufunc(bessel, lx, ly, fwhm_x_su, fwhm_y_su)
        antenna_pattern=antenna_pattern/antenna_pattern.sum(('lx', 'ly'))  # normalize to 1
        # convolve sun and beam
        def convolve_2d(antenna, sun):
            return convolve(antenna, sun, mode='constant', cval=0.0)
        lut = xr.apply_ufunc(convolve_2d, antenna_pattern, sun, input_core_dims=[['lx', 'ly'], [
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
        lut=lut.sortby(['lx', 'ly', 'fwhm_x', 'fwhm_y', 'limb_darkening'])
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
        lx_su=xr.DataArray(lx_su)
        ly_su=xr.DataArray(ly_su)
        fwhm_x=xr.DataArray(fwhm_x)
        fwhm_y=xr.DataArray(fwhm_y)
        limb_darkening=xr.DataArray(limb_darkening)
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
        valid = (px_su > lxmin) & (px_su < lxmax) & (py_su > lymin) & (py_su < lymax)
        return valid


def get_beamcentered_unitvectors(azi_beam, elv_beam):
    """Matrix to convert from cartesian world coordinates to cartesian coordinates 
    in the tangential plane, anchored at the given position on the unit sphere."""
    # Convert to numpy arrays if needed
    azi_beam = np.asarray(azi_beam)
    elv_beam = np.asarray(elv_beam)
    
    # Get beam direction vector (bz)
    bz = np.array(spherical_to_xyz(azi_beam, elv_beam))  # shape: (3, ...)
    
    # World z-axis vector
    world_ze = np.zeros_like(bz)
    world_ze[2] = 1.0
    
    # x: cross-elevation axis (cross product of world_ze and bz)
    bx = np.cross(world_ze, bz, axis=0)
    
    # Normalize bx
    bx_norm = np.linalg.norm(bx, axis=0, keepdims=True)
    bx = bx / bx_norm
    
    # y: co-elevation axis (cross product of bz and bx)
    by = np.cross(bz, bx, axis=0)
    
    return bx, by, bz

def get_world_to_beam_matrix(azi_beam, elv_beam):
    """Get transformation matrix from world to beam coordinates."""
    bx, by, bz = get_beamcentered_unitvectors(azi_beam, elv_beam)
    
    # Stack the unit vectors as rows to create the transformation matrix world to local
    # Shape will be (3, 3, ...) where the first dimension is row, second is column
    world_to_beam = np.stack([bx, by, bz], axis=0)
    
    return world_to_beam

def get_beamcentered_coords(azi_beam, elv_beam, azi_sun, elv_sun):
    """Get beam-centered coordinates of the sun."""
    # Convert to numpy arrays
    azi_sun = np.asarray(azi_sun)
    elv_sun = np.asarray(elv_sun)
    
    # Distance scaling factor
    sun_distance = 360 / (2 * np.pi)  # this way, 1deg sun offset is roughly 1 unit in the local coordinate system
    
    # Get sun position in world coordinates
    sun_xyz = np.array(spherical_to_xyz(azi_sun, elv_sun))  # shape: (3, ...)
    positions = sun_distance * sun_xyz
    
    # Get transformation matrix
    world_to_beam = get_world_to_beam_matrix(azi_beam, elv_beam)
    
    # Transform sun position to beam coordinates
    # Use einsum for matrix multiplication: 'ij...,j...->i...'
    sun_pos_beam = np.einsum('ij...,j...->i...', world_to_beam, positions)
    
    # Extract x and y components (first two rows)
    bx = sun_pos_beam[0]
    by = sun_pos_beam[1]
    
    return bx, by

class SignalSimulator(object):
    def __init__(self, dgamma, domega, dtime, fwhm_x, fwhm_y, backlash_gamma, limb_darkening, lut: LookupTable, rec_noise, sun_signal, sun=None):
        self.lut = lut
        self.fwhm_x = fwhm_x
        self.fwhm_y = fwhm_y
        self.limb_darkening = limb_darkening
        self.sun = sun
        self.rec_noise = rec_noise
        self.sun_signal = sun_signal
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
            "rec_noise": self.rec_noise,
            "sun_signal": self.sun_signal,
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
            f"Sky(Noise): {self.rec_noise:.4f} [lin. units]\n" + \
            f"Sun Brightness: {self.sun_signal:.4f} [lin. units]\n"
    
    def get_sunpos_beamcentered(self, gamma, omega, sun_azi, sun_elv, gammav, omegav):
        beam_azi, beam_elv = self.local_scanner.forward(gamma, omega, gammav=gammav, omegav=omegav)
        bx, by = get_beamcentered_coords(beam_azi, beam_elv, sun_azi, sun_elv)
        return bx, by

    def check_within_lut(self, gamma, omega, sun_azi, sun_elv, gammav, omegav):
        lx, ly = self.get_sunpos_beamcentered(gamma, omega, sun_azi, sun_elv, gammav=gammav, omegav=omegav)
        valid=self.lut.check_within_lut(lx, ly)
        return valid
    
    def signal_from_bc_coords(self, lx, ly):
        sun_contribution = self.lut.lookup(lx=lx, ly=ly, fwhm_x=self.fwhm_x, fwhm_y=self.fwhm_y, limb_darkening=self.limb_darkening)
        sun_sim_linear= self.rec_noise + self.sun_signal*sun_contribution
        return sun_sim_linear
    
    def forward_sun(self, gamma, omega, sun_azi, sun_elv, gammav, omegav):
        # get the tangential coordinates of the sun position
        # Since we are not using the time in the simulation, it is possible to calculate the sun positions only once externally and save the expensive calculation in the fit every time.
        # Therefore, this version of forward exists, which takes the sun position as input.
        lx, ly = self.get_sunpos_beamcentered(gamma, omega, sun_azi, sun_elv, gammav, omegav)
        sun_sim_lin = self.signal_from_bc_coords(lx, ly)
        return sun_sim_lin.values
    
    def forward(self, gamma, omega, time, gammav, omegav):
        sun_azi, sun_elv = self.sun.compute_sun_location(t=time)
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
        beam_azi, beam_elv = self.sun.compute_sun_location(t=time)
        gamma_s, omega_s=self.local_scanner.inverse(beam_azi, beam_elv, gammav=0.0, omegav=0.0, reverse=reverse)
        return gamma_s, omega_s, beam_azi, beam_elv

def sun_lin_from_center_signal(lut: LookupTable, center_lin, rec_noise, fwhm_x, fwhm_y, limb_darkening):
    sun_contribution = lut.lookup(0,0, fwhm_x, fwhm_y, limb_darkening).item()
    sun_linear = (center_lin - rec_noise) / sun_contribution
    return sun_linear
        


def forward_model(params_dict, gamma, omega, sun_azi, sun_elv, gammav, omegav, lut, rec_noise, sun_signal):
    simulator= SignalSimulator(**params_dict, lut=lut, rec_noise=rec_noise, sun_signal=sun_signal)
    # for performance reasons, we we use the forward_sun method and calculate the sun position once externally
    sun_sim_lin = simulator.forward_sun(gamma, omega, sun_azi, sun_elv, gammav, omegav)
    return sun_sim_lin
        
def optimize_function(params_list, gamma, omega, sun_azi, sun_elv, signal_lin, gammav, omegav, lut:LookupTable, rec_noise, sun_signal):
    params_dict= {k: params_list[v] for k, v in SUNSIM_PARAMETER_MAP.items()}
    if sun_signal is None: 
        # assume that the maximum signal is from pointing to the center of the sun-> Determine sun brightness from the max signal and the beam width
        sun_signal=sun_lin_from_center_signal(lut, signal_lin.max(), rec_noise, params_dict['fwhm_x'], params_dict['fwhm_y'], params_dict['limb_darkening'])
    sun_sim_lin = forward_model(params_dict, gamma, omega, sun_azi, sun_elv, gammav, omegav, lut, rec_noise, sun_signal)
    #db error
    error = linear_to_db(sun_sim_lin) - linear_to_db(signal_lin)
    # linear error
    # error = db_to_linear(sun_sim_db) - db_to_linear(signal_db)
    # se= (error**2).sum().item()
    return rmse(error).item()


class SignalSimulationEstimator(object):
    def __init__(self, sun, params_optimize=None, params_guess=None, params_bounds=None, lutpath=None, sky_signal=None, sun_signal=None):
        self.lutpath=lutpath
        self.sun = sun
        # sky_signal and sun_signal are in linear units
        self.rec_noise = sky_signal # assume that the sky emission is almost zero and the signal we receive is actually receiver noise
        self.sun_signal = sun_signal

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
        rec_noise=self.rec_noise
        if rec_noise is None:
            rec_noise = signal_lin.min() 
            #if sun_lin is also None, it will be determined during the optimization based on the beam width and the maximum signal

        time_max = signal_lin.argmax()
        apparent_sun_diameter = self.sun.get_sun_diameter(t=time[time_max])
        lut=LookupTable.load_or_create_if_necessary(self.lutpath, apparent_sun_diameter)

        # dgamma and domega can be None, in which case they are determined dynamically based on the scanner and sun position at the time of maximum signal
        params_guess = self.params_guess.copy()
        params_bounds = self.params_bounds.copy()
        if params_guess['dgamma'] is None or params_guess['domega'] is None:
            gamma_max, omega_max = gamma[time_max], omega[time_max]
            sun_azi, sun_elv = self.sun.compute_sun_location(t=time[time_max])
            dgamma_guess, domega_guess = guess_offsets(gamma_max, omega_max, sun_azi, sun_elv)
            logger.info(f"Estimated dgamma: {dgamma_guess:.4f}, domega: {domega_guess:.4f}")
            if params_guess['dgamma'] is None:
                params_guess['dgamma'] = dgamma_guess
            if params_guess['domega'] is None:
                params_guess['domega'] = domega_guess

        sun_azi, sun_elv = self.sun.compute_sun_location(t=time)
        # check that with the initial guess, the relative difference between sun and scanner is within the lookup table
        init_simulator= SignalSimulator(**params_guess, lut=lut, sun=self.sun, rec_noise=rec_noise, sun_signal=self.sun_signal)
        valid=init_simulator.check_within_lut(gamma, omega, sun_azi, sun_elv, gammav, omegav)
        if not valid.all():
            logger.warning(f'Warning: {(~valid).sum().item()} datapoints are too far away from the sun. They will be ignored.')
            gamma=gamma[valid]
            omega=omega[valid]
            time=time[valid]
            signal_lin=signal_lin[valid]
            gammav=gammav[valid]
            omegav=omegav[valid]
            sun_azi=sun_azi[valid]
            sun_elv=sun_elv[valid]
        optimize_args = (gamma, omega, sun_azi, sun_elv, signal_lin, gammav, omegav, lut, rec_noise, self.sun_signal)
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

        if self.sun_signal is None: 
            sun_lin=sun_lin_from_center_signal(lut, signal_lin.max(), rec_noise, fit_result_dict['fwhm_x'], fit_result_dict['fwhm_y'], fit_result_dict['limb_darkening'])
        else:
            sun_lin=self.sun_signal
        fitted_simulator= SignalSimulator(**fit_result_dict, lut=lut, sun=self.sun, rec_noise=rec_noise, sun_signal=sun_lin)
        return fitted_simulator, opt_res.fun


