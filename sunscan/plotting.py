import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from sunscan.sun_simulation import SunSimulator, norm_signal

def _plot_points_tangent_plane(sun_pos_plot, sun_signal, ax, vmin=0, vmax=1, cmap='turbo'):
    ax.axvline(x=0, color='k', linestyle='--')
    ax.axhline(y=0, color='k', linestyle='--')
    im = ax.scatter(sun_pos_plot.sel(row=0).values, sun_pos_plot.sel(row=1).values,
                    c=sun_signal, vmin=vmin, vmax=vmax, cmap=cmap, s=9.0)
    ax.set_xlabel('Cross-elevation [deg]')
    ax.set_ylabel('Co-elevation [deg]')
    ax.set_aspect('equal')
    return im


def plot_sunscan_simulation(simulator:SunSimulator, gamma, omega, time, signal_original, gammav, omegav, sky):
    sun_azi, sun_elv=sky.compute_sun_location(time)
    signal_normed = norm_signal(signal_original)
    starttime = pd.to_datetime(time.min())
    sun_sim= simulator.forward_sun(gamma, omega, sun_azi, sun_elv, gammav, omegav)
    params=simulator.get_params()

    sun_pos_corrected = simulator.get_sunpos_tangential(gamma, omega, sun_azi, sun_elv, gammav, omegav)
    plane_full_x = xr.DataArray(np.linspace(sun_pos_corrected.isel(row=0).min().item(),
                                sun_pos_corrected.isel(row=0).max().item(), 100), dims='plane_x')
    plane_full_y = xr.DataArray(np.linspace(sun_pos_corrected.isel(row=1).min().item(),
                                sun_pos_corrected.isel(row=1).max().item(), 100), dims='plane_y')
    plane_full_x, plane_full_y = xr.broadcast(plane_full_x, plane_full_y)
    sim_full = simulator._lookup(xr.concat([plane_full_x, plane_full_y], dim='row'))

    #
    fig = plt.figure(figsize=(8, 12))
    
    # Create grid: 3 rows x 4 columns for better control
    # Row 1: columns 0-1 and 2-3 (2 plots)
    # Row 2: columns 0-1 and 2-3 (2 plots) 
    # Row 3: columns 1-2 (1 centered plot)
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2, aspect='auto')  # Top left
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2, aspect='auto')  # Top right
    ax3 = plt.subplot2grid((3, 4), (1, 0), colspan=2, aspect='equal')  # Middle left
    ax4 = plt.subplot2grid((3, 4), (1, 2), colspan=2, aspect='equal')  # Middle right
    ax5 = plt.subplot2grid((3, 4), (2, 1), colspan=2, aspect='equal')  # Bottom center
    axs= [ax1, ax2, ax3, ax4, ax5]
    
    ax = ax1

    def plot_points_gammaomega(gamma, omega, c, ax):
        im = ax.scatter(gamma, omega, c=c, cmap='turbo', s=13)
        ax.set_xlabel('Gamma ("Azimuth axis") [deg]')
        ax.set_ylabel('Omega ("Elevation axis") [deg]')
        return im
    im = plot_points_gammaomega(gamma, omega, signal_original, ax)
    fig.colorbar(im, ax=ax, label='Signal strength [dB]')
    ax = ax2
    im = plot_points_gammaomega(gamma, omega, sun_sim, ax)
    # remove y tick labels
    ax.set_yticklabels([])
    ax.set_ylabel('')
    fig.colorbar(im, ax=ax, label='Simulated signal [normalized]')

    # Plot measurements and simulation with the uncorrected tangent plane positions
    # simulator_noback = SunSimulator(dgamma=params['dgamma'], domega=params['domega'], fwhm_x=params['fwhm_x'], fwhm_y=params['fwhm_y'], limb_darkening=params['limb_darkening'], backlash_gamma=0.0, dtime=0.0, lut=simulator.lut, sky=simulator.sky)

    # sun_pos_noback = simulator_noback.get_sunpos_tangential(gamma, omega, sun_azi, sun_elv, gammav, omegav)
    # sun_sim_noback = simulator_noback.forward_sun(gamma, omega, sun_azi, sun_elv, gammav, omegav)
    # ax = axs[1, 0]
    # im = _plot_points_tangent_plane(sun_pos_noback, signal_normed, ax)
    # ax = axs[1, 1]
    # im = _plot_points_tangent_plane(sun_pos_noback, sun_sim_noback, ax)
    # ax.set_yticklabels([])
    # ax.set_ylabel('')

    ax = ax5
    diff=signal_normed-sun_sim
    im = _plot_points_tangent_plane(sun_pos_corrected, diff , ax, vmin=None, vmax=None, cmap='coolwarm')
    fig.colorbar(im, ax=ax, label='Measured - Simulated [normalized]')

    ax = ax3
    im = _plot_points_tangent_plane(sun_pos_corrected, signal_normed, ax)
    ax = ax4
    ax.pcolormesh(plane_full_x.values, plane_full_y.values, sim_full.values, cmap='turbo', alpha=0.2)
    ax.contour(plane_full_x.values, plane_full_y.values, sim_full.values, levels=[
               0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99], cmap='turbo', linewidths=1)
    im = _plot_points_tangent_plane(sun_pos_corrected, sun_sim, ax)
    ax.set_yticklabels([])
    ax.set_ylabel('')

    # Create a single colorbar for the lower row
    cax = fig.add_axes([0.2, 0.0, 0.7, 0.02])  # Adjust position and size of the colorbar
    fig.colorbar(im, cax=cax, orientation='horizontal', label='Normalized Signal Strength')
    reverse = omega.mean()>90
    fig.suptitle(f"{starttime.strftime('%Y-%m-%d %H:%M')}\n"+"\n".join([f"{k}: {v:.4f}" for k, v in params.items()])+f'\nreverse: {reverse}', fontsize='small')
    ax.set_aspect('equal')

    ax1.annotate('Scanner Coordinates', xy=(-0.4, 0.5), xycoords='axes fraction',
                       ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    # ax3.annotate('Beam-centered coords; w/o backlash/dtime', xy=(-0.35, 0.5), xycoords='axes fraction', ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    ax3.annotate('Beam-centered coords', xy=(-0.35, 0.5), xycoords='axes fraction',
                       ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    # add column labels on the top
    ax1.annotate('Measurement', xy=(0.5, 1.05), xycoords='axes fraction',
                       ha='center', va='center', fontsize=12, fontweight='bold')
    ax2.annotate('Simulation', xy=(0.5, 1.05), xycoords='axes fraction',
                       ha='center', va='center', fontsize=12, fontweight='bold')
    return fig, axs