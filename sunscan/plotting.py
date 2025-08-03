import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from sunscan.sun_simulation import SunSimulator, norm_signal

def _plot_points_tangent_plane(sun_pos_plot, sun_signal, ax):
    ax.axvline(x=0, color='k', linestyle='--')
    ax.axhline(y=0, color='k', linestyle='--')
    im = ax.scatter(sun_pos_plot.sel(row=0).values, sun_pos_plot.sel(row=1).values,
                    c=sun_signal, vmin=0, vmax=1, cmap='turbo', s=9.0)
    ax.set_xlabel('Cross-elevation [deg]')
    ax.set_ylabel('Co-elevation [deg]')
    ax.set_aspect('equal')
    return im


def plot_sunscan_simulation(simulator:SunSimulator, gamma, omega, time, signal_original, gammav, omegav, sky):
    sun_azi, sun_elv=sky.compute_sun_location(time)
    signal_normed = norm_signal(signal_original)
    starttime = pd.to_datetime(time.min())
    sun_sim= simulator.forward_sun(gamma, omega, sun_azi, sun_elv, gammav, omegav)
    simulator_noback=copy.copy(simulator)
    simulator_noback.dtime=0
    simulator_noback.backlash=0

    sun_pos_noback = simulator_noback.get_sunpos_tangential(gamma, omega, sun_azi, sun_elv, gammav, omegav)
    sun_sim_noback = simulator_noback.forward_sun(gamma, omega, sun_azi, sun_elv, gammav, omegav)

    sun_pos_corrected = simulator.get_sunpos_tangential(gamma, omega, sun_azi, sun_elv, gammav, omegav)
    plane_full_x = xr.DataArray(np.linspace(sun_pos_corrected.isel(row=0).min().item(),
                                sun_pos_corrected.isel(row=0).max().item(), 100), dims='plane_x')
    plane_full_y = xr.DataArray(np.linspace(sun_pos_corrected.isel(row=1).min().item(),
                                sun_pos_corrected.isel(row=1).max().item(), 100), dims='plane_y')
    plane_full_x, plane_full_y = xr.broadcast(plane_full_x, plane_full_y)
    sim_full = simulator._lookup(xr.concat([plane_full_x, plane_full_y], dim='row'))

    #
    fig, axs = plt.subplots(3, 2, figsize=(8, 12))  # , layout='tight')
    ax = axs[0, 0]

    def plot_points_gammaomega(gamma, omega, c, ax):
        im = ax.scatter(gamma, omega, c=c, cmap='turbo', s=13)
        ax.set_xlabel('Gamma ("Azimuth axis") [deg]')
        ax.set_ylabel('Omega ("Elevation axis") [deg]')
        return im
    im = plot_points_gammaomega(gamma, omega, signal_original, ax)
    fig.colorbar(im, ax=ax, label='Signal strength [dB]')
    ax = axs[0, 1]
    im = plot_points_gammaomega(gamma, omega, sun_sim, ax)
    # remove y tick labels
    ax.set_yticklabels([])
    ax.set_ylabel('')
    fig.colorbar(im, ax=ax, label='Simulated signal [normalized]')

    # Plot measurements and simulation with the uncorrected tangent plane positions
    ax = axs[1, 0]
    im = _plot_points_tangent_plane(sun_pos_noback, signal_normed, ax)
    ax = axs[1, 1]
    im = _plot_points_tangent_plane(sun_pos_noback, sun_sim_noback, ax)
    ax.set_yticklabels([])
    ax.set_ylabel('')

    # scanner_azi_corrected, scanner_elv_corrected = sunproc.identity_radar_model(
    #     ds.gamma+plot_params[0], ds.omega+plot_params[1])
    # # regardless of the anchor point, if we add the correction to the anchor point, we should get the center of the sun, therefore we simply take the mean
    # sun_center_corrected = _get_tangential_coords(
    #     ds.scanner_azi, ds.scanner_elv, scanner_azi_corrected, scanner_elv_corrected).mean('time_m')
    # ax.axvline(sun_center_corrected.sel(row=0).item(), color='grey', linestyle='--')
    # ax.axhline(sun_center_corrected.sel(row=1).item(), color='grey', linestyle='--')

    ax = axs[2, 0]
    im = _plot_points_tangent_plane(sun_pos_corrected, signal_normed, ax)
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
    params=simulator.get_params()
    fig.suptitle(f"{starttime.strftime('%Y-%m-%d %H:%M')}\n"+"\n".join([f"{k}: {v:.4f}" for k, v in params.items()]), fontsize='small')
    ax.set_aspect('equal')

    axs[0, 0].annotate('Scanner Coordinates', xy=(-0.4, 0.5), xycoords='axes fraction',
                       ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    axs[1, 0].annotate('Beam-centered coords; w/o backlash/dtime', xy=(-0.35, 0.5), xycoords='axes fraction', ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    axs[2, 0].annotate('Beam-centered coords', xy=(-0.35, 0.5), xycoords='axes fraction',
                       ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    # add column labels on the top
    axs[0, 0].annotate('Measurement', xy=(0.5, 1.05), xycoords='axes fraction',
                       ha='center', va='center', fontsize=12, fontweight='bold')
    axs[0, 1].annotate('Simulation', xy=(0.5, 1.05), xycoords='axes fraction',
                       ha='center', va='center', fontsize=12, fontweight='bold')
    return fig, axs