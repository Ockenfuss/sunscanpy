import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sunscan.scanner import IdentityScanner, BacklashScanner
from sunscan.math_utils import geometric_slerp, spherical_to_xyz, cartesian_to_spherical
from sunscan.signal_simulation import SignalSimulator
from sunscan.utils import db_to_linear, linear_to_db

def _plot_points_tangent_plane(sun_pos_bc_x, sun_pos_bc_y, sun_signal, ax, vmin=0, vmax=1, cmap='turbo'):
    ax.axvline(x=0, color='k', linestyle='--')
    ax.axhline(y=0, color='k', linestyle='--')
    im = ax.scatter(sun_pos_bc_x, sun_pos_bc_y,
                    c=sun_signal, vmin=vmin, vmax=vmax, cmap=cmap, s=9.0)
    ax.set_xlabel('Cross-elevation [deg]')
    ax.set_ylabel('Co-elevation [deg]')
    ax.set_aspect('equal')
    return im


def plot_sunscan_simulation(simulator:SignalSimulator, gamma, omega, time, signal_db, gammav, omegav, sun, remove_outliers=False):
    sun_azi, sun_elv=sun.compute_sun_location(time)
    sun_azi, sun_elv = np.asarray(sun_azi), np.asarray(sun_elv)
    sun_pos_bc_x, sun_pos_bc_y = simulator.get_sunpos_beamcentered(gamma, omega, sun_azi, sun_elv, gammav, omegav)
    if remove_outliers:
        x_std= np.std(sun_pos_bc_x)
        x_mean= np.mean(sun_pos_bc_x)
        y_std= np.std(sun_pos_bc_y)
        y_mean= np.mean(sun_pos_bc_y)
        n_std=2
        is_outlier = (np.abs(sun_pos_bc_x-x_mean)>n_std*x_std) | (np.abs(sun_pos_bc_y-y_mean)>n_std*y_std)
        gamma= gamma[~is_outlier]
        omega= omega[~is_outlier]
        gammav= gammav[~is_outlier]
        omegav= omegav[~is_outlier]
        signal_db= signal_db[~is_outlier]
        time= time[~is_outlier]
        sun_pos_bc_x= sun_pos_bc_x[~is_outlier]
        sun_pos_bc_y= sun_pos_bc_y[~is_outlier]
        sun_azi= sun_azi[~is_outlier]
        sun_elv= sun_elv[~is_outlier]
    # signal_normed = norm_signal(signal_original)
    starttime = pd.to_datetime(time.min())
    sun_sim_lin= simulator.forward_sun(gamma, omega, sun_azi, sun_elv, gammav, omegav)
    sun_sim_db= linear_to_db(sun_sim_lin)
    params=simulator.get_params()

    plane_full_x = xr.DataArray(np.linspace(sun_pos_bc_x.min(),
                                sun_pos_bc_x.max(), 100), dims='plane_x')
    plane_full_y = xr.DataArray(np.linspace(sun_pos_bc_y.min(),
                                sun_pos_bc_y.max(), 100), dims='plane_y')
    plane_full_x, plane_full_y = xr.broadcast(plane_full_x, plane_full_y)
    # sun_contribution = simulator.lut.lookup(lx=plane_full_x, ly=plane_full_y, fwhm_x=simulator.fwhm_x, fwhm_y=simulator.fwhm_y, limb_darkening=simulator.limb_darkening)
    # sun_sim_linear= db_to_linear(self.sky_db)*(1-sun_contribution) + db_to_linear(self.sun_db)*sun_contribution
    # sun_sim_db= linear_to_db(sun_sim_linear)
    sim_full_lin = simulator.signal_from_bc_coords(plane_full_x, plane_full_y)
    sim_full_db= linear_to_db(sim_full_lin)

    #
    fig = plt.figure(figsize=(8, 12), layout='tight')
    
    # Create 4x4 grid layout with custom height ratios
    # Row 0: 2 plots (columns 0-1 and 2-3)
    # Row 1: 2 plots (columns 0-1 and 2-3)
    # Row 2: colorbar (columns 0-3, thin height)
    # Row 3: 1 centered plot (columns 1-2)
    
    # Use gridspec for better control over height ratios
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 0.1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0:2])  # Top left
    ax2 = fig.add_subplot(gs[0, 2:4])  # Top right
    ax3 = fig.add_subplot(gs[1, 0:2])  # Middle left
    ax4 = fig.add_subplot(gs[1, 2:4])  # Middle right
    ax_cbar = fig.add_subplot(gs[2, :])  # Colorbar row (full width, thin)
    ax5 = fig.add_subplot(gs[3, 1:3])  # Bottom center plot
    
    # Set aspect ratios
    ax1.set_aspect('auto')
    ax2.set_aspect('auto')
    ax3.set_aspect('equal')
    ax4.set_aspect('equal')
    ax5.set_aspect('equal')
    
    axs = [ax1, ax2, ax3, ax4, ax5]
    
    ax = ax1

    def plot_points_gammaomega(gamma, omega, c, ax, vmin, vmax):
        im = ax.scatter(gamma, omega, c=c, cmap='turbo', s=13, vmin=vmin, vmax=vmax)
        ax.set_xlabel('Gamma ("Azimuth axis") [deg]')
        ax.set_ylabel('Omega ("Elevation axis") [deg]')
        return im
    vmin= min(signal_db.min(), sun_sim_db.min())
    vmax= max(signal_db.max(), sun_sim_db.max())
    im = plot_points_gammaomega(gamma, omega, signal_db, ax, vmin=vmin, vmax=vmax)
    ax = ax2
    im = plot_points_gammaomega(gamma, omega, sun_sim_db, ax, vmin=vmin, vmax=vmax)
    # remove y tick labels
    ax.set_yticklabels([])
    ax.set_ylabel('')

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
    diff=signal_db-sun_sim_db
    quantiles=np.quantile(diff, [0.02, 0.98])
    im = _plot_points_tangent_plane(sun_pos_bc_x, sun_pos_bc_y, diff , ax, vmin=quantiles[0], vmax=quantiles[1], cmap='coolwarm')
    fig.colorbar(im, ax=ax, label='Measured - Simulated [dB]')

    ax = ax3
    im = _plot_points_tangent_plane(sun_pos_bc_x, sun_pos_bc_y, signal_db, ax, vmin=vmin, vmax=vmax)
    ax = ax4
    ax.pcolormesh(plane_full_x.values, plane_full_y.values, sim_full_db.values, cmap='turbo', alpha=0.2)
    contour_levels= np.asarray([0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99])
    contour_levels= vmin+ contour_levels*(vmax-vmin)
    ax.contour(plane_full_x.values, plane_full_y.values, sim_full_db.values, levels=contour_levels, cmap='turbo', linewidths=1)
    im = _plot_points_tangent_plane(sun_pos_bc_x, sun_pos_bc_y, sun_sim_db, ax, vmin=vmin, vmax=vmax)
    ax.set_yticklabels([])
    ax.set_ylabel('')

    # Create a single colorbar for the lower row
    fig.colorbar(im, cax=ax_cbar, orientation='horizontal', label='Signal Strength [dB]')
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


def _plot_fitting_points(ax, beam_azi, beam_elv, scanner_azi, scanner_elv, reverse, enhancement=1.0, plot_connectors=True):
    extrapolated=[geometric_slerp(spherical_to_xyz(scanner_azi[i], scanner_elv[i]), spherical_to_xyz(beam_azi[i], beam_elv[i]), enhancement) for i in range(len(beam_azi))]
    ext_azi, ext_elv=cartesian_to_spherical(np.array(extrapolated))

    art1=ax.scatter(np.deg2rad(scanner_azi), scanner_elv, color='blue', marker='o', s=5)
    # ax.scatter(identity_azi, identity_elv, color='red', marker='x', label='Position if identity model would be correct')
    art2=ax.scatter(np.deg2rad(ext_azi[~reverse]), ext_elv[~reverse], color='green', marker='x')
    art3=ax.scatter(np.deg2rad(ext_azi[reverse]), ext_elv[reverse], color='orange', marker='x')
    if plot_connectors:
        for i in range(len(scanner_azi)):
            ax.plot(np.deg2rad([scanner_azi[i], ext_azi[i]]), [scanner_elv[i], ext_elv[i]], color='grey', linewidth=0.5, linestyle='-')
    ax.invert_yaxis()
    ax.set_ylabel(r"$\theta$ [degrees]", rotation=45)
    ax.yaxis.set_label_coords(0.6, 0.6)
    ax.set_xlabel(r"$\phi$ [degrees]")
    ax.set_ylim(90,0)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    xticks={k: f"{k}º" for k in range(0, 360, 45)}
    xticks[0] = 'N'
    xticks[90] = 'E'
    xticks[180] = 'S'
    xticks[270] = 'W'
    ax.set_xticks(np.deg2rad(list(xticks.keys())), labels=list(xticks.values()))
    return [art1, art2, art3]

def plot_calibrated_pairs(gamma_s, omega_s, azi_beam, elv_beam, scanner_model=None, ax=None, gamma_offset=None, plot_connectors=None, enhancement=1.0):
    if ax is None:
        fig, _ax = plt.subplots(1, 1, subplot_kw={'polar': True}, figsize=(10,10))
    else:
        _ax = ax
    if plot_connectors is None:
        if gamma_offset is None and scanner_model is None:
            plot_connectors = False
        else:
            plot_connectors = True
    if scanner_model is None:
        if gamma_offset is not None:
            scanner_model=BacklashScanner(gamma_offset=gamma_offset, omega_offset=0.0, dtime=0.0, backlash_gamma=0.0, flex=0.0)
        else:
            scanner_model = IdentityScanner()


    reverse=omega_s>90
    scanner_azi, scanner_elv= scanner_model.forward(gamma_s, omega_s, gammav=0, omegav=0)
    artists=_plot_fitting_points(_ax, azi_beam, elv_beam, scanner_azi, scanner_elv, reverse, plot_connectors=plot_connectors, enhancement=enhancement)
    artists[0].set_label(r'Pointing from scanner position $M(\gamma_r, \omega_r)$')
    artists[1].set_label(r'Actual pointing position $\phi_r, \theta_r$ (forward scan)')
    artists[2].set_label(r'Actual pointing position $\phi_r, \theta_r$ (reverse scan)')
    # _ax.set_title("")
    # Create a single figure legend
    handles, labels = _ax.get_legend_handles_labels()  # Get handles and labels from one of the axes
    fig= _ax.get_figure()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=2)  # Adjust location and layout

    if ax is None:
        return fig, _ax