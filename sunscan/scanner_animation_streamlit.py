import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D plots
from sunscan.scanner import GeneralScanner

st.title("Pan-Tilt Chain Visualizer")

# Define default values for all sliders
slider_defaults = {
    'alpha': 0.0,
    'delta': 0.0,
    'beta': 0.0,
    'epsilon': 0.0,
    'azi_offset': 0.0,
    'elv_offset': 0.0,
    'gamma': 0.0,
    'omega': 90.0
}

# Initialize session state for all sliders if not present
for key, val in slider_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

if st.sidebar.button('Reset All Sliders to Zero'):
    for key, val in slider_defaults.items():
        st.session_state[key] = val

st.sidebar.header("Chain Parameters")
alpha = st.sidebar.slider("Alpha (deg)", -30.0, 30.0, value=st.session_state['alpha'], step=0.1, key='alpha')
delta = st.sidebar.slider("Delta (deg)", -30.0, 30.0, value=st.session_state['delta'], step=0.1, key='delta')
azi_offset = st.sidebar.slider("Azimuth Offset (deg)", -30.0, 30.0, value=st.session_state['azi_offset'], step=0.1, key='azi_offset')
beta = st.sidebar.slider("Beta (deg)", -30.0, 30.0, value=st.session_state['beta'], step=0.1, key='beta')
elv_offset = st.sidebar.slider("Elevation Offset (deg)", -30.0, 30.0, value=st.session_state['elv_offset'], step=0.1, key='elv_offset')
epsilon = st.sidebar.slider("Epsilon (deg)", -30.0, 30.0, value=st.session_state['epsilon'], step=0.1, key='epsilon')

st.sidebar.header("Joint Positions")
gamma = st.sidebar.slider("Gamma ('Azimuth') (deg)", -180.0, 180.0, value=st.session_state['gamma'], step=0.1, key='gamma')
omega = st.sidebar.slider("Omega ('Elevation') (deg)", 0.0, 180.0, value=st.session_state['omega'], step=0.1, key='omega')

# Add plot_frames radio selector for instant update
st.sidebar.radio(
    "Show frames:",
    options=["all", "last"],
    key='plot_frames'
)

# Convert degrees to radians
alpha = np.deg2rad(alpha)
delta = np.deg2rad(delta)
beta = np.deg2rad(beta)
epsilon = np.deg2rad(epsilon)
azi_offset = np.deg2rad(azi_offset)
elv_offset = np.deg2rad(elv_offset)
gamma_rad = np.deg2rad(gamma)
omega_rad = np.deg2rad(omega)

scanner = GeneralScanner(azi_offset=azi_offset, elv_offset=elv_offset, alpha=alpha, delta=delta, beta=beta, epsilon=epsilon)
positions = [0, gamma_rad, 0, omega_rad, 0, 0]

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the chain
scanner.chain.plot(positions, ax, plot_frames=st.session_state['plot_frames'])
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([0, 4])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_box_aspect([1, 1, 1])
ax.set_title("Pan-Tilt Chain Visualization")

# Calculate pointing vector using forward_pt_chain
x,y,z = scanner.forward_pointing([gamma_rad], [omega_rad])[0]
azi, elv = scanner.forward([gamma_rad], [omega_rad])
azi_deg = np.rad2deg(azi[0])
elv_deg = np.rad2deg(elv[0])




st.header("Final Pointing Direction")
st.markdown(f"**Cartesian:** x = {x:.3f}, y = {y:.3f}, z = {z:.3f}")
st.markdown(f"**Spherical:** azimuth = {azi_deg:.2f}°, elevation = {elv_deg:.2f}°")

st.pyplot(fig)
