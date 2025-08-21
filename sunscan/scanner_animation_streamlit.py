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
    'gamma_offset': 0.0,
    'omega_offset': 0.0,
    'gamma': 0.0,
    'omega': 90.0,
    'flex': 0.0
}
slider_step=1.0

# Initialize session state for all sliders if not present
for key, val in slider_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

if st.sidebar.button('Reset All Sliders to Zero'):
    for key, val in slider_defaults.items():
        st.session_state[key] = val

st.sidebar.header("Chain Parameters")
alpha = st.sidebar.slider("Alpha (deg)", -30.0, 30.0, value=st.session_state['alpha'], step=slider_step, key='alpha')
delta = st.sidebar.slider("Delta (deg)", -30.0, 30.0, value=st.session_state['delta'], step=slider_step, key='delta')
gamma_offset = st.sidebar.slider("Azimuth Offset (deg)", -30.0, 30.0, value=st.session_state['gamma_offset'], step=slider_step, key='gamma_offset')
beta = st.sidebar.slider("Beta (deg)", -30.0, 30.0, value=st.session_state['beta'], step=slider_step, key='beta')
omega_offset = st.sidebar.slider("Elevation Offset (deg)", -30.0, 30.0, value=st.session_state['omega_offset'], step=slider_step, key='omega_offset')
epsilon = st.sidebar.slider("Epsilon (deg)", -30.0, 30.0, value=st.session_state['epsilon'], step=slider_step, key='epsilon')
flex = st.sidebar.slider("Flex (deg at 0 elevation)", -30.0, 30.0, value=st.session_state['flex'], step=slider_step, key='flex')

st.sidebar.header("Joint Positions")
gamma = st.sidebar.slider("Gamma ('Azimuth') (deg)", 0.0, 360.0, value=st.session_state['gamma'], step=slider_step, key='gamma')
omega = st.sidebar.slider("Omega ('Elevation') (deg)", 0.0, 180.0, value=st.session_state['omega'], step=slider_step, key='omega')

# Add plot_frames radio selector for instant update
st.sidebar.radio(
    "Show frames:",
    options=["all", "last"],
    key='plot_frames'
)

# Add dish plotting control
show_dish = st.sidebar.checkbox("Show Dish", value=True)

st.sidebar.header("Target Pointing")
# Calculate current pointing for default values
scanner_temp = GeneralScanner(gamma_offset=gamma_offset, omega_offset=omega_offset, alpha=alpha, delta=delta, beta=beta, epsilon=epsilon, dtime=0.0, backlash_gamma=0.0, flex=flex)
current_azi, current_elv = scanner_temp.forward(np.array([gamma]), np.array([omega]), gammav=0, omegav=0)

target_azi = st.sidebar.number_input("Target Azimuth (deg)", value=float(current_azi), step=slider_step, format="%.2f")
target_elv = st.sidebar.number_input("Target Elevation (deg)", value=float(current_elv), step=slider_step, format="%.2f")

# Check if target values differ from current pointing
use_target_pointing = abs(target_azi - float(current_azi)) > 0.01 or abs(target_elv - float(current_elv)) > 0.01

if use_target_pointing:
    try:
        # Use inverse kinematics to calculate required gamma/omega
        target_gamma, target_omega = scanner_temp.inverse(target_azi, target_elv)
        # Use calculated values instead of slider values
        gamma = float(target_gamma)
        omega = float(target_omega)
        
        # Calculate actually reached values with forward kinematics
        reached_azi, reached_elv = scanner_temp.forward(np.array([gamma]), np.array([omega]), gammav=0, omegav=0)
        
        # Check if we reached the target within tolerance
        azi_error = abs(float(reached_azi) - target_azi)
        elv_error = abs(float(reached_elv) - target_elv)
        
        if azi_error < 0.1 and elv_error < 0.1:
            st.sidebar.success(f"Target reached: γ={gamma:.1f}°, ω={omega:.1f}°")
        else:
            st.sidebar.warning(f"Target approximated: γ={gamma:.1f}°, ω={omega:.1f}°")
            st.sidebar.info(f"Actually reached: Az={float(reached_azi):.2f}° (Δ={azi_error:.2f}°), El={float(reached_elv):.2f}° (Δ={elv_error:.2f}°)")
            
    except Exception as e:
        st.sidebar.error(f"Could not reach target: {e}")
        # Fall back to slider values if inverse fails

scanner = GeneralScanner(gamma_offset=gamma_offset, omega_offset=omega_offset, alpha=alpha, delta=delta, beta=beta, epsilon=epsilon, dtime=0.0, backlash_gamma=0.0, flex=flex)
positions = scanner._get_joint_positions(np.array([gamma]), np.array([omega]), gammav=0, omegav=0)
last_frame = scanner.chain.forward_kinematics(positions[0])

# Get the last frame position and orientation
dish_center = last_frame[:3, 3]  # Translation vector (position)
dish_rotation = last_frame[:3, :3]  # Rotation matrix (orientation)

# Create parabolic dish geometry
dish_radius = 0.3  # Dish radius in meters
u = np.linspace(-dish_radius, dish_radius, 20)
v = np.linspace(-dish_radius, dish_radius, 20)
U, V = np.meshgrid(u, v)

# Create parabolic surface
parabola_depth = 0.15  # How deep the dish is
Z = parabola_depth * (U**2 + V**2) / dish_radius**2 - parabola_depth  # Parabola equation

# Mask to create circular dish
mask = U**2 + V**2 <= dish_radius**2
Z[~mask] = np.nan

# Transform dish coordinates to world frame
dish_points = np.stack([U, V, Z], axis=-1)  # Removed negative Z
dish_points_flat = dish_points.reshape(-1, 3)

# Apply rotation and translation
dish_points_world = np.full_like(dish_points_flat, np.nan)  # Initialize with NaN
valid_mask = ~np.isnan(dish_points_flat).any(axis=1)  # Find valid points

for i, point in enumerate(dish_points_flat):
    if valid_mask[i]:  # Only transform valid points
        dish_points_world[i] = dish_rotation @ point + dish_center

# Reshape back to grid
dish_points_world = dish_points_world.reshape(dish_points.shape)
X_dish = dish_points_world[:, :, 0]
Y_dish = dish_points_world[:, :, 1]
Z_dish = dish_points_world[:, :, 2]

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# Conditionally plot the parabolic dish
if show_dish:
    # Plot the parabolic dish
    ax.plot_wireframe(X_dish, Y_dish, Z_dish, color='grey', linewidth=0.5)
    
    # Add dish rim for better visualization
    theta = np.linspace(0, 2*np.pi, 50)
    rim_local = np.array([dish_radius * np.cos(theta), 
                          dish_radius * np.sin(theta), 
                          np.zeros(len(theta))]).T
    rim_world = np.array([dish_rotation @ point + dish_center for point in rim_local])
    ax.plot(rim_world[:, 0], rim_world[:, 1], rim_world[:, 2], color='grey', linewidth=2)

# Plot the chain
scanner.chain.plot(positions[0], ax)#, plot_frames=st.session_state['plot_frames'])

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([0, 4])
ax.set_xlabel('S <- X (m) -> N')
ax.set_ylabel('W <- Y (m) -> E')
ax.set_zlabel('Z (m)')
ax.set_box_aspect([1, 1, 1])
ax.set_title("Pan-Tilt Chain Visualization")

ax.invert_yaxis()
ax.view_init(elev=20, azim=-145)  # Rotate view to swap x/y appearance
# Calculate pointing vector using forward_pt_chain
x,y,z = scanner.forward_pointing(np.array([gamma]), np.array([omega]), gammav=0, omegav=0)
azi_deg, elv_deg = scanner.forward(np.array([gamma]), np.array([omega]), gammav=0, omegav=0)




st.header("Final Pointing Direction")
st.markdown(f"**Cartesian:** x = {x:.3f}, y = {y:.3f}, z = {z:.3f}")
st.markdown(f"**Spherical:** azimuth = {azi_deg:.2f}°, elevation = {elv_deg:.2f}°")

st.pyplot(fig)
