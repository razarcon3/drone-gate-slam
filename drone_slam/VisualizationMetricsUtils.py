import gtsam
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot  # or from plotly.offline import iplot if in Jupyter
import matplotlib.pyplot as plt
from BackEnd import get_corner_key
from typing import List

# ids 0-2
gt_gate_positions = np.array([
        [14.3, -0.2, 5.19],
        [26.8, -0.6, 7.19],
        [34.9, -0.3, 8.39]
    ])

def plot_graph_calc_metrics(values: gtsam.Values, drone_pose_count: int, landmark_count: int, gt_drone_positions: List, vio_drone_positions: List, objects_info_gt: dict):
    est_drone_positions = []
    est_corner_positions = []

    # get the positions of the drone
    for i in range(drone_pose_count):
        symbol = gtsam.symbol('x', i)
        drone_pose: gtsam.Pose3 = values.atPose3(symbol)
        est_drone_positions.append(drone_pose.translation())

    # get the corner positions
    for i in range(landmark_count):
        for j in range(4):
            symbol = get_corner_key(i, j)
            est_corner_positions.append(values.atPoint3(symbol))

    est_drone_positions = np.array(est_drone_positions)
    gt_drone_positions = np.array(gt_drone_positions)
    
    est_corner_positions = np.array(est_corner_positions)
    vio_drone_positions = np.array(vio_drone_positions)

    fig = go.Figure()

    # Plot drone positions (markers and lines)
    fig.add_trace(go.Scatter3d(x=est_drone_positions[:, 0], y=est_drone_positions[:, 1] * 0.001, z=est_drone_positions[:, 2],
                               mode='markers+lines',  # Show markers and connect with lines
                               marker=dict(size=2, color='blue'),  # Customize marker size and color
                               line=dict(color='blue'),  # Customize line color
                               name='Estimated Drone Position')) # Name for the legend

    # plot GT positions
    fig.add_trace(go.Scatter3d(x=gt_drone_positions[:, 0], y=gt_drone_positions[:, 1], z=gt_drone_positions[:, 2],
                               mode='markers+lines',  # Show markers and connect with lines
                               marker=dict(size=2, color='green'),  # Customize marker size and color
                               line=dict(color='green'),  # Customize line color
                               name='GT Drone Position')) # Name for the legend

    # plot keyframe positions
    fig.add_trace(go.Scatter3d(x=vio_drone_positions[:, 0], y=vio_drone_positions[:, 1], z=vio_drone_positions[:, 2],
                               mode='markers+lines',  # Show markers and connect with lines
                               marker=dict(size=2, color='red'),  # Customize marker size and color
                               line=dict(color='red'),  # Customize line color
                               name='VIO Drone Position')) # Name for the legend


    # Plot estimated landmark positions (markers)
    fig.add_trace(go.Scatter3d(x=est_corner_positions[:, 0], y=est_corner_positions[:, 1], z=est_corner_positions[:, 2],
                               mode='markers',
                               marker=dict(size=5, color='red', symbol='diamond'),  # Customize marker
                               name='Estimated Landmark Locations')) # Name for the legend

    est_gate_positions = []
    for i in range(landmark_count):
        corners = est_corner_positions[i * 4: i*4+4, :]
        est_gate_positions.append(np.mean(corners, axis=0))
    est_gate_positions = np.array(est_gate_positions)
    
    # plot estimated gate positions
    fig.add_trace(go.Scatter3d(x=est_gate_positions[:, 0], y=est_gate_positions[:, 1], z=est_gate_positions[:, 2],
                               mode='markers',
                               marker=dict(size=5, color='blue', symbol='diamond'),  # Customize marker
                               name='Estimated Gate Locations')) # Name for the legend
    
    # plot GT gate positions    
    fig.add_trace(go.Scatter3d(x=gt_gate_positions[:, 0], y=gt_gate_positions[:, 1], z=gt_gate_positions[:, 2],
                                mode='markers',
                                marker=dict(size=5, color='green', symbol='diamond'),  # Customize marker
                                name='GT Gate Locations')) # Name for the legend

    # Layout settings (optional)
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                                 yaxis_range=[-4,4]),
                      title="SLAM Output",
                      legend_title="Legend")  # Add a title to the legend

    # fig.show()
    plot(fig, filename='se_plot.html', include_plotlyjs=True)
    
    # Calculate Average Drone Positional Distance Error
    error_dronepos_dists_gt_est = np.linalg.norm(gt_drone_positions - est_drone_positions, axis=1)
    error_dronepos_dists_gt_vio = np.linalg.norm(gt_drone_positions - vio_drone_positions, axis=1)
    print(f"Avg Drone Positional Error (GT vs Est): {np.mean(error_dronepos_dists_gt_est)}")
    print(f"Avg Drone Positional Error (GT vs VIO): {np.mean(error_dronepos_dists_gt_vio)}")
    
    # Create the plot
    plt.figure(figsize=(12, 6)) # Optional: Adjust figure size for better readability

    # Plot the first error series
    plt.plot(np.arange(len(error_dronepos_dists_gt_est)), error_dronepos_dists_gt_est, label='Error (Ground Truth vs. Estimation)', color='blue', alpha=0.8)

    # Plot the second error series
    plt.plot(np.arange(len(error_dronepos_dists_gt_vio)), error_dronepos_dists_gt_vio, label='Error (Ground Truth vs. VIO)', color='red', alpha=0.8)

    # Add plot title and labels
    plt.title('Drone Positional Errors Over Time')
    plt.xlabel('Time Step (deciseconds)')
    plt.ylabel('Position Error (meters)') # Add units if known (e.g., meters)

    # Add a legend to identify the lines
    plt.legend()

    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # --- Save the Plot ---
    plot_filename = 'drone_position_residual_errors.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')

    error_gate_dists = np.linalg.norm(gt_gate_positions - est_gate_positions, axis=1)
    print(f"Avg Gate Positional Error: {np.mean(error_gate_dists)}")
    
    for i in range(landmark_count):
        print(f"Gate {i} positional error = {error_gate_dists[i]}")
    
    # Create graphs plotting positional distance error
    