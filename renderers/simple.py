import argparse
import numpy as np
import pyvista as pv
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

def load_npz_file(file_path):
    """Loads an npz file and returns a list of lists of points."""
    data = np.load(file_path, allow_pickle=True)
    paths = []
    for key in data:
        paths.append(data[key])  
    return paths

def get_camera_position(plotter,camera_position_container):
    camera_position = plotter.camera_position
    print("Camera position:", camera_position)

    camera_position_container[0] = camera_position


def key_press_callback(plotter, key, camera_position_container):
    """Callback function triggered when a specific key is pressed."""
    if key == 's':  # Save the camera position when 's' is pressed
        get_camera_position(plotter, camera_position_container)

def plot_points(points, camera_position_container):
    """Displays the points in a PyVista plot window with a button for saving the camera position."""
    point_cloud = pv.PolyData(points)
    plotter = pv.Plotter()

    plotter.add_mesh(point_cloud, render_points_as_spheres=True, point_size=10)

    plotter.add_key_event("s", lambda: key_press_callback(plotter, 's', camera_position_container))
    plotter.show()

def interpolate_path(path, num_points=100):
    # Interpolate the path to create smooth transitions
    t = np.linspace(0, 1, len(path))
    interpolated_t = np.linspace(0, 1, num_points)
    try:
        interpolated_path = np.array([
            np.interp(interpolated_t, t, path[:, i]) for i in range(3)
        ]).T
    except Exception as e:
        print(f"Error interpolating path: {e}")
        return None
    return interpolated_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load an npz file and display points.")
    parser.add_argument("file", type=str, help="Path to the npz file containing the points")
    args = parser.parse_args()

    paths = load_npz_file(args.file)

    print(paths[1])

    camera_position_container = [None]

    flattened_points = [point for sublist in paths for point in sublist]
    plot_points(flattened_points, camera_position_container)

    print(camera_position_container)

    position, focal_point, view_up = camera_position_container[0]
    
    interpolated_paths = [] 
    for path in paths:
        interpolated_paths.append(interpolate_path(path, num_points=10000))

    # Calculate elevation and azimuth from camera position and focal point
    dx = focal_point[0] - position[0]
    dy = focal_point[1] - position[1]
    dz = focal_point[2] - position[2]

    elevation = np.degrees(np.arctan2(dz, np.sqrt(dx**2 + dy**2)))  # Elevation angle
    azimuth = np.degrees(np.arctan2(dy, dx))  # Azimuth angle

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set colors for paths (yellow and blue)
    colors = [cm.winter(i / len(paths)) for i in range(len(paths))]  # Blue to yellow colormap
    # Set colors for the paths
    line_color = 'blue'
    fuzzy_color = 'black'
    electric_blue = '#00FFFF'  # Bright electric blue

    # Plot each path
    for path in interpolated_paths:
        # Create a fuzzy line effect by plotting several black lines
        random_offsets = np.random.normal(0, 0.0001, size=path[:, 2].shape)  # Change the stddev to adjust fuzziness
        ax.plot(path[:, 0], path[:, 1], path[:, 2] + random_offsets, color=fuzzy_color, alpha=0.7, linewidth=4)  # Thicker black line
        
        # Plot the main blue line in the middle
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=line_color, alpha=1.0)

        # Plot the skinny electric blue line on top
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=electric_blue, alpha=1.0, linewidth=0.5)  # Skinny electric blue line


    all_paths = np.concatenate(interpolated_paths)
    x_limits = [all_paths[:, 0].min(), all_paths[:, 0].max()]
    y_limits = [all_paths[:, 1].min(), all_paths[:, 1].max()]
    z_limits = [all_paths[:, 2].min(), all_paths[:, 2].max()]

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_zlim(z_limits)
    # Set labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('Interpolated Particle Paths in Bubble Chamber')
    ax.set_axis_off()  # Hides the axes and ticks

    ax.grid(False)  # Hides the grid
    ax.view_init(elev=elevation, azim=azimuth)

    ax.set_facecolor((1.0, 0.8, 0.0))

    # Save the figure
    plt.savefig('interpolated_paths.png', bbox_inches='tight')
    plt.show()