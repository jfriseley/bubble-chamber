import argparse
import numpy as np
import pyvista as pv
import os
from datetime import datetime
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from bubblelib import PinholeCamera

DEBUG = False
SENSOR_WIDTH = 0.0036#0.036  # Full-frame sensor width
SENSOR_HEIGHT = 0.0024#0.024  # Full-frame sensor height
HORIZONTAL_RESOLUTION = 3600
OUTPUT_DIR='out'

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

def interpolate_path(path, num_points=10):
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

    camera_position_container = [None]

    flattened_points = [point for sublist in paths for point in sublist]
    plot_points(flattened_points, camera_position_container)

    position, focal_point, up_vector = camera_position_container[0]

    print(f"Camera position:{position}, focal_point: {focal_point}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    interpolated_paths = [] 
    for path in paths:
        interpolated_paths.append(interpolate_path(path, num_points=10000))

    camera_position = np.array(position)
    focal_point = np.array(focal_point)
    up_vector = np.array(up_vector)

    camera = PinholeCamera(position, focal_point, up_vector, SENSOR_WIDTH, SENSOR_HEIGHT, HORIZONTAL_RESOLUTION)

    width, height = camera.resolution
    background = Image.new("RGBA", (width, height), (0, 0, 139, 255))
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))  # Fully transparent overlay
    draw = ImageDraw.Draw(overlay)

    rasterised_points = []
    for path in interpolated_paths:
        for point in path:
            rasterised_point =camera.photograph_point(point)
            if rasterised_point is not None:
                rasterised_points.append(rasterised_point)


    # if DEBUG:
    #     camera_points = np.array(camera_points)
    #     ax.scatter(camera_points[:, 0], camera_points[:, 1], camera_points[:, 2], color='b')    
    #     # X axis (red)
    #     ax.quiver(0, 0, 0, 0.01, 0, 0, color='r', arrow_length_ratio=0.1)
    #     # Y-axis (Green)
    #     ax.quiver(0, 0, 0, 0, 0.01, 0, color='g', arrow_length_ratio=0.1)
    #     # Z-axis (Blue)
    #     ax.quiver(0, 0, 0, 0, 0, 0.01, color='b', arrow_length_ratio=0.1)
    #     plt.savefig('points_in_camera_frame')
    #     plt.show()

    
    crisp_radius = 2
    for point in rasterised_points:
        clear = (point[0], point[1])

        # Draw multiple circles for fuzziness
        draw.ellipse([clear[0]-crisp_radius, clear[1]-crisp_radius, clear[0]+crisp_radius, clear[1]+crisp_radius], fill=(230, 255, 255, 128))
        
    for point in rasterised_points:
        fuzzy_radius = int(np.random.uniform(1,5))
        fuzzy_point = (point[0], point[1])
        for r in range(fuzzy_radius):
            # Draw multiple circles for fuzziness
            draw.ellipse([fuzzy_point[0]-r, fuzzy_point[1]-r, fuzzy_point[0]+r, fuzzy_point[1]+r], fill=(0, 255, 255, int(255 * (1 - r / fuzzy_radius))))
        
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get the current timestamp and format it
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a unique filename with the timestamp
    filename = f"pinhole_{timestamp}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Convert particle paths into a dictionary-like format
    blended = Image.alpha_composite(background, overlay)
    blended.save(filepath)