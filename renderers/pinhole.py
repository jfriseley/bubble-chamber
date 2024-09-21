import argparse
import numpy as np
import pyvista as pv
import sys
from datetime import datetime
from PIL import Image, ImageDraw


SENSOR_WIDTH = 0.036  # Full-frame sensor width
SENSOR_HEIGHT = 0.024  # Full-frame sensor height
HORIZONTAL_RESOLUTION = 2500

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

    camera_position_container = [None]

    flattened_points = [point for sublist in paths for point in sublist]
    plot_points(flattened_points, camera_position_container)

    position, focal_point, _ = camera_position_container[0]

    print(f"Camera position:{position}, focal_point: {focal_point}")

    
    interpolated_paths = [] 
    for path in paths:
        interpolated_paths.append(interpolate_path(path, num_points=10000))


    line_color = 'blue'
    fuzzy_color = 'black'
    electric_blue = '#00FFFF'  # Bright electric blue

    camera_position = np.array(position)
    focal_point = np.array(focal_point)

    # Compute the camera basis vectors
    camera_direction = np.array(focal_point) - np.array(camera_position)
    camera_direction = camera_direction/np.linalg.norm(camera_direction)
    focal_length = np.linalg.norm(focal_point - camera_position)

    # Image size
    aspect_ratio = SENSOR_WIDTH/SENSOR_HEIGHT
    image_size = (HORIZONTAL_RESOLUTION, int(HORIZONTAL_RESOLUTION/aspect_ratio)) 
    width, height = image_size
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    projected_points = []
    for path in interpolated_paths:
        for point in path:

            point_camera_space = point - camera_position

            d = np.dot(point_camera_space, camera_direction)

            if d != 0:  # Avoid division by zero
                scale = -focal_length / point_camera_space[2]
                #print(f"scale: {scale}")
                x_img = scale * point_camera_space[0]
                y_img = scale * point_camera_space[1]
                projected_points.append((x_img, y_img))


    image_width, image_height = image_size
    scale = -focal_length / point_camera_space[2]


    scale_x = image_width / SENSOR_WIDTH
    scale_y = image_height / SENSOR_HEIGHT
    rasterised_points = [
        (
            int((x_img*scale_x + image_width/2)),  
            int((image_height / 2) - (y_img * scale_y)) 
        ) for x_img, y_img in projected_points
    ]

    print(rasterised_points)

    for point in rasterised_points:
        draw.ellipse([point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2], fill='blue')

    image.save('interpolated_paths.png')