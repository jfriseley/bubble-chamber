import argparse
import numpy as np
import pyvista as pv
import sys
from datetime import datetime
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DEBUG = False
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


def compute_camera_basis(position, focal_point, up_vector):
    z_c = (position - focal_point) / np.linalg.norm(position - focal_point)
    x_c = np.cross(up_vector, z_c) / np.linalg.norm(np.cross(up_vector, z_c))
    y_c = up_vector #np.cross(z_c, x_c)
    return x_c, y_c, z_c

# def world_to_camera(world_point, camera_position, camera_basis):
#     x_c, y_c, z_c = camera_basis
#     R = np.vstack([x_c, y_c, z_c]).T  # Rotation matrix
#     return R @ (world_point - camera_position)
#     
def world_to_camera(world_point, camera_position, camera_basis):
    # Convert the world point to homogeneous coordinates
    world_point_homogeneous = np.append(world_point, 1)  # Add 1 for homogeneous coordinate
    
    x_c, y_c, z_c = camera_basis
    
    # Create the rotation matrix (3x3) from the camera basis
    R = np.column_stack((x_c, y_c, z_c))  # Rotation matrix (3x3)
    
    # Create the translation vector (3x1)
    T = camera_position  
    
    # Create the full transformation matrix (4x4)
    transformation_matrix = np.eye(4)  # Initialize as an identity matrix
    transformation_matrix[:3, :3] = R  # Set the rotation part
    transformation_matrix[:3, 3] = T  # Set the translation part

    # Multiply the homogeneous world point by the transformation matrix
    camera_point_homogeneous = np.linalg.inv(transformation_matrix) @ world_point_homogeneous
    
    return list(camera_point_homogeneous[:3]/camera_point_homogeneous[3])  # Return only the 3D part

# 4. Convert from camera coordinates to image plane coordinates
def camera_to_image_plane(camera_point, focal_length):
    x, y, z = camera_point
    image_x = -(focal_length * x) / z
    image_y = -(focal_length * y) / z
    return image_x, image_y

# 5. Convert from image plane coordinates to pixel coordinates
def image_to_pixel(image_point, resolution, sensor_width, sensor_height):
    image_x, image_y = image_point
    width_px, height_px = resolution
    pixel_x = (image_x + sensor_width / 2) * (width_px / sensor_width)
    pixel_y = (sensor_height / 2 - image_y) * (height_px / sensor_height)
    return int(pixel_x), int(pixel_y)

def undo_pinhole_camera_inversion(image):
    flipped_vertical = image.transpose(Image.FLIP_TOP_BOTTOM)
    corrected_image = flipped_vertical.transpose(Image.FLIP_LEFT_RIGHT)
    return corrected_image



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


    line_color = 'blue'
    fuzzy_color = 'black'
    electric_blue = '#00FFFF'  # Bright electric blue

    camera_position = np.array(position)
    focal_point = np.array(focal_point)
    up_vector = np.array(up_vector)

    # Compute the camera basis vectors
    focal_length = np.linalg.norm(focal_point - camera_position)

    # Image size
    aspect_ratio = SENSOR_WIDTH/SENSOR_HEIGHT
    image_size = (HORIZONTAL_RESOLUTION, int(HORIZONTAL_RESOLUTION/aspect_ratio)) 
    width, height = image_size
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    pixel_coords = []

    camera_basis = compute_camera_basis(camera_position, focal_point, up_vector)

    camera_points = []
    rasterised_points = []
    for path in interpolated_paths:
        for point in path:

            camera_point = world_to_camera(point, camera_position, camera_basis)
            camera_points.append(camera_point)
            if camera_point[2] <= 0:
                image_point = camera_to_image_plane(camera_point, focal_length)
                pixel_coords = image_to_pixel(image_point, image_size, SENSOR_WIDTH, SENSOR_HEIGHT)
                rasterised_points.append(pixel_coords)
            else:
                print(f"Skipping point behind camera")

    if DEBUG:
        camera_points = np.array(camera_points)
        ax.scatter(camera_points[:, 0], camera_points[:, 1], camera_points[:, 2], color='b')    
        # X axis (red)
        ax.quiver(0, 0, 0, 0.01, 0, 0, color='r', arrow_length_ratio=0.1)
        # Y-axis (Green)
        ax.quiver(0, 0, 0, 0, 0.01, 0, color='g', arrow_length_ratio=0.1)
        # Z-axis (Blue)
        ax.quiver(0, 0, 0, 0, 0, 0.01, color='b', arrow_length_ratio=0.1)
        plt.savefig('points_in_camera_frame')
        plt.show()


    for point in rasterised_points:
        draw.ellipse([point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1], fill='blue')

    #image = undo_pinhole_camera_inversion(image)
    image.save('interpolated_paths.png')