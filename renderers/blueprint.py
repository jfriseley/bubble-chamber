import argparse
import numpy as np
import os
from datetime import datetime
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


from bubblelib import PinholeCamera, load_paths, plot_points

DEBUG = False
SENSOR_WIDTH = 0.0036#0.036  # Full-frame sensor width
SENSOR_HEIGHT = 0.0024#0.024  # Full-frame sensor height
HORIZONTAL_RESOLUTION = 3600
OUTPUT_DIR='out'


def interpolate_path(path, num_points=10):
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

    paths = load_paths(args.file)

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


    crisp_radius = 2
    for point in rasterised_points:
        clear = (point[0], point[1])
        draw.ellipse([clear[0]-crisp_radius, clear[1]-crisp_radius, clear[0]+crisp_radius, clear[1]+crisp_radius], fill=(230, 255, 255, 200))
        
    for point in rasterised_points:
        fuzzy_radius = int(np.random.uniform(1,5))
        fuzzy_point = (point[0], point[1])
        for r in range(fuzzy_radius):
            # Draw multiple circles for fuzziness
            draw.ellipse([fuzzy_point[0]-r, fuzzy_point[1]-r, fuzzy_point[0]+r, fuzzy_point[1]+r], fill=(0, 255, 255, int(255 * (1 - r / fuzzy_radius))))
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"blueprint_{timestamp}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    blended = Image.alpha_composite(background, overlay)
    blended.save(filepath)
    print(f"Saved to {filepath}")