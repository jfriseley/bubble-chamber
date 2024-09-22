from dataclasses import dataclass, field
from typing import Tuple, Union, List
import pyvista as pv
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D

def load_paths(file_path):
    """Loads an npz file and returns a list of lists of points."""
    data = np.load(file_path, allow_pickle=True)
    paths = []
    for key in data:
        paths.append(data[key])  
    return paths

def plot_points(points, camera_position_container):
    """Displays the points in a PyVista plot window with a button for saving the camera position."""
    point_cloud = pv.PolyData(points)
    plotter = pv.Plotter()

    plotter.add_mesh(point_cloud, render_points_as_spheres=True, point_size=10)

    plotter.add_key_event("s", lambda: key_press_callback(plotter, 's', camera_position_container))
    plotter.show()

def key_press_callback(plotter, key, camera_position_container):
    """Callback function triggered when a specific key is pressed."""
    if key == 's':  # Save the camera position when 's' is pressed
        get_camera_position(plotter, camera_position_container)

def get_camera_position(plotter,camera_position_container):
    camera_position = plotter.camera_position
    print("Camera position:", camera_position)
    camera_position_container[0] = camera_position


@dataclass
class PinholeCamera:
    position: np.ndarray
    focal_point: np.ndarray
    up_vector: np.ndarray
    sensor_width: float 
    sensor_height: float
    resolution_width: int
    focal_length: float = field(init=False)
    resolution: Tuple[int, int] = field(init=False)

    def __post_init__(self):
        self.focal_length = np.linalg.norm(self.focal_point - self.position)
        aspect_ratio = self.sensor_width/self.sensor_height
        self.resolution = (self.resolution_width, int(self.resolution_width/aspect_ratio)) 

    def _compute_basis(self):
        z_c = (self.position - self.focal_point) / np.linalg.norm(self.position - self.focal_point)
        x_c = np.cross(self.up_vector, z_c) / np.linalg.norm(np.cross(self.up_vector, z_c))
        y_c = self.up_vector 
        return x_c, y_c, z_c
    
    def _world_to_camera(self, world_point) -> List[float]:
        # Convert the world point to homogeneous coordinates

        camera_basis = self._compute_basis()
        world_point_homogeneous = np.append(world_point, 1)  # Add 1 for homogeneous coordinate
        
        x_c, y_c, z_c = camera_basis
        
        # Create the rotation matrix (3x3) from the camera basis
        R = np.column_stack((x_c, y_c, z_c))  # Rotation matrix (3x3)
        
        # Create the translation vector (3x1)
        T = self.position  
        
        # Create the full transformation matrix (4x4)
        transformation_matrix = np.eye(4)  # Initialize as an identity matrix
        transformation_matrix[:3, :3] = R  # Set the rotation part
        transformation_matrix[:3, 3] = T  # Set the translation part

        # Multiply the homogeneous world point by the transformation matrix
        camera_point_homogeneous = np.linalg.inv(transformation_matrix) @ world_point_homogeneous
        
        return list(camera_point_homogeneous[:3]/camera_point_homogeneous[3])  # Return only the 3D part

    def _camera_to_image_plane(self, camera_point) -> Tuple[float, float]:
        x, y, z = camera_point
        image_x = -(self.focal_length * x) / z
        image_y = -(self.focal_length * y) / z
        return image_x, image_y

    def _image_to_pixel(self, image_point) -> Tuple[int, int]:
        image_x, image_y = image_point
        width_px, height_px = self.resolution
        pixel_x = (image_x + self.sensor_width / 2) * (width_px / self.sensor_width)
        pixel_y = (self.sensor_height / 2 - image_y) * (height_px / self.sensor_height)
        return int(pixel_x), int(pixel_y)

    def photograph_point(self, world_point) -> Union[Tuple[int, int], None]:
        camera_point = self._world_to_camera(world_point)
        if camera_point[2] <= 0:
            image_point = self._camera_to_image_plane(camera_point)
            pixel_coords = self._image_to_pixel(image_point)
            return pixel_coords

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
