import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
import pyvista as pv
import os
from datetime import datetime


NUM_PARTICLES = 5
OUTPUT_DIR = "data"


@dataclass
class Particle:
    position: np.ndarray  # 3D position of the particle
    velocity: np.ndarray  # 3D velocity of the particle
    charge: float  # Charge of the particle (positive/negative/neutral)
    mass: float  # Mass of the particle

    def __post_init__(self):
        self.path = [self.position.copy()]

    def compute_acceleration(self, magnetic_field):
        # Calculate the Lorentz force
        force = self.charge * np.cross(self.velocity, magnetic_field)
        return force / self.mass

    def update(self, magnetic_field, time_step):
        acceleration = self.compute_acceleration(magnetic_field)
        self.velocity += acceleration * time_step
        self.position += self.velocity * time_step
        self.path.append(self.position.copy())


def generate_random_velocity(magnitude: float) -> np.ndarray:
    # Generate random spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
    phi = np.random.uniform(0, 2 * np.pi)  # Polar angle

    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Scale to the desired magnitude
    direction = np.array([x, y, z])
    return direction * magnitude

def save_particle_paths(filepath, paths):
    # Create a dictionary to hold paths
    path_dict = {f'path_{i}': np.array(path) for i, path in enumerate(paths)}
    
    # Save the dictionary as an npz file
    np.savez(filepath, **path_dict)



if __name__ == "__main__":

    time_step = 0.1

    particles = []
    for _ in range(NUM_PARTICLES):
        position = np.zeros(3)

        velocity = generate_random_velocity(1e8)

        particle = Particle(
            position=position, velocity=velocity, charge=-1.6e-19, mass=9.11e-31
        )
        particles.append(particle)

    magnetic_field = np.array([0.0, 0.0, 1.0])
    time_step = 1e-12

    path = []

    for _ in range(100):
        for particle in particles:
            particle.update(magnetic_field, time_step)

    paths = [p.path for p in particles]
    flattened_list = [array for sublist in paths for array in sublist]
    pv.plot(flattened_list)

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get the current timestamp and format it
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a unique filename with the timestamp
    filename = f"lorentz_{timestamp}.npz"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Convert particle paths into a dictionary-like format
    save_particle_paths(filepath, paths)
