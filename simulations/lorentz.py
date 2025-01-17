import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
import pyvista as pv
import os
from datetime import datetime
import random

NUM_PARTICLES = 8
OUTPUT_DIR = "data"
SCALE=0.1

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

def generate_random_position(scale):
    return np.random.uniform(-scale / 2, scale / 2, 3)

def generate_random_velocity(magnitude: float) -> np.ndarray:
    # Generate random spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
    phi = np.arccos(np.random.uniform(-1, 1))  # Polar angle, using arccos to get uniform distribution

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


    particles = []
    velocity = np.array([-0.1, 0.11, 1])
    velocity = velocity / np.linalg.norm(velocity)
    velocity *= 1e8
    # generate electrons
    for _ in range(NUM_PARTICLES):
        position = generate_random_position(SCALE)

        particle = Particle(
            position=position, velocity=generate_random_velocity(random.uniform(1e7, 1e8)), charge=-1.6e-19, mass=9.11e-31
        )
        particles.append(particle)

    #generate protons
    for _ in range(NUM_PARTICLES):
        position = generate_random_position(SCALE)
        particle = Particle(
            position=position, velocity=velocity, charge=1.6e-19, mass=1.67e-27
        )
        particles.append(particle)

    # # generate helium nuclei
    # for _ in range(NUM_PARTICLES):
    #     position = generate_random_position(SCALE)

    #     particle = Particle(
    #         position=position, velocity=generate_random_velocity(random.uniform(1e7, 1e8)), charge=3.2e-19, mass=6.644e-27
    #     )
    #     particles.append(particle)

    #generate positrons
    for _ in range(NUM_PARTICLES):
        position = generate_random_position(SCALE)
        particle = Particle(
            position=position, velocity=generate_random_velocity(random.uniform(1e7, 1e8)), charge=1.6e-19, mass=9.11e-31
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
