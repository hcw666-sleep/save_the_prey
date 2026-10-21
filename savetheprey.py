"""Headless, RGB-only Save the Prey coursework environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw


IMAGE_SIZE = 800
WORLD_LIMIT = 1.0
OBSTACLES = np.asarray([[-0.35, 0.35], [0.35, 0.35], [0.0, -0.35]], dtype=np.float64)
PREY_RGB = (89, 217, 89)
PREDATOR_RGB = (217, 89, 89)
SHELTER_RGB = (204, 153, 204)
OBSTACLE_RGB = (64, 64, 64)


@dataclass
class EpisodeLayout:
    prey: np.ndarray
    predator: np.ndarray
    shelter: np.ndarray
    obstacles: np.ndarray = field(default_factory=lambda: OBSTACLES.copy())
    name: str = "random"


class SaveThePrey:
    """Small deterministic physics model with an image-only policy interface."""

    dt = 0.1
    prey_radius = 0.025
    predator_radius = 0.025
    obstacle_radius = 0.05
    shelter_radius = 0.05
    prey_max_speed = 0.25
    predator_max_speed = 0.28
    prey_max_accel = 1.0
    predator_max_accel = 0.85
    capture_distance = prey_radius + predator_radius
    obstacle_distance = 0.075
    shelter_distance = prey_radius + shelter_radius

    def __init__(self, seed: int = 0, max_steps: int = 1200):
        self.rng = np.random.default_rng(seed)
        self.max_steps = int(max_steps)
        self.steps = 0
        self.reset()

    def sample_layout(self, name: str = "random") -> EpisodeLayout:
        obstacles = OBSTACLES.copy()

        def sample_point(clearance: float) -> np.ndarray:
            for _ in range(10000):
                point = self.rng.uniform(-0.88, 0.88, size=2)
                if np.all(np.linalg.norm(obstacles - point, axis=1) > clearance):
                    return point
            raise RuntimeError("Could not sample a non-overlapping initial point")

        prey = sample_point(0.11)
        predator = sample_point(0.12)
        while np.linalg.norm(predator - prey) < 0.18:
            predator = sample_point(0.12)
        shelter = sample_point(0.12)
        while np.linalg.norm(shelter - prey) < 0.16:
            shelter = sample_point(0.12)
        return EpisodeLayout(prey, predator, shelter, obstacles, name)

    def reset(self, layout: Optional[EpisodeLayout] = None) -> np.ndarray:
        self.layout = layout or self.sample_layout()
        self.prey = np.asarray(self.layout.prey, dtype=np.float64).copy()
        self.predator = np.asarray(self.layout.predator, dtype=np.float64).copy()
        self.shelter = np.asarray(self.layout.shelter, dtype=np.float64).copy()
        self.obstacles = np.asarray(self.layout.obstacles, dtype=np.float64).copy()
        self.prey_velocity = np.zeros(2, dtype=np.float64)
        self.predator_velocity = np.zeros(2, dtype=np.float64)
        self.steps = 0
        self.status = "running"
        return self.render()

    @staticmethod
    def _to_pixel(point: np.ndarray) -> tuple[int, int]:
        x = int(round((float(point[0]) + WORLD_LIMIT) * (IMAGE_SIZE - 1) / 2))
        y = int(round((WORLD_LIMIT - float(point[1])) * (IMAGE_SIZE - 1) / 2))
        return x, y

    @staticmethod
    def _pixel_radius(world_radius: float) -> int:
        return max(1, int(round(world_radius * (IMAGE_SIZE - 1) / 2)))

    def render(self) -> np.ndarray:
        image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, IMAGE_SIZE - 1, IMAGE_SIZE - 1), outline=(204, 102, 102), width=2)
        for point in self.obstacles:
            cx, cy = self._to_pixel(point)
            r = self._pixel_radius(self.obstacle_radius)
            draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=OBSTACLE_RGB)
        cx, cy = self._to_pixel(self.shelter)
        r = self._pixel_radius(self.shelter_radius)
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=SHELTER_RGB)
        cx, cy = self._to_pixel(self.predator)
        r = self._pixel_radius(self.predator_radius)
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=PREDATOR_RGB)
        cx, cy = self._to_pixel(self.prey)
        r = self._pixel_radius(self.prey_radius)
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=PREY_RGB)
        return np.asarray(image, dtype=np.uint8)

    @staticmethod
    def _limited(vector: np.ndarray, magnitude: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm > magnitude and norm > 0:
            return vector * (magnitude / norm)
        return vector

    def step(self, acceleration: np.ndarray) -> tuple[np.ndarray, str]:
        if self.status != "running":
            return self.render(), self.status
        action = self._limited(np.asarray(acceleration, dtype=np.float64).reshape(2), self.prey_max_accel)
        predator_delta = self.prey - self.predator
        predator_action = predator_delta * 3.0 - self.predator_velocity * 1.5
        # Steer around obstacles before contact; the contact resolver below is
        # a final geometric guard against penetration at the simulation step.
        predator_clearance = self.obstacle_radius + self.predator_radius
        for obstacle in self.obstacles:
            offset = self.predator - obstacle
            distance = float(np.linalg.norm(offset))
            avoid_distance = predator_clearance + 0.12
            if distance < avoid_distance:
                if distance < 1e-8:
                    offset = np.asarray([1.0, 0.0])
                strength = (avoid_distance - distance) / 0.12
                predator_action += offset / distance * (1.2 * strength)
        predator_action = self._limited(predator_action, self.predator_max_accel)
        self.prey_velocity = self._limited(self.prey_velocity + action * self.dt, self.prey_max_speed)
        self.predator_velocity = self._limited(self.predator_velocity + predator_action * self.dt,
                                               self.predator_max_speed)
        self.prey += self.prey_velocity * self.dt
        self.predator = self._move_predator_without_penetration(
            self.predator + self.predator_velocity * self.dt
        )
        self.steps += 1

        if np.linalg.norm(self.prey - self.predator) < self.capture_distance:
            self.status = "captured"
        elif np.any(np.linalg.norm(self.obstacles - self.prey, axis=1) < self.obstacle_distance):
            self.status = "obstacle_collision"
        elif np.any(np.abs(self.prey) > 0.975):
            self.status = "out_of_bounds"
        elif np.linalg.norm(self.prey - self.shelter) < self.shelter_distance:
            self.status = "success"
        elif self.steps >= self.max_steps:
            self.status = "timeout"
        return self.render(), self.status

    def _move_predator_without_penetration(self, position: np.ndarray) -> np.ndarray:
        """Keep the predator outside obstacle disks and remove inward velocity."""
        minimum_distance = self.obstacle_radius + self.predator_radius
        position = np.asarray(position, dtype=np.float64).copy()
        # A few passes handle the rare case where corrections meet two disks.
        for _ in range(3):
            corrected = False
            for obstacle in self.obstacles:
                offset = position - obstacle
                distance = float(np.linalg.norm(offset))
                if distance < minimum_distance:
                    normal = offset / distance if distance > 1e-8 else np.asarray([1.0, 0.0])
                    position = obstacle + normal * minimum_distance
                    inward_speed = float(np.dot(self.predator_velocity, normal))
                    if inward_speed < 0:
                        self.predator_velocity -= inward_speed * normal
                    corrected = True
            if not corrected:
                break
        return position
