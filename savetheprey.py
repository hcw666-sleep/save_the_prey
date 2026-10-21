"""Image-only Save the Prey controller running on the instructor environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from multiagent.environment import MultiAgentEnv
from multiagent.scenarios.simple_tag import Scenario


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
    """Compatibility wrapper preserving the image-based policy and CLI API."""

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
    obstacle_distance = obstacle_radius + prey_radius
    shelter_distance = prey_radius + shelter_radius

    def __init__(self, seed: int = 0, max_steps: int = 1200):
        self.rng = np.random.default_rng(seed)
        self.max_steps = int(max_steps)
        self.scenario = Scenario()
        self.world = self.scenario.make_world()
        self.teacher_env = MultiAgentEnv(
            self.world,
            reset_callback=self.scenario.reset_world,
            reward_callback=self.scenario.reward,
            observation_callback=self.scenario.observation,
            done_callback=self.scenario.is_done,
            shared_viewer=True,
        )
        self.steps = 0
        self.reset()
        # The policy needs the instructor renderer's framebuffer, but Tkinter
        # is the sole user-facing animation window.
        for viewer in self.teacher_env.viewers:
            if viewer is not None:
                viewer.window.set_visible(False)

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

    @property
    def prey(self) -> np.ndarray:
        return self.world.agents[1].state.p_pos

    @property
    def predator(self) -> np.ndarray:
        return self.world.agents[0].state.p_pos

    @property
    def prey_velocity(self) -> np.ndarray:
        return self.world.agents[1].state.p_vel

    @property
    def predator_velocity(self) -> np.ndarray:
        return self.world.agents[0].state.p_vel

    @property
    def shelter(self) -> np.ndarray:
        return self.world.check[0].state.p_pos

    @property
    def obstacles(self) -> np.ndarray:
        return np.asarray([landmark.state.p_pos for landmark in self.world.landmarks])

    def reset(self, layout: Optional[EpisodeLayout] = None) -> np.ndarray:
        self.layout = layout or self.sample_layout()
        agent_pos = np.asarray(
            [self.layout.predator, self.layout.prey], dtype=np.float64
        ).copy()
        self.teacher_env.reset(
            agent_pos,
            np.asarray(self.layout.shelter, dtype=np.float64).copy(),
            np.asarray(self.layout.obstacles, dtype=np.float64).copy(),
        )
        for agent in self.world.agents:
            agent.action.u = np.zeros(2, dtype=np.float64)
        self.steps = 0
        self.status = "running"
        return self.render()

    def render(self) -> np.ndarray:
        image = self.teacher_env.render("rgb_array")[0]
        if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise RuntimeError(f"Instructor renderer returned unexpected image shape {image.shape}")
        return image

    def close(self) -> None:
        for viewer in self.teacher_env.viewers:
            if viewer is not None:
                viewer.close()
        self.teacher_env.viewers = [None] * len(self.teacher_env.viewers)

    @staticmethod
    def _limited(vector: np.ndarray, magnitude: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm > magnitude and norm > 0:
            return vector * (magnitude / norm)
        return vector

    def step(self, acceleration: np.ndarray) -> tuple[np.ndarray, str]:
        if self.status != "running":
            return self.render(), self.status

        prey_action = self._limited(
            np.asarray(acceleration, dtype=np.float64).reshape(2), self.prey_max_accel
        )
        predator_delta = self.prey - self.predator
        predator_action = predator_delta * 3.0 - self.predator_velocity * 1.5
        predator_clearance = self.obstacle_radius + self.predator_radius
        for obstacle in self.obstacles:
            offset = self.predator - obstacle
            distance = float(np.linalg.norm(offset))
            avoid_distance = predator_clearance + 0.12
            if distance < avoid_distance:
                if distance < 1e-8:
                    offset = np.asarray([1.0, 0.0])
                    distance = 1.0
                strength = (avoid_distance - distance) / 0.12
                predator_action += offset / distance * (1.2 * strength)

        self.world.agents[0].action.u = self._limited(
            predator_action, self.predator_max_accel
        )
        self.world.agents[1].action.u = prey_action
        self.world.step()
        self._keep_predator_out_of_obstacles()
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

    def _keep_predator_out_of_obstacles(self) -> None:
        minimum_distance = self.obstacle_radius + self.predator_radius
        predator = self.world.agents[0]
        for _ in range(3):
            corrected = False
            for obstacle in self.obstacles:
                offset = predator.state.p_pos - obstacle
                distance = float(np.linalg.norm(offset))
                if distance < minimum_distance:
                    normal = offset / distance if distance > 1e-8 else np.asarray([1.0, 0.0])
                    predator.state.p_pos = obstacle + normal * minimum_distance
                    inward_speed = float(np.dot(predator.state.p_vel, normal))
                    if inward_speed < 0:
                        predator.state.p_vel -= inward_speed * normal
                    corrected = True
            if not corrected:
                break
