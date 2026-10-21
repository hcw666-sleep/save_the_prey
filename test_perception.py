"""Focused regression checks for both supported RGB render palettes."""

import unittest

import numpy as np
from PIL import Image, ImageDraw

from perception import extract_features
from savetheprey import IMAGE_SIZE, EpisodeLayout, SaveThePrey
from scripts import _layout_from_dict


def world_to_pixel(point):
    return (
        int(round((point[0] + 1) * (IMAGE_SIZE - 1) / 2)),
        int(round((1 - point[1]) * (IMAGE_SIZE - 1) / 2)),
    )


def draw_circle(draw, point, radius, color):
    x, y = world_to_pixel(point)
    r = int(round(radius * (IMAGE_SIZE - 1) / 2))
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


class PerceptionCompatibilityTests(unittest.TestCase):
    def make_frame(self, palette, obstacle_positions):
        frame = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, IMAGE_SIZE - 1, IMAGE_SIZE - 1),
                       outline=(204, 102, 102), width=2)
        draw_circle(draw, (-0.72, -0.65), 0.025, palette["prey"])
        draw_circle(draw, (0.62, -0.68), 0.025, palette["predator"])
        if palette.get("square_shelter"):
            x, y = world_to_pixel((0.70, 0.72))
            draw.rectangle((x - 10, y - 10, x + 10, y + 10), fill=palette["shelter"])
        else:
            draw_circle(draw, (0.70, 0.72), 0.05, palette["shelter"])
        for point in obstacle_positions:
            draw_circle(draw, point, 0.05, palette["obstacle"])
        return np.asarray(frame, dtype=np.uint8)

    def test_self_renderer_palette(self):
        palette = {
            "prey": (89, 217, 89), "predator": (217, 89, 89),
            "shelter": (204, 153, 204), "obstacle": (64, 64, 64),
        }
        image = self.make_frame(palette, [(-0.55, 0.52), (-0.35, 0.52), (0.1, -0.35)])
        features, prey = extract_features(image)
        self.assertEqual(features.shape, (16,))
        np.testing.assert_allclose(prey, [-0.72, -0.65], atol=0.003)
        np.testing.assert_allclose(prey + features[2:4] * 2, [0.62, -0.68], atol=0.003)

    def test_original_pyglet_blended_palette_and_row_obstacles(self):
        palette = {
            "prey": (172, 236, 172), "predator": (236, 172, 172),
            "shelter": (204, 153, 204), "obstacle": (63, 63, 63),
            "square_shelter": True,
        }
        image = self.make_frame(palette, [(-0.55, 0.52), (-0.35, 0.52), (0.1, -0.35)])
        features, prey = extract_features(image)
        self.assertEqual(features.shape, (16,))
        np.testing.assert_allclose(prey, [-0.72, -0.65], atol=0.003)
        np.testing.assert_allclose(prey + features[2:4] * 2, [0.62, -0.68], atol=0.003)

    def test_vertical_obstacle_arrangement(self):
        palette = {
            "prey": (172, 236, 172), "predator": (236, 172, 172),
            "shelter": (204, 153, 204), "obstacle": (63, 63, 63),
            "square_shelter": True,
        }
        image = self.make_frame(palette, [(0.1, 0.65), (0.1, 0.35), (0.1, -0.35)])
        features, _ = extract_features(image)
        self.assertEqual(features.shape, (16,))

    def test_overlapping_obstacles_use_three_center_fallback(self):
        palette = {
            "prey": (172, 236, 172), "predator": (236, 172, 172),
            "shelter": (204, 153, 204), "obstacle": (63, 63, 63),
            "square_shelter": True,
        }
        image = self.make_frame(palette, [(-0.50, 0.50), (-0.42, 0.50), (0.10, -0.35)])
        features, _ = extract_features(image)
        centers = features[4:10].reshape(3, 2) * 2
        self.assertEqual(centers.shape, (3, 2))

    def test_default_obstacle_coordinates_are_preserved(self):
        palette = {
            "prey": (89, 217, 89), "predator": (217, 89, 89),
            "shelter": (204, 153, 204), "obstacle": (64, 64, 64),
        }
        image = self.make_frame(palette, [(-0.35, 0.35), (0.35, 0.35), (0.0, -0.35)])
        features, prey = extract_features(image)
        centers = prey + features[4:10].reshape(3, 2) * 2
        centers = centers[np.lexsort((centers[:, 0], -centers[:, 1]))]
        np.testing.assert_allclose(
            centers, np.asarray([[-0.35, 0.35], [0.35, 0.35], [0.0, -0.35]]),
            atol=0.004,
        )

    def test_missing_obstacle_pixels_degrade_without_crashing(self):
        palette = {
            "prey": (89, 217, 89), "predator": (217, 89, 89),
            "shelter": (204, 153, 204), "obstacle": (64, 64, 64),
        }
        image = self.make_frame(palette, [])
        features, _ = extract_features(image)
        self.assertEqual(features.shape, (16,))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_official_layout_rejects_invalid_coordinates(self):
        bad = {"prey": [2, 0], "predator": [0, 0], "shelter": [0, 0]}
        with self.assertRaisesRegex(ValueError, "within"):
            _layout_from_dict(bad, 0)
        bad = {"prey": [0.1], "predator": [0, 0], "shelter": [0, 0]}
        with self.assertRaisesRegex(ValueError, "two finite"):
            _layout_from_dict(bad, 0)


class PredatorObstacleTests(unittest.TestCase):
    def test_predator_stays_outside_obstacle_disks_while_chasing(self):
        env = SaveThePrey(max_steps=500)
        layout = EpisodeLayout(
            prey=np.asarray([-0.5, 0.5]),
            predator=np.asarray([0.0, 0.0]),
            shelter=np.asarray([0.5, -0.5]),
            obstacles=np.asarray([[-0.35, 0.35], [0.35, 0.35], [0.0, -0.35]]),
        )
        env.reset(layout)
        minimum_distance = env.obstacle_radius + env.predator_radius

        for _ in range(180):
            env.step(np.zeros(2))
            distances = np.linalg.norm(env.obstacles - env.predator, axis=1)
            self.assertTrue(np.all(distances >= minimum_distance - 1e-9))


if __name__ == "__main__":
    unittest.main()
