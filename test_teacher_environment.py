"""Integration checks for the instructor-provided simulation environment."""

import json
import unittest
from pathlib import Path

import numpy as np

from perception import extract_features
from savetheprey import EpisodeLayout, SaveThePrey


ROOT = Path(__file__).resolve().parent


class TeacherEnvironmentTests(unittest.TestCase):
    def setUp(self):
        data = json.loads((ROOT / "configs" / "official_test_layout.json").read_text(encoding="utf-8"))
        value = data["layouts"][0]
        self.layout = EpisodeLayout(
            prey=np.asarray(value["prey"], dtype=np.float64),
            predator=np.asarray(value["predator"], dtype=np.float64),
            shelter=np.asarray(value["shelter"], dtype=np.float64),
            obstacles=np.asarray(value["obstacles"], dtype=np.float64),
            name=value["name"],
        )
        self.env = SaveThePrey(seed=0, max_steps=10)

    def tearDown(self):
        self.env.close()

    def test_teacher_renderer_produces_features_at_official_coordinates(self):
        image = self.env.reset(self.layout)
        features, prey = extract_features(image, dt=self.env.dt)

        self.assertEqual(image.shape, (800, 800, 3))
        self.assertEqual(features.shape, (16,))
        np.testing.assert_allclose(self.env.predator, [0.0, 0.0])
        np.testing.assert_allclose(prey, [-0.5, 0.5], atol=0.015)
        np.testing.assert_allclose(self.env.shelter, [0.5, -0.5])

    def test_continuous_policy_action_advances_teacher_world_once(self):
        self.env.reset(self.layout)
        before = self.env.prey.copy()

        image, status = self.env.step(np.asarray([1.0, 0.0]))

        self.assertEqual(self.env.steps, 1)
        self.assertEqual(status, "running")
        self.assertEqual(image.shape, (800, 800, 3))
        self.assertTrue(np.all(np.isfinite(self.env.prey)))
        self.assertGreater(float(np.linalg.norm(self.env.prey - before)), 0.0)

    def test_reset_does_not_mutate_the_requested_layout(self):
        original_prey = self.layout.prey.copy()
        self.env.reset(self.layout)
        self.env.step(np.asarray([1.0, 0.0]))
        self.env.reset(self.layout)

        np.testing.assert_array_equal(self.layout.prey, original_prey)
        np.testing.assert_allclose(self.env.prey, original_prey)


if __name__ == "__main__":
    unittest.main()
