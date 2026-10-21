"""NumPy MLP, image-derived teacher, and safe fused controller."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FEATURES = 16


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.zeros(2) if fallback is None else fallback.copy()
    return vector / norm


def teacher_action(features: np.ndarray) -> np.ndarray:
    """Potential-field demonstrator operating on the same image-only features."""
    f = np.asarray(features, dtype=np.float64)
    target = f[0:2] * 2
    predator = f[2:4] * 2
    obstacles = f[4:10].reshape(3, 2) * 2
    velocity = f[10:12] * 0.25
    position = (f[12:14] * 2) - 1
    direction = _unit(target)
    desired_velocity = direction * min(0.23, float(np.linalg.norm(target)) * 0.9)

    predator_distance = max(float(np.linalg.norm(predator)), 1e-5)
    if predator_distance < 0.42:
        closing = max(0.0, float(np.dot(velocity, _unit(predator))))
        strength = (0.42 - predator_distance) / 0.42
        desired_velocity -= _unit(predator) * (0.34 * strength + 0.12 * closing)

    for offset in obstacles:
        distance = max(float(np.linalg.norm(offset)), 1e-5)
        if distance < 0.24:
            away = -_unit(offset)
            strength = (0.24 - distance) / 0.24
            desired_velocity += away * (0.46 * strength)

    edge_margin = 0.90 - np.abs(position)
    for axis in range(2):
        if edge_margin[axis] < 0.2:
            desired_velocity[axis] += np.sign(-position[axis]) * (0.2 - edge_margin[axis]) * 1.6

    desired_velocity = desired_velocity / max(1.0, np.linalg.norm(desired_velocity) / 0.23)
    return np.clip((desired_velocity - velocity) / 0.1, -1.0, 1.0).astype(np.float32)


class MLP:
    def __init__(self, widths=(16, 64, 64, 2), seed=2026):
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(widths[:-1], widths[1:]):
            scale = np.sqrt(2 / (fan_in + fan_out))
            self.weights.append((rng.standard_normal((fan_in, fan_out)) * scale).astype(np.float32))
            self.biases.append(np.zeros(fan_out, dtype=np.float32))

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float32)
        for weight, bias in zip(self.weights, self.biases):
            values = np.tanh(values @ weight + bias)
        return values

    def save(self, path: Path, config: dict) -> None:
        arrays = {f"w{i}": w for i, w in enumerate(self.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        arrays["config_json"] = np.asarray(json.dumps(config, sort_keys=True))
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path):
        data = np.load(path, allow_pickle=False)
        layer_count = len([k for k in data.files if k.startswith("w") and k[1:].isdigit()])
        widths = [data["w0"].shape[0]] + [data[f"w{i}"].shape[1] for i in range(layer_count)]
        model = cls(widths=tuple(widths))
        model.weights = [data[f"w{i}"].astype(np.float32) for i in range(layer_count)]
        model.biases = [data[f"b{i}"].astype(np.float32) for i in range(layer_count)]
        model.config = json.loads(str(data["config_json"]))
        return model


def train_mlp(x: np.ndarray, y: np.ndarray, epochs=160, batch_size=256, learning_rate=0.002,
              seed=2026) -> tuple[MLP, list[float]]:
    rng = np.random.default_rng(seed)
    model = MLP(seed=seed)
    mw = [np.zeros_like(w) for w in model.weights]
    vw = [np.zeros_like(w) for w in model.weights]
    mb = [np.zeros_like(b) for b in model.biases]
    vb = [np.zeros_like(b) for b in model.biases]
    losses = []
    step = 0
    n = len(x)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            activations = [x[idx].astype(np.float32)]
            for weight, bias in zip(model.weights, model.biases):
                activations.append(np.tanh(activations[-1] @ weight + bias))
            delta = 2 * (activations[-1] - y[idx]) / len(idx)
            grad_w = [None] * len(model.weights)
            grad_b = [None] * len(model.biases)
            for i in reversed(range(len(model.weights))):
                grad_w[i] = activations[i].T @ delta
                grad_b[i] = delta.sum(axis=0)
                if i:
                    delta = (delta @ model.weights[i].T) * (1 - activations[i] ** 2)
            step += 1
            for i in range(len(model.weights)):
                mw[i] = 0.9 * mw[i] + 0.1 * grad_w[i]
                vw[i] = 0.999 * vw[i] + 0.001 * (grad_w[i] ** 2)
                mb[i] = 0.9 * mb[i] + 0.1 * grad_b[i]
                vb[i] = 0.999 * vb[i] + 0.001 * (grad_b[i] ** 2)
                model.weights[i] -= learning_rate * (mw[i] / (1 - 0.9 ** step)) / (np.sqrt(vw[i] / (1 - 0.999 ** step)) + 1e-8)
                model.biases[i] -= learning_rate * (mb[i] / (1 - 0.9 ** step)) / (np.sqrt(vb[i] / (1 - 0.999 ** step)) + 1e-8)
        losses.append(float(np.mean((model.predict(x) - y) ** 2)))
    return model, losses


def fused_action(model: MLP, features: np.ndarray, mlp_weight: float = 0.25) -> np.ndarray:
    learned = model.predict(features[None, :])[0]
    safe = teacher_action(features)
    return np.clip((1 - mlp_weight) * safe + mlp_weight * learned, -1, 1)
