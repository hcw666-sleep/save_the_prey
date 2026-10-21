"""Extract the policy's fixed 16-value observation from an RGB frame."""

from __future__ import annotations

import numpy as np

from savetheprey import IMAGE_SIZE, OBSTACLE_RGB, PREDATOR_RGB, PREY_RGB, SHELTER_RGB

PERCEPTION_SCALE = 4

RGB_PALETTES = {
    "prey": ((PREY_RGB, 0), ((172, 236, 172), 42)),
    "predator": ((PREDATOR_RGB, 0), ((236, 172, 172), 42)),
    "shelter": ((SHELTER_RGB, 0),),
    "obstacle": ((OBSTACLE_RGB, 0), ((63, 63, 63), 42)),
}


def _color_mask(image: np.ndarray, role: str) -> np.ndarray:
    colors = RGB_PALETTES[role]
    distances = [
        np.sum((image.astype(np.int32) - np.asarray(color, dtype=np.int32)) ** 2, axis=2)
        <= tolerance ** 2
        for color, tolerance in colors
    ]
    return np.logical_or.reduce(distances)


def _centroid(image: np.ndarray, role: str) -> np.ndarray:
    mask = _color_mask(image, role)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError(f"Could not locate {role} in the RGB frame")
    return np.asarray([xs.mean(), ys.mean()], dtype=np.float64)


def _world(point_px: np.ndarray) -> np.ndarray:
    original_px = point_px * PERCEPTION_SCALE
    return np.asarray([
        original_px[0] * 2 / (IMAGE_SIZE - 1) - 1,
        1 - original_px[1] * 2 / (IMAGE_SIZE - 1),
    ])


def _obstacle_centers(image: np.ndarray) -> np.ndarray:
    mask = _color_mask(image, "obstacle")
    runs = []
    previous = []
    for y, row in enumerate(mask):
        padded = np.concatenate(([False], row, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        current = []
        for start, end in zip(edges[::2], edges[1::2]):
            node = len(runs)
            runs.append([y, int(start), int(end), node])
            current.append(node)
            for prior in previous:
                py, pstart, pend, _ = runs[prior]
                if pend >= start - 1 and pstart <= end:
                    root_a, root_b = node, prior
                    while runs[root_a][3] != root_a:
                        root_a = runs[root_a][3]
                    while runs[root_b][3] != root_b:
                        root_b = runs[root_b][3]
                    if root_a != root_b:
                        runs[root_b][3] = root_a
        previous = current

    components = {}
    for node, (y, start, end, _) in enumerate(runs):
        root = node
        while runs[root][3] != root:
            root = runs[root][3]
        count = end - start
        sums = components.setdefault(root, [0, 0, 0])
        sums[0] += count
        sums[1] += y * count
        sums[2] += (start + end - 1) * count / 2
    centers = []
    for count, sum_y, sum_x in components.values():
        if count >= 8:
            centers.append(_world(np.asarray([sum_x / count, sum_y / count])))
    if len(centers) != 3:
        # Close circles can touch in the raster image and form one component.
        # Recover three stable approximate centers from the dark-pixel cloud so
        # a valid layout does not crash the policy path.
        ys, xs = np.nonzero(mask)
        points = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
        if len(points) < 20:
            if centers:
                visible = [np.asarray(center, dtype=np.float64) for center in centers]
            elif len(points):
                visible = [_world(points.mean(axis=0))]
            else:
                # No obstacle pixels remain after occlusion. Continue with a
                # neutral image-space estimate instead of aborting evaluation.
                visible = [np.zeros(2, dtype=np.float64)]
            while len(visible) < 3:
                visible.append(visible[-1].copy())
            centers = visible[:3]
            centers = np.asarray(centers)
            return centers[np.lexsort((centers[:, 0], -centers[:, 1]))]
        centered = points - points.mean(axis=0)
        axis = np.linalg.svd(centered, full_matrices=False)[2][0]
        projection = centered @ axis
        quantiles = np.quantile(projection, [0.2, 0.5, 0.8])
        seeds = np.asarray([
            points[np.argmin(np.abs(projection - value))]
            for value in quantiles
        ], dtype=np.float64)
        for _ in range(32):
            distances = np.sum((points[:, None, :] - seeds[None, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)
            updated = np.asarray([
                points[labels == index].mean(axis=0) if np.any(labels == index) else seeds[index]
                for index in range(3)
            ])
            if np.allclose(updated, seeds, atol=1e-3):
                break
            seeds = updated
        centers = [_world(center) for center in seeds]
    centers = np.asarray(centers)
    return centers[np.lexsort((centers[:, 0], -centers[:, 1]))]


def extract_features(image: np.ndarray, previous_prey: np.ndarray | None = None,
                     dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized features and current prey position, using pixels only."""
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected an {IMAGE_SIZE}x{IMAGE_SIZE} RGB image, got {image.shape}")
    # Object radii are at least 10 source pixels, so 4x point sampling retains
    # enough interior color pixels while reducing full-frame mask work 16x.
    image = image[::PERCEPTION_SCALE, ::PERCEPTION_SCALE]
    prey = _world(_centroid(image, "prey"))
    predator = _world(_centroid(image, "predator"))
    shelter = _world(_centroid(image, "shelter"))
    obstacles = _obstacle_centers(image)
    velocity = np.zeros(2) if previous_prey is None else (prey - previous_prey) / dt
    values = np.concatenate([
        (shelter - prey) / 2,
        (predator - prey) / 2,
        ((obstacles - prey) / 2).reshape(-1),
        velocity / 0.25,
        (prey + 1) / 2,
        (1 - prey) / 2,
    ])
    return values.astype(np.float32), prey
