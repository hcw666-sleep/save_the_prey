"""Training and evaluation entry points for Save the Prey."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from perception import extract_features
from policy import MLP, fused_action, teacher_action, train_mlp
from savetheprey import IMAGE_SIZE, EpisodeLayout, OBSTACLES, SaveThePrey


ROOT = Path(__file__).resolve().parent
MODEL_DEFAULT = ROOT / "models" / "visual_policy.npz"
RESULTS_DEFAULT = ROOT / "results"


def _collect_data(episodes: int, seed: int, max_steps: int):
    env = SaveThePrey(seed=seed, max_steps=max_steps)
    features, labels = [], []
    outcomes = {}
    for _ in range(episodes):
        image = env.reset()
        previous = None
        for _ in range(max_steps):
            feature, prey = extract_features(image, previous, env.dt)
            action = teacher_action(feature)
            features.append(feature)
            labels.append(action)
            previous = prey
            image, status = env.step(action)
            if status != "running":
                outcomes[status] = outcomes.get(status, 0) + 1
                break
    env.close()
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32), outcomes


def train(args):
    started = time.time()
    x, y, outcomes = _collect_data(args.episodes, args.seed, args.max_steps)
    model, losses = train_mlp(x, y, epochs=args.epochs, batch_size=args.batch_size,
                              learning_rate=args.learning_rate, seed=args.seed)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "algorithm": "NumPy MLP behavior cloning",
        "input": "800x800 RGB image, 16 pixel-derived features",
        "output": "2D acceleration, each component clipped to [-1, 1], then vector-limited by the environment",
        "hidden_widths": [64, 64],
        "train_seed": args.seed,
        "training_episodes": args.episodes,
        "training_samples": len(x),
        "epochs": args.epochs,
        "dt": 0.1,
        "prey_max_speed": 0.25,
        "predator_max_speed": 0.28,
        "obstacles": OBSTACLES.tolist(),
        "teacher_episode_outcomes": outcomes,
        "initial_mse": losses[0],
        "final_mse": losses[-1],
    }
    model.save(args.model, config)
    metrics = {"config": config, "losses": losses, "elapsed_seconds": time.time() - started}
    (args.model.parent / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(args.model), **config,
                      "elapsed_seconds": round(time.time() - started, 2)}, indent=2))


def _layout_from_dict(value: dict, index: int) -> EpisodeLayout:
    try:
        points = {
            "prey": np.asarray(value["prey"], dtype=np.float64),
            "predator": np.asarray(value["predator"], dtype=np.float64),
            "shelter": np.asarray(value["shelter"], dtype=np.float64),
            "obstacles": np.asarray(value.get("obstacles", OBSTACLES), dtype=np.float64),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Layout {index + 1} has missing or invalid coordinates: {exc}") from exc
    for name in ("prey", "predator", "shelter"):
        point = points[name]
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError(f"Layout {index + 1} {name} must be two finite coordinates")
        if np.any(np.abs(point) > 1.0):
            raise ValueError(f"Layout {index + 1} {name} must lie within [-1, 1]")
    obstacles = points["obstacles"]
    if obstacles.shape != (3, 2) or not np.all(np.isfinite(obstacles)):
        raise ValueError(f"Layout {index + 1} obstacles must contain exactly three finite 2D points")
    if np.any(np.abs(obstacles) > 1.0):
        raise ValueError(f"Layout {index + 1} obstacle coordinates must lie within [-1, 1]")
    if np.linalg.norm(points["prey"] - points["predator"]) < SaveThePrey.capture_distance:
        raise ValueError(f"Layout {index + 1} starts with prey and predator overlapping")
    if np.any(np.linalg.norm(obstacles - points["prey"], axis=1) < SaveThePrey.obstacle_distance):
        raise ValueError(f"Layout {index + 1} starts with prey overlapping an obstacle")
    return EpisodeLayout(points["prey"], points["predator"], points["shelter"], obstacles,
                         str(value.get("name", f"layout_{index + 1:03d}")))


def _read_layouts(path: Path | None, episodes: int, seed: int):
    if path is None:
        env = SaveThePrey(seed=seed)
        layouts = [env.sample_layout(f"random_{i + 1:03d}") for i in range(episodes)]
        env.close()
        return layouts
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    values = data["layouts"] if isinstance(data, dict) else data
    return [_layout_from_dict(value, i) for i, value in enumerate(values)]


def _run_layout(layout: EpisodeLayout, model: MLP | None, weight: float, max_steps: int,
                env: SaveThePrey):
    image = env.reset(layout)
    previous = None
    for _ in range(max_steps):
        features, prey = extract_features(image, previous, env.dt)
        action = teacher_action(features) if model is None else fused_action(model, features, weight)
        previous = prey
        image, status = env.step(action)
        if status != "running":
            break
    return {"name": layout.name, "status": status, "steps": env.steps,
            "prey_final": env.prey.tolist(), "predator_final": env.predator.tolist(),
            "shelter": env.shelter.tolist()}


def evaluate(args):
    model = None if args.teacher_baseline else MLP.load(args.model)
    layouts = _read_layouts(args.layouts, args.episodes, args.seed)
    env = SaveThePrey(seed=0, max_steps=args.max_steps)
    try:
        results = [_run_layout(layout, model, args.mlp_weight, args.max_steps, env)
                   for layout in layouts]
    finally:
        env.close()
    counts = {key: sum(row["status"] == key for row in results)
              for key in ("success", "captured", "obstacle_collision", "out_of_bounds", "timeout")}
    summary = {
        "seed": args.seed,
        "model": args.model.as_posix() if model else "teacher_baseline",
        "mlp_weight": args.mlp_weight if model else 0.0,
        "episodes": len(results),
        "rates": {key: value / max(1, len(results)) for key, value in counts.items()},
        "mean_steps": float(np.mean([row["steps"] for row in results])) if results else 0.0,
        "counts": counts,
        "episodes_detail": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes_detail"}, indent=2))
    print(f"episode log: {args.output}")


def play(args):
    try:
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
    except ImportError as exc:
        raise SystemExit("The animation window requires Python with Tkinter support") from exc

    model = MLP.load(args.model)
    layouts = _read_layouts(args.layouts, args.episodes, args.seed)
    if len(layouts) != 1:
        raise SystemExit("play expects exactly one layout")

    root = tk.Tk()
    root.title("Save the Prey")
    root.resizable(False, False)
    env = SaveThePrey(seed=args.seed, max_steps=args.max_steps)
    root.protocol("WM_DELETE_WINDOW", lambda: (env.close(), root.destroy()))
    state = {"paused": False, "image": None, "previous": None,
             "photo": None, "finished": False}

    image_label = ttk.Label(root)
    image_label.pack(padx=12, pady=(12, 6))
    status_label = ttk.Label(root, anchor="center", font=("Segoe UI", 12, "bold"))
    status_label.pack(fill="x", padx=12, pady=4)
    controls = ttk.Frame(root, padding=(12, 4, 12, 12))
    controls.pack(fill="x")

    def reset_layout():
        state["previous"] = None
        state["finished"] = False
        state["image"] = env.reset(layouts[0])
        update_view()

    def update_view():
        pil_image = Image.fromarray(state["image"])
        if args.display_size != IMAGE_SIZE:
            pil_image = pil_image.resize((args.display_size, args.display_size), Image.Resampling.LANCZOS)
        state["photo"] = ImageTk.PhotoImage(pil_image)
        image_label.configure(image=state["photo"])
        status = env.status
        if state["paused"] and status == "running":
            status = "paused"
        status_label.configure(text=f"{layouts[0].name}    |    {status}    |    {env.steps} steps")

    def toggle_pause(_event=None):
        if env.status != "running":
            return
        state["paused"] = not state["paused"]
        update_view()

    def restart(_event=None):
        state["paused"] = False
        reset_layout()

    def toggle_fullscreen(_event=None):
        root.attributes("-fullscreen", not root.attributes("-fullscreen"))

    ttk.Button(controls, text="暂停 / 继续", command=toggle_pause).pack(side="left")
    ttk.Button(controls, text="重新开始", command=restart).pack(side="left", padx=(8, 0))
    ttk.Label(controls, text="空格：暂停    R：重开    F11：全屏").pack(side="right")
    root.bind("<space>", toggle_pause)
    root.bind("r", restart)
    root.bind("<F11>", toggle_fullscreen)
    root.bind("<Escape>", lambda _event: root.attributes("-fullscreen", False))

    def tick():
        if not state["paused"] and env.status == "running":
            features, prey = extract_features(state["image"], state["previous"], env.dt)
            action = fused_action(model, features, args.mlp_weight)
            state["previous"] = prey
            state["image"], _ = env.step(action)
            update_view()
        root.after(max(1, round(1000 / args.fps)), tick)

    reset_layout()
    root.after(max(1, round(1000 / args.fps)), tick)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Save the Prey coursework")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="collect image-only teacher demonstrations and train an MLP")
    train_parser.add_argument("--episodes", type=int, default=120)
    train_parser.add_argument("--max-steps", type=int, default=1200)
    train_parser.add_argument("--epochs", type=int, default=160)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=0.002)
    train_parser.add_argument("--seed", type=int, default=2026)
    train_parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    train_parser.set_defaults(func=train)
    eval_parser = commands.add_parser("evaluate", help="evaluate on random or supplied layouts")
    eval_parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    eval_parser.add_argument("--layouts", type=Path)
    eval_parser.add_argument("--episodes", type=int, default=30)
    eval_parser.add_argument("--max-steps", type=int, default=1200)
    eval_parser.add_argument("--mlp-weight", type=float, default=0.25)
    eval_parser.add_argument("--teacher-baseline", action="store_true",
                             help="evaluate the image-derived teacher without the learned MLP")
    eval_parser.add_argument("--seed", type=int, default=5000)
    eval_parser.add_argument("--output", type=Path, default=RESULTS_DEFAULT / "evaluation.json")
    eval_parser.set_defaults(func=evaluate)
    play_parser = commands.add_parser("play", help="play one test layout in an animation window")
    play_parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    play_parser.add_argument("--layouts", type=Path, default=ROOT / "configs" / "official_test_layout.json")
    play_parser.add_argument("--episodes", type=int, default=1)
    play_parser.add_argument("--max-steps", type=int, default=1200)
    play_parser.add_argument("--mlp-weight", type=float, default=0.25)
    play_parser.add_argument("--seed", type=int, default=5000)
    play_parser.add_argument("--fps", type=int, default=30)
    play_parser.add_argument("--display-size", type=int, default=600)
    play_parser.set_defaults(func=play)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
