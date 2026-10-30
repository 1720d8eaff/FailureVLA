"""Read and validate the public G1-D trajectory format."""
from pathlib import Path
import h5py
import numpy as np

SHAPES = {"joints": 7, "fingers": 6, "eef_left": 10, "eef_right": 10}


def load_episode(path):
    with h5py.File(path, "r") as f:
        t = np.asarray(f["t"], dtype=np.float64)
        if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError(f"{path}: invalid or non-increasing timestamps")
        values = {}
        for name, width in SHAPES.items():
            a = np.asarray(f[name], dtype=np.float32)
            if a.shape != (len(t), width) or not np.isfinite(a).all():
                raise ValueError(f"{path}: invalid {name}: {a.shape}")
            values[name] = a
        instruction = f["instruction"][()]
        values["instruction"] = instruction.decode("utf-8") if isinstance(instruction, bytes) else str(instruction)
        values["t"] = t
    return values


def files(root):
    paths = sorted(Path(root).glob("*.h5"))
    if not paths:
        raise ValueError(f"No HDF5 trajectories in {root}")
    return paths


def state(ep):
    # Finger values are in the device's raw 0-2000 scale; joints are radians.
    return np.concatenate((ep["joints"], ep["fingers"] / 2000.0), axis=1)
