"""Validate trajectories and print a compact dataset summary."""
import argparse
from .data import files, load_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="data/traj")
    args = parser.parse_args()
    paths = files(args.root)
    frames, duration = 0, 0.0
    for path in paths:
        ep = load_episode(path)
        frames += len(ep["t"])
        duration += ep["t"][-1] - ep["t"][0]
    print(f"episodes={len(paths)} frames={frames} duration_s={duration:.1f}")


if __name__ == "__main__":
    main()
