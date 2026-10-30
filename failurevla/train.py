"""A reproducible state-history next-state baseline, split by episode."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .data import files, load_episode, state


def samples(paths, history, stride):
    xs, ys = [], []
    for path in paths:
        s = state(load_episode(path))
        for i in range(history - 1, len(s) - 1, stride):
            xs.append(s[i-history+1:i+1].reshape(-1))
            ys.append(s[i+1] - s[i])
    if not xs:
        raise ValueError("No samples; reduce history or check trajectories")
    return np.asarray(xs), np.asarray(ys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/traj")
    p.add_argument("--output", default="runs/baseline")
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if min(a.history, a.stride, a.epochs) < 1:
        p.error("history, stride and epochs must be positive")
    paths = files(a.data)
    if len(paths) < 2:
        p.error("at least two episodes are required")
    rng = np.random.default_rng(a.seed)
    paths = [paths[i] for i in rng.permutation(len(paths))]
    n_test = max(1, len(paths) // 5)
    test_paths, train_paths = paths[:n_test], paths[n_test:]
    x, y = samples(train_paths, a.history, a.stride)
    xt, yt = samples(test_paths, a.history, a.stride)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.maximum(std, 1e-6)
    torch.manual_seed(a.seed)
    model = nn.Sequential(nn.Linear(x.shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 13))
    loader = DataLoader(TensorDataset(torch.from_numpy((x-mean)/std), torch.from_numpy(y)), batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(a.seed))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(a.epochs):
        model.train()
        loss_sum = 0.0
        for bx, by in loader:
            opt.zero_grad()
            loss = ((model(bx) - by)**2).mean()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(bx)
        print(f"epoch={epoch+1} train_mse={loss_sum/len(x):.8f}")
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy((xt-mean)/std)).numpy()
    err = np.abs(prediction-yt)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "input_mean": torch.from_numpy(mean), "input_std": torch.from_numpy(std), "history": a.history}, out / "model.pt")
    result = {"seed": a.seed, "history": a.history, "stride": a.stride, "epochs": a.epochs,
              "train_episodes": [p.name for p in train_paths], "test_episodes": [p.name for p in test_paths],
              "test_samples": len(yt), "joint_delta_mae_rad": float(err[:, :7].mean()),
              "finger_delta_mae_raw": float(err[:, 7:].mean() * 2000),
              "zero_delta_joint_mae_rad": float(np.abs(yt[:, :7]).mean()),
              "zero_delta_finger_mae_raw": float(np.abs(yt[:, 7:]).mean() * 2000)}
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    main()
