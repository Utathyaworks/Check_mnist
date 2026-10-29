"""MNIST CNN classifier: trains on GPU, logs metrics, saves plots to results/."""
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

EPOCHS = 8
BATCH = 128
OUT = "results"
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev}", torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)


def load(train):
    ds = datasets.MNIST("data", train, download=True)
    x = ds.data.float().div(255).sub(0.1307).div(0.3081).unsqueeze(1)
    return x.to(dev), ds.targets.to(dev)


Xtr, ytr = load(True)
Xte, yte = load(False)

model = nn.Sequential(
    nn.Conv2d(1, 32, 3), nn.ReLU(),
    nn.Conv2d(32, 64, 3), nn.ReLU(),
    nn.MaxPool2d(2), nn.Flatten(), nn.Dropout(0.25),
    nn.Linear(9216, 128), nn.ReLU(),
    nn.Linear(128, 10),
).to(dev)
opt = torch.optim.Adam(model.parameters(), 1e-3)


@torch.no_grad()
def evaluate():
    model.eval()
    logits = torch.cat([model(Xte[i:i + 1000]) for i in range(0, len(Xte), 1000)])
    loss = F.cross_entropy(logits, yte).item()
    pred = logits.argmax(1)
    return loss, (pred == yte).float().mean().item() * 100, pred


hist = []
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(Xtr), device=dev)
    run, n = 0.0, 0
    for i in range(0, len(Xtr), BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
        loss.backward()
        opt.step()
        run += loss.item() * len(idx)
        n += len(idx)
    vl, va, pred = evaluate()
    hist.append((ep, run / n, vl, va))
    print(f"epoch {ep}/{EPOCHS} train_loss {run / n:.4f} test_loss {vl:.4f} "
          f"test_acc {va:.2f}% ({time.time() - t0:.0f}s)", flush=True)

with open(f"{OUT}/metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "train_loss", "test_loss", "test_acc"])
    w.writerows(hist)
torch.save(model.state_dict(), f"{OUT}/mnist_cnn.pt")

# Plot 1: loss and accuracy curves
h = np.array(hist)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(h[:, 0], h[:, 1], label="train")
ax[0].plot(h[:, 0], h[:, 2], label="test")
ax[0].set(title="Loss", xlabel="epoch")
ax[0].legend()
ax[1].plot(h[:, 0], h[:, 3], color="tab:green")
ax[1].set(title="Test accuracy (%)", xlabel="epoch")
fig.tight_layout()
fig.savefig(f"{OUT}/curves.png", dpi=120)

# Plot 2: confusion matrix
y, p = yte.cpu().numpy(), pred.cpu().numpy()
cm = np.zeros((10, 10), int)
np.add.at(cm, (y, p), 1)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set(title="Confusion matrix", xlabel="predicted", ylabel="true",
       xticks=range(10), yticks=range(10))
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im)
fig.tight_layout()
fig.savefig(f"{OUT}/confusion.png", dpi=120)

# Plot 3: sample predictions (wrong ones first, if any)
order = np.concatenate([np.where(y != p)[0], np.where(y == p)[0]])[:16]
fig, axes = plt.subplots(2, 8, figsize=(12, 3.6))
for a, k in zip(axes.ravel(), order):
    a.imshow(Xte[k, 0].cpu(), cmap="gray")
    a.set_title(f"t={y[k]} p={p[k]}", color="green" if y[k] == p[k] else "red", fontsize=9)
    a.axis("off")
fig.tight_layout()
fig.savefig(f"{OUT}/samples.png", dpi=120)

print(f"final test_acc {hist[-1][3]:.2f}%  | saved to {OUT}/", flush=True)
