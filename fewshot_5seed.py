"""
fewshot_5seed.py
Few-shot ImageNet (DINOv3 ViT-L/16, p=1024), Optimized method, k=100.
For each shot count and each of 5 seeds:
  1. Sample `shots` per class from train.
  2. Compute centers, sigmas from those shots (mean / std of the few-shot subset).
  3. Adam-optimize per-class weights on those centers/sigmas (2000 steps).
  4. Top-100 features per class.
  5. Evaluate nearest-centroid (using few-shot centers) on full TRAIN and full TEST.
Output: CSV with one row per (shot, seed), printed mean+/-std table.
"""
import os, time, sys, csv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

SHOTS = [2, 3, 4, 5, 10, 20, 50, 100, 200]
SEEDS = [0, 1, 2, 3, 4]
K = 100
LR, STEPS = 0.01, 2000
INIT_STD = 0.5

OUT_CSV = r"data/fewshot_5seed.csv"


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    if 'X' in d: return d['X'].astype(np.float32), d['y'].astype(np.int64)
    return d['features'].astype(np.float32), d['labels'].astype(np.int64)


print("Loading data...", flush=True)
X_tr, y_tr = load_npz(r"data/dinov3_train_fixed.npz")
X_te, y_te = load_npz(r"data/dinov3_val_fixed.npz")
C = int(y_tr.max()) + 1
p = X_tr.shape[1]

# Pre-sort train by class for fast few-shot sampling
order_tr = np.argsort(y_tr, kind='stable')
X_tr_s = X_tr[order_tr]; y_tr_s = y_tr[order_tr]
counts_tr = np.bincount(y_tr_s, minlength=C)
ends_tr = np.cumsum(counts_tr); starts_tr = ends_tr - counts_tr
print(f"C={C}, p={p}, n_train={len(y_tr)}, n_test={len(y_te)}", flush=True)


def sample_shots(n_shot, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(C):
        s, e = starts_tr[c], ends_tr[c]
        n_avail = e - s
        n_take = min(n_shot, n_avail)
        idx.extend((s + rng.choice(n_avail, n_take, replace=False)).tolist())
    return np.array(idx)


def compute_stats_fewshot(X_few, y_few):
    centers = np.zeros((C, p), dtype=np.float32)
    sigmas = np.zeros((C, p), dtype=np.float32)
    for c in range(C):
        Xc = X_few[y_few == c]
        centers[c] = Xc.mean(0)
        sigmas[c] = Xc.std(0) + 1e-3 if len(Xc) > 1 else np.full(p, 1.0, dtype=np.float32)
    return centers, sigmas


def optimize_weights(centers_t, sigmas_t, seed, chunk=200):
    """Vectorized per-class Adam: chunk classes, batched matmul."""
    torch.manual_seed(seed)
    weights = torch.zeros((C, p), device=device, dtype=torch.float32)
    for cs in range(0, C, chunk):
        ce = min(cs + chunk, C)
        CC = ce - cs
        centers_chunk = centers_t[cs:ce]
        sig_chunk = sigmas_t[cs:ce]
        diffs = (centers_chunk.unsqueeze(1) - centers_t.unsqueeze(0)).abs()  # (CC, C, p)
        diag_mask = torch.zeros((CC, C), device=device)
        for i in range(CC):
            diag_mask[i, cs + i] = float('inf')

        u = (INIT_STD * torch.randn(CC, p, device=device)).requires_grad_(True)
        opt = torch.optim.Adam([u], lr=LR)
        for _ in range(STEPS):
            opt.zero_grad()
            W = torch.softmax(u, dim=1)
            num = torch.einsum('ilj,ij->il', diffs, W) + diag_mask
            min_num = num.min(dim=1).values
            denom = (sig_chunk * W).sum(dim=1)
            (-(min_num / denom).sum()).backward()
            opt.step()
        with torch.no_grad():
            weights[cs:ce] = torch.softmax(u, dim=1)
        del diffs, u, opt
        torch.cuda.empty_cache()
    return weights.cpu().numpy()


def nc_acc(X, y, topk_indices, centers_t):
    """Core protocol."""
    correct = 0
    for c in range(C):
        idx = (y == c).nonzero()[0]
        if len(idx) == 0: continue
        Xc = torch.tensor(X[idx], device=device)
        feat = topk_indices[c]
        d = torch.cdist(Xc[:, feat], centers_t[:, feat])
        correct += (d.argmin(1) == c).sum().item()
    return correct / len(y)


with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['shots', 'seed', 'train', 'test'])
    writer.writeheader()

    for shots in SHOTS:
        for seed in SEEDS:
            t0 = time.time()
            idx = sample_shots(shots, seed)
            X_few = X_tr_s[idx]
            y_few = y_tr_s[idx]
            centers, sigmas = compute_stats_fewshot(X_few, y_few)
            centers_t = torch.tensor(centers, device=device)
            sigmas_t = torch.tensor(sigmas, device=device)

            weights = optimize_weights(centers_t, sigmas_t, seed)
            topk = [np.argsort(weights[l])[-K:] for l in range(C)]

            tr = nc_acc(X_tr, y_tr, topk, centers_t)
            te = nc_acc(X_te, y_te, topk, centers_t)
            elapsed = time.time() - t0
            print(f"shots={shots:3d} seed={seed}: train={tr:.4f} test={te:.4f}  ({elapsed:.0f}s)", flush=True)
            writer.writerow({'shots': shots, 'seed': seed, 'train': tr, 'test': te})
            f.flush()

print(f"\nWrote {OUT_CSV}")

import pandas as pd
df = pd.read_csv(OUT_CSV)
print("\n=== mean +/- std (5 seeds) ===")
g = df.groupby('shots').agg({'train': ['mean', 'std'], 'test': ['mean', 'std']})
print(g.round(4))
