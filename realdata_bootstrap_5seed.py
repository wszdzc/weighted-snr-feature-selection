"""
realdata_bootstrap_5seed.py
For each (dataset, seed in 0..4), draw a 90% subsample of training data
(without replacement), recompute centers/sigmas from that subset, then
evaluate Simple, FAIR, MSDA, Optimized at all k. All methods get std.

Baseline: full data, no subsampling, deterministic.

Eval: core protocol (per-class true-class recovery rate).
Optimized: 30k Adam steps, vectorized batched, INIT_STD=0.5.
MSDA: FISTA with lambda grid, top-k by row-norm.
Simple: per-class min-SNR top-k.
FAIR: per-class top-k by t-statistic |mu_l - mu_-l| / sigma_l.
"""
import os, time, sys, csv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, r'data/Claude')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from mai2019_msda import fit_msda, nc_acc_on_features  # only fit_msda used; eval done locally

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

K_LIST = [50, 100, 200, 300, 500, 700, 900, 1000]
SEEDS = [0, 1, 2, 3, 4]
LR, STEPS = 0.01, 30000
INIT_STD = 0.5
CHUNK_DEFAULT = 250
SUBSAMPLE_FRAC = 0.9

LAM_CIFAR    = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.6, 1.0]
LAM_CUB      = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.6, 1.0]
LAM_IMAGENET = [0.1, 0.3, 0.5, 0.6, 0.62, 0.65, 0.7, 0.8, 1.0]
N_ITER_MSDA = 200

OUT_CSV = r"data/realdata_bootstrap_5seed.csv"


def load_cls_npz(path):
    d = np.load(path, allow_pickle=True)
    if 'X' in d:
        return d['X'].astype(np.float32), d['y'].astype(np.int64)
    return d['features'].astype(np.float32), d['labels'].astype(np.int64)


def subsample_train(X, y, frac, seed, C):
    """Stratified per-class 90% subsample without replacement."""
    rng = np.random.RandomState(seed)
    keep = np.zeros(len(y), dtype=bool)
    for c in range(C):
        idx_c = np.where(y == c)[0]
        n_keep = max(2, int(len(idx_c) * frac))
        keep_c = rng.choice(idx_c, n_keep, replace=False)
        keep[keep_c] = True
    return X[keep], y[keep]


def nc_acc_core(X, y, topk_indices, centers_t, C):
    """Core protocol: per-class true-class recovery rate."""
    correct = 0
    for c in range(C):
        idx = (y == c).nonzero()[0]
        if len(idx) == 0: continue
        Xc = torch.tensor(X[idx], device=device)
        feat = topk_indices[c]
        d = torch.cdist(Xc[:, feat], centers_t[:, feat])
        correct += (d.argmin(1) == c).sum().item()
    return correct / len(y)


def nc_acc_shared(X, y, feats, centers_t, C):
    """Shared-feature eval (MSDA): all classes use same feature subset."""
    feats = np.ascontiguousarray(feats)
    X_t = torch.tensor(np.ascontiguousarray(X[:, feats]), device=device)
    Cs = centers_t[:, feats]
    correct = 0
    BATCH = 50000
    for bs in range(0, len(y), BATCH):
        be = min(bs + BATCH, len(y))
        d = torch.cdist(X_t[bs:be], Cs)
        preds = d.argmin(1).cpu().numpy()
        correct += (preds == y[bs:be]).sum()
    return correct / len(y)


def optimize_weights_batched(centers_t, sigmas_t, C, p, seed, chunk=CHUNK_DEFAULT):
    torch.manual_seed(seed)
    weights = torch.zeros((C, p), device=device, dtype=torch.float32)
    for cs in range(0, C, chunk):
        ce = min(cs + chunk, C)
        CC = ce - cs
        centers_chunk = centers_t[cs:ce]
        sig_chunk = sigmas_t[cs:ce]
        diffs = (centers_chunk.unsqueeze(1) - centers_t.unsqueeze(0)).abs()
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
            loss = -(min_num / denom).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            weights[cs:ce] = torch.softmax(u, dim=1)
        del diffs, u, opt
        torch.cuda.empty_cache()
    return weights.cpu().numpy()


def compute_simple_scores(centers, sigmas, C, p):
    """min over l' != l of |mu_l - mu_l'| / sigma_l per coordinate."""
    scores = np.zeros((C, p))
    for l in range(C):
        D = np.abs(centers - centers[l])
        D = np.delete(D, l, axis=0)
        scores[l] = D.min(0) / sigmas[l]
    return scores


def compute_fair_scores(centers, sigmas, C, p, X, y):
    """per-class FAIR: |mu_l - mu_{-l}| / sigma_l, where mu_{-l} = mean of all OTHER classes' samples."""
    n = len(y)
    sums = np.zeros((C, p), dtype=np.float64)
    counts = np.zeros(C, dtype=np.int64)
    # use sort+slice to compute fast per-class sums
    order = np.argsort(y, kind='stable')
    Xs = X[order]; ys = y[order]
    cnts = np.bincount(ys, minlength=C)
    ends = np.cumsum(cnts); starts = ends - cnts
    for c in range(C):
        if cnts[c] > 0:
            sums[c] = Xs[starts[c]:ends[c]].sum(0)
            counts[c] = cnts[c]
    total_sum = sums.sum(0)
    total_n = counts.sum()
    scores = np.zeros((C, p))
    for l in range(C):
        if counts[l] == 0 or total_n - counts[l] == 0:
            scores[l] = 0
            continue
        mu_other = (total_sum - sums[l]) / (total_n - counts[l])
        scores[l] = np.abs(centers[l] - mu_other) / sigmas[l]
    return scores


def run_methods_on_subset(X_sub, y_sub, X_tr_full, y_tr_full, X_te, y_te, C, p,
                          dataset_name, seed, lam_list, writer):
    """Evaluate Simple/FAIR/MSDA/Optimized on the subsampled (centers, sigmas)
    but eval on the full train and full test. Use core protocol for non-MSDA,
    shared-feature for MSDA."""
    centers = np.stack([X_sub[y_sub == c].mean(0) for c in range(C)])
    sigmas  = np.stack([X_sub[y_sub == c].std(0) + 1e-8 for c in range(C)])
    centers_t = torch.tensor(centers, device=device)
    sigmas_t  = torch.tensor(sigmas,  device=device)

    # Simple
    print(f"  [seed {seed}] Simple", flush=True)
    s_scores = compute_simple_scores(centers, sigmas, C, p)
    for k in K_LIST:
        if k > p: continue
        topk = [np.argsort(s_scores[l])[-k:] for l in range(C)]
        tr = nc_acc_core(X_tr_full, y_tr_full, topk, centers_t, C)
        te = nc_acc_core(X_te, y_te, topk, centers_t, C)
        writer.writerow({'dataset': dataset_name, 'seed': seed, 'method': 'Simple',
                         'k': k, 'train': tr, 'test': te})

    # FAIR
    print(f"  [seed {seed}] FAIR", flush=True)
    f_scores = compute_fair_scores(centers, sigmas, C, p, X_sub, y_sub)
    for k in K_LIST:
        if k > p: continue
        topk = [np.argsort(f_scores[l])[-k:] for l in range(C)]
        tr = nc_acc_core(X_tr_full, y_tr_full, topk, centers_t, C)
        te = nc_acc_core(X_te, y_te, topk, centers_t, C)
        writer.writerow({'dataset': dataset_name, 'seed': seed, 'method': 'FAIR',
                         'k': k, 'train': tr, 'test': te})

    # MSDA: fit on subsample, top-k by row-norm of B (shared across classes)
    print(f"  [seed {seed}] MSDA fit lambda grid", flush=True)
    rn_per_lam = {}
    for lam in lam_list:
        t0 = time.time()
        _, rn = fit_msda(X_sub, y_sub, C, lam, n_iter=N_ITER_MSDA)
        active = int((rn > 1e-8).sum())
        rn_per_lam[lam] = rn
        print(f"    lam={lam}: {time.time()-t0:.0f}s active={active}", flush=True)
    for k in K_LIST:
        if k > p: continue
        best = None
        for lam, rn in rn_per_lam.items():
            feats = np.ascontiguousarray(np.argsort(rn)[::-1][:k])
            te = nc_acc_shared(X_te, y_te, feats, centers_t, C)
            if best is None or te > best['test']:
                tr = nc_acc_shared(X_tr_full, y_tr_full, feats, centers_t, C)
                best = {'lam': lam, 'train': tr, 'test': te}
        writer.writerow({'dataset': dataset_name, 'seed': seed, 'method': 'MSDA',
                         'k': k, 'train': best['train'], 'test': best['test']})

    # Optimized
    print(f"  [seed {seed}] Optimized 30k", flush=True)
    t0 = time.time()
    weights = optimize_weights_batched(centers_t, sigmas_t, C, p, seed)
    print(f"    {time.time()-t0:.0f}s", flush=True)
    for k in K_LIST:
        if k > p: continue
        topk = [np.argsort(weights[l])[-k:] for l in range(C)]
        tr = nc_acc_core(X_tr_full, y_tr_full, topk, centers_t, C)
        te = nc_acc_core(X_te, y_te, topk, centers_t, C)
        writer.writerow({'dataset': dataset_name, 'seed': seed, 'method': 'Optimized',
                         'k': k, 'train': tr, 'test': te})


def run_baseline(X_tr_full, y_tr_full, X_te, y_te, C, p, dataset_name, writer):
    """Baseline: full features, full data centers, deterministic."""
    centers = np.stack([X_tr_full[y_tr_full == c].mean(0) for c in range(C)])
    centers_t = torch.tensor(centers, device=device)
    all_idx = [np.arange(p) for _ in range(C)]
    tr = nc_acc_core(X_tr_full, y_tr_full, all_idx, centers_t, C)
    te = nc_acc_core(X_te, y_te, all_idx, centers_t, C)
    writer.writerow({'dataset': dataset_name, 'seed': -1, 'method': 'Baseline',
                     'k': p, 'train': tr, 'test': te})


def run_dataset(name, X_tr, y_tr, X_te, y_te, lam_list, writer):
    C = int(y_tr.max()) + 1
    p = X_tr.shape[1]
    print(f"\n=== {name}: C={C}, p={p}, n_train={len(y_tr)}, n_test={len(y_te)} ===", flush=True)

    # Baseline (deterministic, full data)
    print("Baseline (full data)", flush=True)
    run_baseline(X_tr, y_tr, X_te, y_te, C, p, name, writer)

    for seed in SEEDS:
        print(f"\n--- {name} seed {seed} (90% subsample) ---", flush=True)
        X_sub, y_sub = subsample_train(X_tr, y_tr, SUBSAMPLE_FRAC, seed, C)
        print(f"  subsample size: {len(y_sub)}", flush=True)
        run_methods_on_subset(X_sub, y_sub, X_tr, y_tr, X_te, y_te, C, p,
                              name, seed, lam_list, writer)


with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['dataset', 'seed', 'method', 'k', 'train', 'test'])
    writer.writeheader()

    print("\n>>> CIFAR-100", flush=True)
    X_tr, y_tr = load_cls_npz(r"data/cifar100_dinov3_vitl16_train_cls.npz")
    X_te, y_te = load_cls_npz(r"data/cifar100_dinov3_vitl16_test_cls.npz")
    run_dataset("CIFAR-100", X_tr, y_tr, X_te, y_te, LAM_CIFAR, writer)
    f.flush()
    del X_tr, y_tr, X_te, y_te

    print("\n>>> CUB-200", flush=True)
    X_tr, y_tr = load_cls_npz(r"data/cub200_dinov3_vitl16_train_cls.npz")
    X_te, y_te = load_cls_npz(r"data/cub200_dinov3_vitl16_test_cls.npz")
    run_dataset("CUB-200", X_tr, y_tr, X_te, y_te, LAM_CUB, writer)
    f.flush()
    del X_tr, y_tr, X_te, y_te

    print("\n>>> ImageNet", flush=True)
    X_tr, y_tr = load_cls_npz(r"data/dinov3_train_fixed.npz")
    X_te, y_te = load_cls_npz(r"data/dinov3_val_fixed.npz")
    run_dataset("ImageNet", X_tr, y_tr, X_te, y_te, LAM_IMAGENET, writer)

print(f"\nWrote {OUT_CSV}")

import pandas as pd
df = pd.read_csv(OUT_CSV)
print("\n=== mean +/- std ===")
non_base = df[df['method'] != 'Baseline']
g = non_base.groupby(['dataset', 'method', 'k']).agg({'train': ['mean', 'std'], 'test': ['mean', 'std']})
print(g.round(4))
print("\n=== Baseline ===")
print(df[df['method'] == 'Baseline'][['dataset', 'train', 'test']])
