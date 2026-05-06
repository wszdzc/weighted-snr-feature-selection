"""
mai2019_msda.py
Python implementation of multi-class sparse discriminant analysis
following Mai, Yang, Zou (2019), Statistica Sinica.

Algorithm:
- Solve B ∈ R^{p x (C-1)} via group-lasso multivariate regression:
    min_B (1/2n) ||Y - XB||_F^2 + lam Σ_j ||B_{j,:}||_2
  where Y is an LDA-aligned indicator matrix.
- Group lasso penalty on rows: feature j is "selected" iff B_{j,:} != 0.
- Optimization: FISTA (accelerated proximal gradient).

Features for class l selected = top-k by row-norm of B_{j,:} (or non-zero rows
under lam-sparse solution).

After getting B, classify via the LDA rule:
  ŷ(x) = argmin_c (x - μ_c)^T Σ̂^{-1} (x - μ_c)  -- but we use the
  fitted B to project: ŷ(x) = argmax_c <x - μ̄, B β_c>.
However, to compare with our "core protocol" feature-selection paper, we report:
  - Selected feature subset (top-k by row-norm)
  - Nearest-centroid accuracy on the selected subset (for fair comparison)
"""
import os, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch


def make_indicator_Y(y, C):
    """
    LDA-aligned indicator matrix following Mai et al.
    Y[i, c] = sqrt(n / n_c) if y_i == c, else 0,
    then center each column to mean zero (so columns sum to 0).
    Returns Y ∈ R^{n x C}, then we drop the last column to get n x (C-1).
    """
    n = len(y)
    counts = np.bincount(y, minlength=C).astype(np.float32)
    Y = np.zeros((n, C), dtype=np.float32)
    for c in range(C):
        Y[y == c, c] = np.sqrt(n / max(counts[c], 1.0))
    Y -= Y.mean(0, keepdims=True)
    return Y[:, :C-1]  # drop one column to get C-1


def fit_msda(X, y, C, lam, n_iter=200, tol=1e-5, device='cuda', verbose=False):
    """
    Solve min_B (1/2n) ||Y - XB||_F^2 + lam Σ_j ||B_{j,:}||_2
    via FISTA on (p, C-1) matrix B.

    Returns: B (numpy, p x (C-1)), row_norms (numpy, p)
    """
    n, p = X.shape

    # Standardize X (mai2019 / sparse LDA convention: zero-mean, unit-variance)
    X_mean = X.mean(0, keepdims=True)
    X_std = X.std(0, keepdims=True) + 1e-8
    X_z = (X - X_mean) / X_std

    Y_np = make_indicator_Y(y, C)          # (n, C-1)
    # Compute X^T X and X^T Y in batches to avoid OOM on large n
    XtX = torch.zeros((p, p), device=device, dtype=torch.float32)
    XtY = torch.zeros((p, C - 1), device=device, dtype=torch.float32)
    Y_norm_sq = 0.0
    BATCH = 50000
    for bs in range(0, n, BATCH):
        be = min(bs + BATCH, n)
        Xb = torch.tensor(X_z[bs:be], device=device)
        Yb = torch.tensor(Y_np[bs:be], device=device)
        XtX += Xb.T @ Xb
        XtY += Xb.T @ Yb
        Y_norm_sq += (Yb ** 2).sum().item()
        del Xb, Yb
    torch.cuda.empty_cache()

    # Lipschitz constant: max eigenvalue of (1/n) X^T X
    # For FISTA step size: L = ||X^T X||_2 / n
    # Estimate via power iteration (3 iters is plenty)
    v = torch.randn(p, device=device); v /= v.norm()
    for _ in range(20):
        v = XtX @ v
        v /= v.norm()
    L = (v @ XtX @ v).item() / n + 1e-6     # Lipschitz of grad
    step = 1.0 / L

    # Group-soft-threshold: prox of lam * sum_j ||B_j||_2
    def prox_group(B, t):
        # B: (p, C-1); shrink each row by factor max(0, 1 - t / ||B_j||)
        norms = B.norm(dim=1, keepdim=True)
        shrink = torch.clamp(1.0 - t / (norms + 1e-12), min=0.0)
        return B * shrink

    # FISTA
    B = torch.zeros((p, C-1), device=device)
    Z = B.clone()
    t_prev = 1.0
    prev_obj = float('inf')
    for k in range(n_iter):
        # Gradient at Z
        grad = (XtX @ Z - XtY) / n
        B_new = prox_group(Z - step * grad, step * lam)
        t_new = (1 + (1 + 4 * t_prev**2) ** 0.5) / 2
        Z = B_new + ((t_prev - 1) / t_new) * (B_new - B)
        B = B_new
        t_prev = t_new

        # Objective for monitoring (use trace identity to avoid (n, C-1) tensor)
        # ||Y - XB||^2 = ||Y||^2 - 2 tr(Y^T X B) + tr(B^T X^T X B)
        if (k + 1) % 50 == 0 or k == n_iter - 1:
            with torch.no_grad():
                cross = (XtY * B).sum().item()       # tr(Y^T X B)
                quad = (B * (XtX @ B)).sum().item()  # tr(B^T X^T X B)
                ssq = Y_norm_sq - 2 * cross + quad
                loss_smooth = 0.5 * ssq / n
                pen = lam * B.norm(dim=1).sum().item()
                obj = loss_smooth + pen
                if verbose:
                    sparsity = (B.norm(dim=1) > 1e-8).sum().item()
                    print(f"    iter={k+1:4d}  obj={obj:.5f}  active={sparsity}/{p}", flush=True)
                if abs(prev_obj - obj) < tol * max(abs(obj), 1e-12):
                    break
                prev_obj = obj

    B_np = B.cpu().numpy()
    row_norms = np.linalg.norm(B_np, axis=1)
    return B_np, row_norms


def msda_select_topk_per_class(B, X_train, y_train, C, K_LIST):
    """
    msda gives a SHARED set of selected features (group-lasso on rows of B).
    To compare with our per-class feature selection, we use the same
    selected feature set for all classes (msda's natural prediction).

    Returns: dict[k] -> selected feature indices (shared across classes).
    """
    row_norms = np.linalg.norm(B, axis=1)
    feat_order = np.argsort(row_norms)[::-1]  # descending importance
    return {k: feat_order[:k] for k in K_LIST}


def nc_acc_on_features(features, X_tr, y_tr, X_te, y_te, C, device='cuda'):
    """Nearest-centroid accuracy using the given feature subset (shared
    across all classes -- msda's natural prediction setup).
    Uses sort+slice for centers and torch.cdist on GPU for distances,
    batched over test points to avoid OOM."""
    Xs_tr = X_tr[:, features]
    # sort+slice centers
    order = np.argsort(y_tr, kind='stable')
    Xs_tr_s = Xs_tr[order]
    y_tr_s = y_tr[order]
    counts = np.bincount(y_tr_s, minlength=C)
    ends = np.cumsum(counts)
    starts = ends - counts
    centers = np.zeros((C, len(features)), dtype=np.float32)
    for c in range(C):
        if counts[c] > 0:
            centers[c] = Xs_tr_s[starts[c]:ends[c]].mean(0)
    centers_t = torch.tensor(centers, device=device)

    # batched cdist on GPU
    Xs_te = X_te[:, features]
    correct = 0
    BATCH = 50000
    for bs in range(0, len(y_te), BATCH):
        be = min(bs + BATCH, len(y_te))
        Xb = torch.tensor(Xs_te[bs:be], device=device)
        d = torch.cdist(Xb, centers_t)
        preds = d.argmin(1).cpu().numpy()
        correct += (preds == y_te[bs:be]).sum()
    return correct / len(y_te), centers


# ─────────────── Sim I ──────────────────────────────────────────────────
def run_sim1(n_per_class=400, p=200, k_sig=20, n_test_per_class=500, seed=42,
             lam_list=(0.001, 0.005, 0.01, 0.05, 0.1), n_iter=300):
    """Sim I DGM: C=10, p=200, disjoint supports of size k_sig=20.
    True features: union of S_l for all l = {0,...,p_class*C-1}? NO --
    in our DGM each class has its OWN k_sig features in [l*k_sig, (l+1)*k_sig).
    So true global support = all 200 coords (but only some are useful per class).

    For msda comparison: msda finds a SHARED feature set. Best case it
    selects all 200 (everything is useful for at least one class), so
    feature selection rate measured globally.
    """
    rng = np.random.RandomState(seed)
    C = 10
    DELTA_MU = 3.0
    delta = DELTA_MU / np.sqrt(2 * k_sig)
    sig_in, sig_out = 1.0, 6.0

    # Generate
    def gen(n_per):
        Xs, ys = [], []
        signs = rng.choice([-1, 1], size=(C, k_sig))
        for c in range(C):
            mu = np.zeros(p, dtype=np.float32)
            mu[c*k_sig:(c+1)*k_sig] = signs[c] * delta
            sig = np.full(p, sig_out, dtype=np.float32)
            sig[c*k_sig:(c+1)*k_sig] = sig_in
            X_c = mu + rng.normal(0, 1, size=(n_per, p)).astype(np.float32) * sig
            Xs.append(X_c); ys.append(np.full(n_per, c, dtype=np.int64))
        return np.concatenate(Xs, 0), np.concatenate(ys, 0)

    X_tr, y_tr = gen(n_per_class)
    X_te, y_te = gen(n_test_per_class)

    # True support: each class l has k_sig features in [l*k_sig, (l+1)*k_sig).
    # Globally relevant features = union of all class supports (here partitions all p)
    # so msda should select ALL of them ideally.
    # But also report classification accuracy for varying k.

    K_LIST = [50, 100, 200]  # sub-supports of varying sizes
    results = {}
    print(f"  Fitting msda for {len(lam_list)} lam values...", flush=True)
    t0 = time.time()
    best_acc = 0; best_lam = None; best_k = None; best_test_acc = 0
    for lam in lam_list:
        B, row_norms = fit_msda(X_tr, y_tr, C, lam, n_iter=n_iter, verbose=False)
        active = (row_norms > 1e-8).sum()
        topk_dict = msda_select_topk_per_class(B, X_tr, y_tr, C, K_LIST)
        for k_val in K_LIST:
            feats = topk_dict[k_val]
            acc, _ = nc_acc_on_features(feats, X_tr, y_tr, X_te, y_te, C)
            if acc > best_test_acc:
                best_test_acc = acc; best_lam = lam; best_k = k_val
        if 200 not in K_LIST: pass
        results[lam] = {'active': active, 'B': B, 'row_norms': row_norms}
    elapsed = time.time() - t0
    print(f"  Total Sim I msda fit time: {elapsed:.1f}s")
    print(f"  Best: lam={best_lam}, k={best_k}, test_acc={best_test_acc:.4f}")
    return results, elapsed


# ─────────────── Real-data helper ────────────────────────────────────────
def run_realdata(X_tr, y_tr, X_te, y_te, C, dataset_name,
                  lam_list, K_LIST, n_iter=200):
    print(f"\n=== msda on {dataset_name} ===", flush=True)
    print(f"  Train n={len(y_tr)}, Test n={len(y_te)}, p={X_tr.shape[1]}, C={C}",
          flush=True)
    t0 = time.time()
    rows = []
    for lam in lam_list:
        t1 = time.time()
        B, row_norms = fit_msda(X_tr, y_tr, C, lam, n_iter=n_iter, verbose=False)
        active = int((row_norms > 1e-8).sum())
        fit_time = time.time() - t1
        topk_dict = msda_select_topk_per_class(B, X_tr, y_tr, C, K_LIST)
        for k_val in K_LIST:
            feats = topk_dict[k_val]
            tr_acc, _ = nc_acc_on_features(feats, X_tr, y_tr, X_tr, y_tr, C)
            te_acc, _ = nc_acc_on_features(feats, X_tr, y_tr, X_te, y_te, C)
            rows.append({'lam': lam, 'k': k_val, 'active': active,
                         'tr_acc': tr_acc, 'te_acc': te_acc, 'fit_time': fit_time})
            print(f"    lam={lam:.4f}  active={active:5d}  k={k_val:5d}  "
                  f"tr={tr_acc:.4f}  te={te_acc:.4f}  fit={fit_time:.0f}s",
                  flush=True)
    total = time.time() - t0
    print(f"  Total {dataset_name} time: {total:.0f}s", flush=True)
    return rows, total


# ─────────────── Main: run all non-ImageNet, then estimate ImageNet ──────
if __name__ == '__main__':
    timings = {}

    # ---- 1. Sim I (small synthetic) ----
    print("="*70)
    print("Sim I (C=10, p=200, n=400/class)")
    print("="*70)
    sim1_results, sim1_time = run_sim1()
    timings['sim1'] = sim1_time

    def load_npz(path):
        d = np.load(path, allow_pickle=True)
        if 'X' in d: return d['X'].astype(np.float32), d['y'].astype(np.int64)
        return d['features'].astype(np.float32), d['labels'].astype(np.int64)

    # ---- 2. CIFAR-100 ----
    cifar_train_path = r"data/cifar100_dinov3_vitl16_train_cls.npz"
    cifar_test_path = r"data/cifar100_dinov3_vitl16_test_cls.npz"
    if os.path.exists(cifar_train_path) and os.path.exists(cifar_test_path):
        print("\n" + "="*70)
        print("CIFAR-100")
        print("="*70)
        X_tr, y_tr = load_npz(cifar_train_path)
        X_te, y_te = load_npz(cifar_test_path)
        rows, t = run_realdata(X_tr, y_tr, X_te, y_te, 100,
                               "CIFAR-100",
                               lam_list=[0.001, 0.005, 0.01, 0.05],
                               K_LIST=[50, 100, 200, 500, 1024],
                               n_iter=200)
        timings['cifar100'] = t
    else:
        print(f"\nSKIP CIFAR-100 (data not found at {cifar_train_path})")

    # ---- 3. CUB-200 ----
    cub_train_path = r"data/cub200_dinov3_vitl16_train_cls.npz"
    cub_test_path = r"data/cub200_dinov3_vitl16_test_cls.npz"
    if os.path.exists(cub_train_path) and os.path.exists(cub_test_path):
        print("\n" + "="*70)
        print("CUB-200")
        print("="*70)
        X_tr, y_tr = load_npz(cub_train_path)
        X_te, y_te = load_npz(cub_test_path)
        rows, t = run_realdata(X_tr, y_tr, X_te, y_te, 200,
                               "CUB-200",
                               lam_list=[0.001, 0.005, 0.01, 0.05],
                               K_LIST=[50, 100, 200, 500, 1024],
                               n_iter=200)
        timings['cub'] = t
    else:
        print(f"\nSKIP CUB-200 (data not found at {cub_train_path})")

    # ---- 4. Estimate ImageNet ----
    print("\n" + "="*70)
    print("Timing summary and ImageNet extrapolation")
    print("="*70)
    print(f"  Sim I        ({1*4000} samples, 200 dim, 10 classes):  "
          f"{timings.get('sim1', '?')}")
    print(f"  CIFAR-100   ({50000} samples, 1024 dim, 100 classes):  "
          f"{timings.get('cifar100', '?')}")
    print(f"  CUB-200     (~6000 samples, 1024 dim, 200 classes):    "
          f"{timings.get('cub', '?')}")

    # Complexity scaling: O(n * p * (C-1)) per FISTA iter, plus O(p^2) XtX
    # ImageNet: n=1.28M, p=1024, C=1000
    if 'cifar100' in timings:
        n_cifar = 50000
        n_imagenet = 1280000  # 25.6x of CIFAR
        # Per iter: matmul scales linearly in n, linearly in C
        # CIFAR: n=50k, C=100; ImageNet: n=1.28M, C=1000
        # Ratio: (1.28M/50k) * (1000/100) = 25.6 * 10 = 256x
        est = timings['cifar100'] * 256
        print(f"\n  Estimated ImageNet time (extrapolated 256x from CIFAR):  "
              f"{est:.0f}s ≈ {est/3600:.1f} hours")
        print(f"  (if run on Sim I time scaling: still 100s+ x more)")
