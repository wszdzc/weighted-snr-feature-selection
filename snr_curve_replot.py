"""
snr_curve_replot.py
Replot the per-class SNR curve (baseline vs Optimized top-k=100) on ImageNet
with larger font sizes (legend + axis labels close to caption font size).

SNR_l(w) = min_{l' != l} (sum_j w_j |mu_lj - mu_l'j|) / (sum_j w_j sigma_lj)

Two curves:
- Baseline: w = uniform over all p features
- Optimized: w = uniform over top-k=100 features selected per class
Classes sorted by Baseline SNR.
"""
import os, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import matplotlib.pyplot as plt

device = "cpu"
print(f"Device: {device}", flush=True)

# -- Load data -----------------------------------------------------------------
def load_npz(path):
    d = np.load(path, allow_pickle=True)
    if 'X' in d: return d['X'].astype(np.float32), d['y'].astype(np.int64)
    return d['features'].astype(np.float32), d['labels'].astype(np.int64)

X_tr, y_tr = load_npz(r"data/dinov3_train_fixed.npz")
C = int(y_tr.max()) + 1
p = X_tr.shape[1]
print(f"C={C}, p={p}, n_train={len(y_tr)}", flush=True)

# -- Pre-sort by class for fast per-class stats --------------------------------
print("Pre-sorting...", flush=True)
order = np.argsort(y_tr, kind='stable')
X_s = X_tr[order]
y_s = y_tr[order]
counts = np.bincount(y_s, minlength=C)
ends = np.cumsum(counts)
starts = ends - counts

# -- Centers (cached) and sigmas ----------------------------------------------
CENTERS_NPZ = r"data/imagenet_centers.npz"
SIGMAS_NPZ = r"data/imagenet_sigmas.npz"

print("Loading centers...", flush=True)
centers = np.load(CENTERS_NPZ)['centers'].astype(np.float32)

if os.path.exists(SIGMAS_NPZ):
    print(f"Loading cached sigmas from {SIGMAS_NPZ}...", flush=True)
    sigmas = np.load(SIGMAS_NPZ)['sigmas'].astype(np.float32)
else:
    print("Computing sigmas (sort+slice)...", flush=True)
    sigmas = np.zeros((C, p), dtype=np.float32)
    for c in range(C):
        sigmas[c] = X_s[starts[c]:ends[c]].std(axis=0) + 1e-8
    np.savez(SIGMAS_NPZ, sigmas=sigmas)
    print(f"  saved -> {SIGMAS_NPZ}", flush=True)

# -- Optimized weights (already trained) ---------------------------------------
W = np.load(r"data/cil_weights_all_1000_parallel.npz",
            allow_pickle=True)['weights'].astype(np.float32)
print(f"Loaded weights: {W.shape}", flush=True)


# -- Compute per-class SNR for given weight vectors -----------------------
def compute_snr(centers_t, sigmas_t, w):
    """
    centers_t: (C, p) torch tensor
    sigmas_t:  (C, p) torch tensor
    w: (C, p) torch tensor — per-class weight vector (rows sum to 1, w_j>=0)
    Returns: (C,) numpy array of SNR_l(w_l)
    """
    snrs = np.zeros(C, dtype=np.float32)
    for l in range(C):
        mu_l = centers_t[l]                        # (p,)
        D = (centers_t - mu_l).abs()               # (C, p)
        D_other = torch.cat([D[:l], D[l+1:]], 0)   # (C-1, p)
        sig_l = sigmas_t[l]                        # (p,)
        wl = w[l]                                  # (p,)
        signal = (D_other @ wl).min().item()
        noise = (sig_l @ wl).item()
        snrs[l] = signal / max(noise, 1e-12)
    return snrs


# -- Build baseline weights (uniform) and optimized weights (top-k=100 mask) --
K = 100

W_base = np.full((C, p), 1.0 / p, dtype=np.float32)

W_opt_topk = np.zeros((C, p), dtype=np.float32)
for c in range(C):
    topk_idx = np.argsort(W[c])[-K:]
    W_opt_topk[c, topk_idx] = 1.0 / K

print("Computing SNRs...", flush=True)
centers_t = torch.tensor(centers, device=device)
sigmas_t = torch.tensor(sigmas, device=device)
w_base_t = torch.tensor(W_base, device=device)
w_opt_t = torch.tensor(W_opt_topk, device=device)

t0 = time.time()
snr_base = compute_snr(centers_t, sigmas_t, w_base_t)
print(f"  baseline: {time.time()-t0:.1f}s", flush=True)
t0 = time.time()
snr_opt = compute_snr(centers_t, sigmas_t, w_opt_t)
print(f"  optimized: {time.time()-t0:.1f}s", flush=True)

# -- Sort classes by baseline SNR ---------------------------------------------
order_snr = np.argsort(snr_base)
snr_base_sorted = snr_base[order_snr]
snr_opt_sorted = snr_opt[order_snr]

# -- Plot with bigger fonts ----------------------------------------------------
print("Plotting...", flush=True)
fig, ax = plt.subplots(figsize=(13, 4))
x_axis = np.arange(C)
ax.plot(x_axis, snr_base_sorted, color="steelblue", lw=1.6,
        label=f"Baseline (all {p} features)")
ax.plot(x_axis, snr_opt_sorted, color="tomato", lw=1.6,
        label=f"Optimized (top-$k$={K})")

# Larger font sizes (close to caption text size)
ax.set_xlabel("Class index (sorted by baseline SNR)", fontsize=18)
ax.set_ylabel("SNR$_l(\\mathbf{w})$", fontsize=18)
ax.tick_params(axis='both', labelsize=15)
ax.legend(fontsize=16, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()

out_png1 = r"data/snr_curves_sorted.png"
out_png2 = r"c:\Users\user\693652c3749f1dbacbf22795\images\prospectus_imagenet_sorted_snr_k100.png"
out_png3 = r"images/prospectus_imagenet_sorted_snr_k100.png"
plt.savefig(out_png1, dpi=200)
plt.savefig(out_png2, dpi=200)
plt.savefig(out_png3, dpi=200)
print(f"Saved figures.", flush=True)
print("Done.", flush=True)
