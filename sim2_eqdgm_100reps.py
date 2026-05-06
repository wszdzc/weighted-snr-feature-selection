"""
sim2_eqdgm.py
-------------
Simulation II: n_theory vs n_emp for equally-spaced means DGM.
Fixed k=5, varying p. Goal: show both grow as log(p) (same rate).

DGM: C=10 classes, k=5 signal features (first k coords).
     mu[c,j] = c * delta_mu  for j < k,  else 0.
     All features Gaussian noise sigma=1 (homoscedastic).

Parameters: delta_mu=5.0, sigma=1.0
  M_l     = delta_mu / sigma = 5.0   (continuous SNR, LP-exact)
  Delta_l = delta_mu / (k*sigma) = 1.0  (discrete SNR gap, A4)

n_theory : binary search  2*eps_n(n,p,M_l) < Delta_l
n_emp    : adaptive 2^(1/4) grid from n_start upward;
           P(exact recovery of {0,...,k-1}) >= 0.99 over REPS replicates.
"""

import os, math, csv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import matplotlib.pyplot as plt

# ── Parameters ────────────────────────────────────────────────────────────
C          = 10
K          = 5
DELTA_MU   = 0.5
SIGMA      = 1.0
DELTA      = 0.01     # theory/empirical failure probability
TARGET     = 0.99     # empirical target P(exact recovery)
REPS       = 100      # replicates per (n, p) evaluation (was 20, now 100 per advisor)
LR         = 0.01
STEPS      = 300
P_LIST     = [100, 200, 400, 800, 1600, 3200]
BASE_SEED  = 2026
GRID_STEP  = 2 ** 0.25   # ≈ 1.189

M_L     = DELTA_MU / SIGMA            # = 5.0
DELTA_L = DELTA_MU / (K * SIGMA)     # = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"C={C}, k={K}, delta_mu={DELTA_MU}, sigma={SIGMA}, delta={DELTA}")
print(f"M_l={M_L:.3f},  Delta_l={DELTA_L:.4f}")
print()

# ── DGM sampling ──────────────────────────────────────────────────────────
def sample_data(p, n, seed):
    rng = np.random.default_rng(seed)
    data = {}
    for c in range(C):
        mu = np.zeros(p)
        mu[:K] = c * DELTA_MU
        X = mu + rng.normal(0, SIGMA, size=(n, p))
        data[c] = torch.tensor(X, dtype=torch.float32, device=device)
    return data

def compute_centers(data):
    return torch.stack([data[c].mean(0) for c in range(C)])  # (C, p)

# ── Adam SNR optimizer for one class ──────────────────────────────────────
def optimize_class(l, centers_t, lr, steps):
    mu_l    = centers_t[l]
    D       = (centers_t - mu_l).abs()          # (C, p)
    D_other = torch.cat([D[:l], D[l+1:]], 0)   # (C-1, p)

    u   = torch.zeros(centers_t.shape[1], device=device, requires_grad=True)
    opt = torch.optim.Adam([u], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        w     = torch.softmax(u, 0)
        snr   = (D_other @ w).min() / SIGMA
        (-snr).backward()
        opt.step()
    with torch.no_grad():
        w = torch.softmax(u, 0).cpu().numpy()
    return w

# ── Exact recovery check (all C classes) ──────────────────────────────────
def check_recovery(p, n, seed):
    data      = sample_data(p, n, seed)
    centers_t = compute_centers(data)
    true_set  = set(range(K))
    for l in range(C):
        w     = optimize_class(l, centers_t, LR, STEPS)
        top_k = set(np.argsort(w)[-K:].tolist())
        if top_k != true_set:
            return False
    return True

def eval_prob(p, n, reps):
    successes = 0
    for r in range(reps):
        seed = (BASE_SEED * 31337 + p * 997 + n * 53 + r * 104729) % (2**32 - 1)
        if check_recovery(p, n, seed):
            successes += 1
    return successes / reps

# ── n_theory (theorem formula) ────────────────────────────────────────────
def eps_n(n, p):
    L1 = math.log(4 * p * C / DELTA)
    L2 = math.log(8 * p / DELTA)
    CA = 2 * math.sqrt(2 * L1 / n)
    CB = 2 * math.sqrt(2 * L2 / n) + 6 * L2 / n
    if CB >= 1:
        return float('inf')
    return (CA + M_L * CB) / (1 - CB)

def compute_n_theory(p):
    lo, hi = 1, 10_000_000
    while lo < hi:
        mid = (lo + hi) // 2
        if 2 * eps_n(mid, p) < DELTA_L:
            hi = mid
        else:
            lo = mid + 1
    return lo

# ── n_emp search with 2^(1/4) grid ───────────────────────────────────────
def find_n_emp(p):
    n = 5
    while n <= 500_000:
        prob = eval_prob(p, n, REPS)
        print(f"    n={n:7d}  P={prob:.3f}")
        if prob >= TARGET:
            return n
        n = max(n + 1, int(n * GRID_STEP))
    return None

# ── Main loop ─────────────────────────────────────────────────────────────
results = []
for p in P_LIST:
    n_th = compute_n_theory(p)
    print(f"\np={p}:  n_theory = {n_th}")
    n_emp = find_n_emp(p)
    print(f"  n_emp = {n_emp}")
    results.append({"p": p, "n_theory": n_th, "n_emp": n_emp if n_emp else -1})

# ── Summary table ─────────────────────────────────────────────────────────
print("\n\n=== Summary Table ===")
print(f"{'p':>6}  {'n_theory':>10}  {'n_emp':>8}")
print("-" * 30)
for r in results:
    emp_str = str(r["n_emp"]) if r["n_emp"] > 0 else ">500k"
    print(f"{r['p']:>6}  {r['n_theory']:>10}  {emp_str:>8}")

# ── Save CSV ──────────────────────────────────────────────────────────────
out_csv = r"data/sim2_eqdgm_results_100reps.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["p", "n_theory", "n_emp"])
    writer.writeheader()
    writer.writerows(results)
print(f"\nSaved CSV: {out_csv}")

# ── Plot ──────────────────────────────────────────────────────────────────
ps       = [r["p"] for r in results]
n_th_arr = [r["n_theory"] for r in results]
ps_emp   = [r["p"]  for r in results if r["n_emp"] > 0]
n_emp_arr= [r["n_emp"] for r in results if r["n_emp"] > 0]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ps, n_th_arr, "o-", color="steelblue", lw=2, label="n_theory")
ax.plot(ps_emp, n_emp_arr, "s--", color="tomato", lw=2, label="n_emp")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(ps)
ax.set_xticklabels([str(p) for p in ps])
ax.set_xlabel("p", fontsize=13)
ax.set_ylabel("n", fontsize=13)
ax.set_title("Recovery threshold n vs p  (k=5)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(False)
plt.tight_layout()

out_png1 = r"data/sim2_eqdgm_plot.png"
out_png2 = r"c:\Users\user\693652c3749f1dbacbf22795\images\prospectus_sim2_threshold.png"
plt.savefig(out_png1, dpi=200)
plt.savefig(out_png2, dpi=200)
print(f"Saved figures.")
