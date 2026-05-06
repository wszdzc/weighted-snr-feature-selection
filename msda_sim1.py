"""MSDA on Sim I (disjoint-support, C=10, p=200, k_sig=20)."""
import os, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys; sys.path.insert(0, r'data/Claude')
import numpy as np
from mai2019_msda import fit_msda, nc_acc_on_features

C = 10
p = 200
k_sig = 20
DELTA_MU = 3.0
delta_mu = DELTA_MU / np.sqrt(2 * k_sig)
sig_in, sig_out = 1.0, 6.0
n_test_per = 500
REPEATS = 30
N_LIST = [50, 100, 200, 400, 800, 1600, 3200, 6400]
LAM_LIST = [0.001, 0.01, 0.05, 0.1, 0.5]
K_FEAT_LIST = [50, 100, 200]  # for top-k after msda

def gen(n_per, seed):
    rng = np.random.RandomState(seed)
    Xs, ys = [], []
    signs = rng.choice([-1, 1], size=(C, k_sig))
    for c in range(C):
        mu = np.zeros(p, dtype=np.float32)
        mu[c*k_sig:(c+1)*k_sig] = signs[c] * delta_mu
        sig = np.full(p, sig_out, dtype=np.float32)
        sig[c*k_sig:(c+1)*k_sig] = sig_in
        X_c = mu + np.random.randn(n_per, p).astype(np.float32) * sig
        Xs.append(X_c); ys.append(np.full(n_per, c, dtype=np.int64))
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


print(f"{'n':>5}  {'best_lam':>9}  {'best_k':>7}  {'mean_acc':>9}  {'std':>6}", flush=True)
results = {}
for n_per in N_LIST:
    accs_per_rep = []
    for rep in range(REPEATS):
        seed = 2026 * 31 + n_per * 7 + rep * 13
        X_tr, y_tr = gen(n_per, seed)
        X_te, y_te = gen(n_test_per, seed + 1_000_000)
        best_acc = 0.0
        for lam in LAM_LIST:
            B, rn = fit_msda(X_tr, y_tr, C, lam, n_iter=200)
            feat_order = np.argsort(rn)[::-1]
            for k_val in K_FEAT_LIST:
                feats = feat_order[:k_val]
                acc, _ = nc_acc_on_features(feats, X_tr, y_tr, X_te, y_te, C)
                if acc > best_acc:
                    best_acc = acc
        accs_per_rep.append(best_acc)
    accs_per_rep = np.array(accs_per_rep)
    mean = accs_per_rep.mean(); std = accs_per_rep.std()
    print(f"{n_per:>5}  {'-':>9}  {'best':>7}  {mean:>9.4f}  {std:>6.4f}", flush=True)
    results[n_per] = (mean, std)

print("\n=== Summary ===")
for n_per in N_LIST:
    m, s = results[n_per]
    print(f"  n={n_per:5d}: {m:.4f} +/- {s:.4f}")
