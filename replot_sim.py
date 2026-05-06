"""Reload CSV and replot Sim I with n up to 6400 only."""
import os, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_results.csv")
OUT1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_results.png")
OUT2 = r"images/prospectus_sim_results.png"
OUT3 = r"images/prospectus_sim_results.png"

COL = {
    "Baseline":  "tab:blue",
    "FAIR":      "tab:orange",
    "MSDA":      "tab:purple",
    "Simple":    "tab:green",
    "Optimized": "tab:red",
}

df = pd.read_csv(CSV)
df = df[df["n_train"] <= 6400]   # trim to n ≤ 6400

x   = df["n_train"].values
ekw = dict(capsize=3)

# MSDA accuracy (mean ± std over 30 reps) from msda_sim1.py
msda_acc_mean = [0.1126, 0.1102, 0.1042, 0.0962, 0.0831, 0.0669, 0.0503, 0.0400]
msda_acc_std  = [0.0034, 0.0039, 0.0039, 0.0031, 0.0045, 0.0038, 0.0035, 0.0040]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.errorbar(x, df["acc_base_mean"],   yerr=df["acc_base_std"],
            marker="D", linestyle="--", color=COL["Baseline"],  label="Baseline",  **ekw)
ax.errorbar(x, df["acc_fair_mean"],   yerr=df["acc_fair_std"],
            marker="o",               color=COL["FAIR"],        label="FAIR",      **ekw)
ax.errorbar(x, msda_acc_mean,         yerr=msda_acc_std,
            marker="v",               color=COL["MSDA"],        label="MSDA",      **ekw)
ax.errorbar(x, df["acc_simple_mean"], yerr=df["acc_simple_std"],
            marker="s",               color=COL["Simple"],      label="Simple",    **ekw)
ax.errorbar(x, df["acc_opt_mean"],    yerr=df["acc_opt_std"],
            marker="^",               color=COL["Optimized"],   label="Optimized", **ekw)
ax.set_xscale("log")
ax.set_xlabel("n  (train samples per class)", fontsize=18)
ax.set_ylabel("Classification Accuracy", fontsize=18)
ax.set_title("Classification Accuracy vs n", fontsize=19)
ax.tick_params(labelsize=15)
ax.legend(fontsize=16); ax.grid(False)

ax = axes[1]
ax.errorbar(x, df["fs_fair_mean"],   yerr=df["fs_fair_std"],
            marker="o", color=COL["FAIR"],      label="FAIR",      **ekw)
ax.errorbar(x, df["fs_simple_mean"], yerr=df["fs_simple_std"],
            marker="s", color=COL["Simple"],    label="Simple",    **ekw)
ax.errorbar(x, df["fs_opt_mean"],    yerr=df["fs_opt_std"],
            marker="^", color=COL["Optimized"], label="Optimized", **ekw)
ax.set_xscale("log")
ax.set_xlabel("n  (train samples per class)", fontsize=18)
ax.set_ylabel("Feature Recovery Rate", fontsize=18)
ax.set_title("Feature Recovery Rate vs n", fontsize=19)
ax.tick_params(labelsize=15)
ax.legend(fontsize=16); ax.grid(False)

plt.tight_layout()
for out in [OUT1, OUT2, OUT3]:
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close()
