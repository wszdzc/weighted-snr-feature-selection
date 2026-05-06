"""Replot Sim II threshold figure from existing CSV with larger fonts."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = r"data/sim2_eqdgm_results_100reps.csv"
OUT_PROSPECTUS = r"images/prospectus_sim2_threshold.png"
OUT_NEURIPS    = r"images/prospectus_sim2_threshold.png"

df = pd.read_csv(CSV)
ps = df["p"].tolist()
n_th = df["n_theory"].tolist()
n_emp = df["n_emp"].tolist()

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ps, n_th, "o-", color="steelblue", lw=2, label="n_theory")
ax.plot(ps, n_emp, "s--", color="tomato", lw=2, label="n_emp")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(ps)
ax.set_xticklabels([str(p) for p in ps])
ax.set_xlabel("p", fontsize=18)
ax.set_ylabel("n", fontsize=18)
ax.set_title("Recovery threshold n vs p  (k=5)", fontsize=19)
ax.tick_params(labelsize=15)
ax.legend(fontsize=16)
ax.grid(False)
plt.tight_layout()

for out in [OUT_PROSPECTUS, OUT_NEURIPS]:
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close()
