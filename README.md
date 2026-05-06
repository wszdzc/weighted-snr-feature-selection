# Weighted-SNR Feature Selection — Code release

This repository contains the code accompanying the NeurIPS 2026 submission
"Weighted-SNR Feature Selection for Multi-class Classification" (paper ID
withheld for double-blind review).

## Layout

```
.
├── mai2019_msda.py                 # MSDA baseline (Mai et al. 2019), FISTA from scratch
├── msda_sim1.py                    # Sim I: MSDA evaluation (used for Sim I Table 1 row)
├── replot_sim.py                   # Sim I figure (Figure 1 in paper)
├── sim2_eqdgm_100reps.py           # Sim II: 100-replicate threshold experiment
├── replot_sim2.py                  # Sim II figure (Figure 2 in paper)
├── realdata_bootstrap_5seed.py     # Real-data 90% bootstrap, 5 seeds, all methods
├── fewshot_5seed.py                # Few-shot ImageNet 5-seed experiment
├── snr_curve_replot.py             # SNR curve figure on ImageNet (Figure 3)
├── gen_realdata_full_table.py      # Builds the LaTeX rows for Appendix B's full k-sweep table
├── *.csv                           # Saved experimental outputs used by the figure scripts
├── requirements.txt
└── README.md
```

## Reproduction map

| Paper artifact            | Script                                  | Output                                  |
|--------------------------:|:----------------------------------------|:----------------------------------------|
| Sim I (Table 1, Figure 1) | `replot_sim.py` (reads `sim_results.csv`) | `prospectus_sim_results.png`            |
| Sim I MSDA row            | `msda_sim1.py`                          | console output (mean +/- std vs n)      |
| Sim II (Table 2, Figure 2)| `sim2_eqdgm_100reps.py` then `replot_sim2.py` | `sim2_eqdgm_results_100reps.csv` + PNG |
| Real-data Tables 6 & B    | `realdata_bootstrap_5seed.py`           | `realdata_bootstrap_5seed.csv`          |
| Real-data LaTeX rows      | `gen_realdata_full_table.py`            | stdout                                  |
| Few-shot table & figure   | `fewshot_5seed.py`                      | `fewshot_5seed.csv`                     |
| SNR curve figure          | `snr_curve_replot.py`                   | `prospectus_imagenet_sorted_snr_k100.png`|

## Data

The scripts expect frozen DINOv3 ViT-L/16 features (`p=1024`) saved as
`.npz` files in a local `data/` directory. Each file contains
`X` (`float32`, `[n, p]`) and `y` (`int64`, `[n]`). Expected names:

```
data/
├── cifar100_dinov3_vitl16_train_cls.npz
├── cifar100_dinov3_vitl16_test_cls.npz
├── cub200_dinov3_vitl16_train_cls.npz
├── cub200_dinov3_vitl16_test_cls.npz
├── dinov3_train_fixed.npz       # ImageNet train features
└── dinov3_val_fixed.npz         # ImageNet val features
```

The DINOv3 ViT-L/16 encoder is publicly available; please follow the
authors' instructions to extract features. CIFAR-100, CUB-200-2011,
and ImageNet are used under their standard research-use terms.

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.11, PyTorch 2.7 + CUDA 11.8, on a single NVIDIA
RTX 3060 (8.6 GB).

## Notes

- The proposed weighted-SNR Optimized selector uses Adam with
  `lr=0.01` and 30,000 steps. The weight optimization is vectorized
  across classes and chunked to fit in 8 GB of VRAM.
- MSDA is implemented from scratch as accelerated proximal gradient
  descent (FISTA) on a class-aligned indicator-matrix regression
  objective; lambda is tuned per `k` on test accuracy.
- Bootstrap subsampling draws 90% of training samples per class
  (without replacement) for each of 5 seeds.
