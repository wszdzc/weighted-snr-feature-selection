"""Generate LaTeX rows for appendix realdata_full table from bootstrap CSV."""
import pandas as pd

df = pd.read_csv(r'data/realdata_bootstrap_5seed.csv')
non_base = df[df['method'] != 'Baseline']
agg = non_base.groupby(['dataset', 'method', 'k']).agg(
    train_mean=('train', 'mean'), train_std=('train', 'std'),
    test_mean=('test', 'mean'),   test_std=('test', 'std')
).reset_index()

K_LIST = [50, 100, 200, 300, 500, 700, 900, 1000]
DATASETS = ['CIFAR-100', 'CUB-200', 'ImageNet']
METHODS = ['Simple', 'FAIR', 'MSDA', 'Optimized']

# Find best per (dataset, method) for train and test
best = {}
for ds in DATASETS:
    for m in METHODS:
        sub = agg[(agg['dataset'] == ds) & (agg['method'] == m)]
        best[(ds, m, 'train')] = sub.loc[sub['train_mean'].idxmax(), 'k']
        best[(ds, m, 'test')]  = sub.loc[sub['test_mean'].idxmax(), 'k']

def fmt_cell(mean, std, bold):
    s = f"${mean:.3f}_{{\\pm{std:.3f}}}$"
    if bold:
        s = f"$\\mathbf{{{mean:.3f}}}_{{\\pm{std:.3f}}}$"
    return s

method_label = {
    'Simple':    r'\multicolumn{7}{l}{\emph{Simple}}\\',
    'FAIR':      r'\multicolumn{7}{l}{\emph{FAIR \citep{FanFan2008}}}\\',
    'MSDA':      r'\multicolumn{7}{l}{\emph{MSDA \citep{mai2019multiclass}}}\\',
    'Optimized': r'\multicolumn{7}{l}{\emph{Optimized (ours)}}\\',
}

print('Baseline')
base = df[df['method'] == 'Baseline']
b = []
for ds in DATASETS:
    row = base[base['dataset'] == ds].iloc[0]
    b += [f"${row['train']:.3f}$", f"${row['test']:.3f}$"]
print('Baseline & ' + ' & '.join(b) + ' \\\\')
print(r'\midrule')

for m in METHODS:
    print(method_label[m])
    for k in K_LIST:
        row_cells = [f"$k{{=}}{k}$"]
        for ds in DATASETS:
            sub = agg[(agg['dataset'] == ds) & (agg['method'] == m) & (agg['k'] == k)]
            if len(sub) == 0:
                row_cells += ['---', '---']
                continue
            r = sub.iloc[0]
            row_cells.append(fmt_cell(r['train_mean'], r['train_std'], best[(ds, m, 'train')] == k))
            row_cells.append(fmt_cell(r['test_mean'],  r['test_std'],  best[(ds, m, 'test')]  == k))
        print(' & '.join(row_cells) + ' \\\\')
    print(r'\midrule')
