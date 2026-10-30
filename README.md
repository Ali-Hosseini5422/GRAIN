# GRAIN

**GRAIN — Graph-conditioned Residual Adaptive Incentive Network for Non-stationary Hierarchical Mobile Crowdsensing.**

Complete reference simulation of GRAIN. The manuscript equations, Algorithm 1, and the printed paper numbers are kept separate:

- `python -m grain --check` verifies every printed claim to the published digit (14.3%, 27.1%, 34%, and the regret curves ending at `(1000, 54)` and `(1000, 102)`).
- `python -m grain --dataset air` runs Algorithm 1 on a real data slice and writes plots and CSVs. That run is the reference implementation; it is not a copy of the synthesized KnowAir table. The paper presents that table as component-wise evidence.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer. The only runtime dependency is `matplotlib`.

## Run

```bash
python -m grain --check
python -m grain --dataset air --sigma0 0.08 --out results
python -m grain --dataset traffic --out results
python -m grain --all --out results
python -m unittest discover -s tests
```

`--sigma0` must stay inside the manuscript interval `[0.05, 0.15]`. The default seed is `20260301`, and every run is deterministic.

Output under `results/`:

| File | Contents |
| --- | --- |
| `fig_manuscript_regret.png` | GRAIN regret curves drawn from the exact TikZ knots |
| `fig_manuscript_sw.png` | KnowAir social-welfare table |
| `fig_manuscript_ablation.png` | Manuscript ablation |
| `fig_sim_regret_air.png` | Cumulative regret of the reference run |
| `fig_sim_objectives_*.png` | Six objectives of the reference run |
| `tables/knowair_main.csv` | Main table, digit for digit |
| `rounds_air.csv` | Per-round time series |

## Data

The slices in `data/` let the project run offline. Rebuild from upstream with:

```bash
python scripts/download_slices.py --beijing
```

| Role in the paper | Real link | What the repository holds |
| --- | --- | --- |
| Beijing / KnowAir, 720 hours | [UCI 501](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) and [Kaggle](https://www.kaggle.com/datasets/manojkumard28/beijing-air-quality-data-dataset) | 12 stations, first 720 hours from 2013-03-01 |
| California PeMS, 32 sensors × 288 steps | [METR-LA / DCRNN](https://github.com/liyaguang/DCRNN), [Hugging Face](https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/traffic/metr_la.zip), [PeMS](https://pems.dot.ca.gov/) and [Kaggle](https://www.kaggle.com/datasets/orvile/traffic-and-weather-datasets) | 32 sensors and 288 five-minute steps |

The Kaggle traffic bundle is about 875 MB and requires an account. The committed traffic slice is the public PeMS extract (METR-LA) with the same dimensions as the paper.

Kagglehub identifiers:

```python
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "manojkumard28/beijing-air-quality-data-dataset", "")
kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "orvile/traffic-and-weather-datasets", "")
```

## What is implemented

- Graph state update `X_{t+1} = A X_t + B U_t + W_t`
- `α_t = 1 + η P̂_t / (1 + P̂_t)` and `σ_t = σ_0 (1 + γ P̂_t / t)` with `γ = 1`, matching Algorithm 1
- Four-head residual cross-attention and top-2 expert routing
- Smoothed critical-cost payments, collusion groups of size 3, and an anticipation factor in `[0.85, 1.15]`
- Lipschitz lemma `L = L_0 (1 + L_RCAS + L_MoE)` and step size `η_t = t^{-1/3} (P̂_t L)^{-1/3}`
- Explicit bound `ε = σ √log(1/δ) + N^{-1/2}` with leading constant 1 (the unit-constant reading of the paper’s big-O, not a fitted curve)

The hidden dimension of the run is `d = 8` so the simulation stays interactive on a laptop. The paper reports `d = 64` for the edge budget; the operators are the same equations.

## Layout

```
grain/           Python package
tests/           Manuscript-digit checks, RNG, and determinism
data/            Public slices
scripts/         Rebuild data from the upstream links
results/         Created by a run
```

## GitHub

```bash
cd grain
git init
git add .
git commit -m "Add the GRAIN reference simulation"
git branch -M main
git remote add origin git@github.com:<user>/grain.git
git push -u origin main
```

## Paper authors

Seyed Ali Hosseini, Omid Sojodishijani, and Vahid Khajehvand. Department of Computer, Qa.c., Islamic Azad University, Qazvin, Iran.
