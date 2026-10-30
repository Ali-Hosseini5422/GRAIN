"""GRAIN — Graph-conditioned Residual Adaptive Incentive Network for Non-stationary Hierarchical Mobile Crowdsensing.

Reference implementation.
"""

from __future__ import annotations
import json
from pathlib import Path
from grain.simulate import Trace

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BEIJING_JSON = DATA_DIR / "beijing_prsa_720.json"
TRAFFIC_JSON = DATA_DIR / "metr_la_32x288.json"
SOURCES = {
    "beijing": {
        "title": "Beijing Multi-Site Air-Quality (PRSA)",
        "uci": "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data",
        "download": "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip",
        "doi": "10.24432/C5RK5G",
        "kaggle": "https://www.kaggle.com/datasets/manojkumard28/beijing-air-quality-data-dataset",
        "kaggle_slug": "manojkumard28/beijing-air-quality-data-dataset",
        "window": "12 stations × first 720 hours from 2013-03-01, NA forward-filled",
    },
    "traffic": {
        "title": "METR-LA traffic speed (Caltrans PeMS)",
        "paper": "https://github.com/liyaguang/DCRNN",
        "download": "https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/traffic/metr_la.zip",
        "pems": "https://pems.dot.ca.gov/",
        "kaggle": "https://www.kaggle.com/datasets/orvile/traffic-and-weather-datasets",
        "kaggle_slug": "orvile/traffic-and-weather-datasets",
        "window": "32 sensors × first 288 steps (5 min), farthest-point sample, no missing values",
    },
}

def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)

def load_beijing(path: Path | None = None) -> Trace:
    obj = _load(path or BEIJING_JSON)
    pm = obj["pm25"]
    stations = len(pm)
    hours = len(pm[0])
    signal = [[min(pm[s][t] / 250.0, 1.0) for s in range(stations)] for t in range(hours)]
    edges = [(int(e[0]), int(e[1])) for e in obj["edgesKm"]]
    return Trace(kind="air", nodes=stations, edges=edges, value_signal=signal)

def load_traffic(path: Path | None = None) -> Trace:
    obj = _load(path or TRAFFIC_JSON)
    speed = obj["speed"]
    sensors = len(speed)
    steps = len(speed[0])
    signal = [
        [max(0.0, min(1.0, 1.0 - speed[s][t] / 75.0)) for s in range(sensors)] for t in range(steps)
    ]
    edges = [(int(e[0]), int(e[1])) for e in obj["edgesKm"]]
    return Trace(kind="traffic", nodes=sensors, edges=edges, value_signal=signal)

def load_dataset(name: str) -> Trace:
    if name in {"air", "beijing", "knowair"}:
        return load_beijing()
    if name in {"traffic", "pems", "metr"}:
        return load_traffic()
    raise ValueError(f"unknown dataset {name!r}; use 'air' or 'traffic'")
