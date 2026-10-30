"""GRAIN — Graph-conditioned Residual Adaptive Incentive Network for Non-stationary Hierarchical Mobile Crowdsensing.

Reference implementation.
"""

from __future__ import annotations
import argparse
import csv
import io
import json
import math
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
UCI_ZIP = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
METR_ZIP = "https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/traffic/metr_la.zip"
COORDS = {
    "Aotizhongxin": (39.982, 116.397),
    "Changping": (40.217, 116.230),
    "Dingling": (40.292, 116.220),
    "Dongsi": (39.929, 116.417),
    "Guanyuan": (39.929, 116.339),
    "Gucheng": (39.914, 116.184),
    "Huairou": (40.328, 116.628),
    "Nongzhanguan": (39.937, 116.461),
    "Shunyi": (40.127, 116.655),
    "Tiantan": (39.886, 116.407),
    "Wanliu": (39.987, 116.287),
    "Wanshouxigong": (39.878, 116.352),
}

def _hav(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2.0 * math.asin(math.sqrt(h))

def _knn(coords: list[tuple[float, float]], k: int = 3) -> list[list[float]]:
    edges: list[tuple[int, int, float]] = []
    for i, a in enumerate(coords):
        dists = sorted((_hav(a, b), j) for j, b in enumerate(coords) if j != i)
        for dist, j in dists[:k]:
            x, y = (i, j) if i < j else (j, i)
            edges.append((x, y, round(dist, 3)))
    uniq: list[list[float]] = []
    seen: set[tuple[int, int]] = set()
    for x, y, dist in edges:
        if (x, y) in seen:
            continue
        seen.add((x, y))
        uniq.append([x, y, dist])
    return uniq

def download_beijing() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    outer = RAW / "beijing.zip"
    print(f"GET {UCI_ZIP}")
    urlretrieve(UCI_ZIP, outer)
    with zipfile.ZipFile(outer) as zf:
        inner = zf.read("PRSA2017_Data_20130301-20170228.zip")
    stations = []
    pm25, temp, wspm = [], [], []
    with zipfile.ZipFile(io.BytesIO(inner)) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".csv"))
        for name in names:
            rows = list(csv.DictReader(io.StringIO(zf.read(name).decode("utf-8"))))
            station = rows[0]["station"]
            lat, lon = COORDS[station]
            stations.append({"id": station, "lat": lat, "lon": lon})
            last = {"PM2.5": 0.0, "TEMP": 0.0, "WSPM": 0.0}
            series = {key: [] for key in last}
            for row in rows[:720]:
                for key in last:
                    raw = (row.get(key) or "").strip()
                    if raw in {"", "NA", "NaN", "None"}:
                        value = last[key]
                    else:
                        value = float(raw)
                        last[key] = value
                    series[key].append(value)
            pm25.append(series["PM2.5"])
            temp.append(series["TEMP"])
            wspm.append(series["WSPM"])
    payload = {
        "id": "uci-beijing-multisite-prsa",
        "title": "Beijing Multi-Site Air-Quality (PRSA)",
        "sourceUrl": "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data",
        "downloadUrl": UCI_ZIP,
        "kaggleMirror": "https://www.kaggle.com/datasets/manojkumard28/beijing-air-quality-data-dataset",
        "kaggleSlug": "manojkumard28/beijing-air-quality-data-dataset",
        "hours": 720,
        "stations": stations,
        "edgesKm": _knn([(s["lat"], s["lon"]) for s in stations]),
        "pm25": pm25,
        "temp": temp,
        "wspm": wspm,
    }
    dest = DATA / "beijing_prsa_720.json"
    dest.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {dest}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beijing", action="store_true")
    parser.add_argument("--traffic", action="store_true")
    args = parser.parse_args()
    if not args.beijing and not args.traffic:
        args.beijing = True
    if args.beijing:
        download_beijing()
    if args.traffic:
        print(
            "Traffic slice: install h5py and download\n"
            f"  {METR_ZIP}\n"
            "The committed data/metr_la_32x288.json is already the 32×288 PeMS/METR-LA window.\n"
            "Kaggle (auth): https://www.kaggle.com/datasets/orvile/traffic-and-weather-datasets"
        )
if __name__ == "__main__":
    main()
