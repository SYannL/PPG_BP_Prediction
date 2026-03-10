# March PPG → SBP (Deep Learning, Minimal)

This repository is a **minimal, reproducible** project for:

- Preprocessing: `PPG(finger+wrist) + HR → SBP`
- Nature-style preprocessing visualizations (PNG)
- A PyTorch deep learning baseline (GPU / Colab)

Note: by default, raw data is not meant to be committed to GitHub. You can either:
- **Recommended (traceable)**: start from raw `.bin/.b` and generate `derived/` tables via scripts
- **Quick**: directly provide `derived/finger.csv`, `derived/wrist.csv`, `derived/labels.csv`

---

## Project structure

```
PPG_BP_Prediction/
├── data/
│   ├── raw/
│   │   ├── New data.xlsx
│   │   ├── recordsfinger/
│   │   └── recordswrist/
│   ├── ppg_csv/
│   │   ├── finger/
│   │   └── wrist/
│   └── derived/
│       ├── finger.csv
│       ├── wrist.csv
│       └── labels.csv
├── figures/                  # generated figures (PNG)
├── src/
│   ├── preprocess_march_sbp.py
│   ├── visualize_march_preprocess.py
│   └── train_march_sbp_torch.py
├── requirements.txt
├── .gitignore
└── (no Colab-specific README)
```

---

## 1) Run locally (recommended)

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Preprocess (outputs NPZ)

```bash
python src/preprocess_march_sbp.py --march-dir data/derived --out march_sbp_dataset.npz
```

### Visualize (PNG)

```bash
python src/visualize_march_preprocess.py --march-dir data/derived --out-dir figures --seed 42
```

### Train (GPU/CPU)

```bash
python src/train_march_sbp_torch.py --data march_sbp_dataset.npz --seed 42
```

To force CPU:

```bash
python src/train_march_sbp_torch.py --data march_sbp_dataset.npz --seed 42 --cpu
```

---

## 2) Data preparation

### A) Start from raw binaries (recommended, traceable)

1) Place raw data:

- `data/raw/New data.xlsx`
- `data/raw/recordsfinger/*.bin` (or `*.BIN`)
- `data/raw/recordswrist/*.bin` / `*.b`

2) Batch convert raw -> per-record PPG CSV:

```bash
python src/convert_march_raw_to_ppg_csv.py --march-raw data/raw --out data/ppg_csv
```

3) Build model-ready tables from Excel + PPG CSV (written to `data/derived/`):

```bash
python src/build_march_tables_from_xlsx.py --march-raw data/raw --ppg-csv data/ppg_csv --out data/derived
```

### B) Use derived tables directly

You need these three files under `data/derived/`:

- `finger.csv`
- `wrist.csv`
- `labels.csv`

They are aligned by `index`: the same `index` refers to the same session/sample.

