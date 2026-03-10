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

---

## 3) Methods (high‑level, non‑CS explanation)

This section explains **what the pipeline does conceptually**, without going into low‑level code.

### 3.1 Data and preprocessing

- **Raw signals**:
  - Each recording session has:
    - Finger PPG: infrared (IR) and red light channels.
    - Wrist PPG: infrared (IR) and red light channels.
    - Heart rate (HR): one value per session from the blood pressure device.
    - Blood pressure: systolic blood pressure (SBP) and diastolic (DBP).
    - Posture label: `sit`, `lay`, or `plank`.
- **Alignment between files**:
  - The Excel file (`New data.xlsx`) describes each session (who, when, posture, BP, HR).
  - The finger and wrist `.csv` files contain the PPG waveforms for each session.
  - We build an `index` key so that **the same row** in:
    - `finger.csv`
    - `wrist.csv`
    - `labels.csv`
    always refers to **the same physical recording**.
- **Handling missing or short signals**:
  - Some raw recordings are shorter or have missing points.
  - Instead of discarding them, we:
    - Pad short sequences with missing values.
    - Fill missing values using simple interpolation.
  - This keeps as many sessions as possible while avoiding obvious artifacts.
- **Signal cleaning**:
  - We apply a **band‑pass filter** to keep only the heart‑related frequencies (roughly 0.5–8 Hz).
  - We **downsample** the data to a lower rate (10 Hz) so that:
    - We keep the waveform shape.
    - The model becomes smaller and easier to train.
  - For each session and each PPG channel we perform **per‑sample normalization** (z‑score):
    - We subtract the mean and divide by the standard deviation of that session.
    - This lets the model focus on the **shape and relative changes** of the waveform, not absolute sensor values.
- **HR normalization**:
  - HR is normalized across the whole dataset (global z‑score) so that it can be used as a comparable numeric feature.
- **Final preprocessed dataset (`march_sbp_dataset.npz`)**:
  - `X`: shape `(N, 600, 4)`
    - 600 time points per session, 4 channels:
      1. Finger IR
      2. Finger red
      3. Wrist IR
      4. Wrist red
  - `hr`: shape `(N, 1)` normalized heart rate.
  - `y`: shape `(N,)` systolic blood pressure (SBP).
  - `group`: subject identifier (string).
  - `state`: posture (`sit`, `lay`, `plank`).
  - `index`: integer index that links back to the original tables.

### 3.2 SBP prediction model (regression)

- **Goal**: predict **continuous SBP** values from PPG and HR.
- **Input to the model**:
  - The preprocessed PPG tensor `X` (600 time steps × 4 channels).
  - The normalized heart rate `hr`.
- **Model idea** (very high level):
  - The model has **two branches**:
    - One branch processes **finger PPG** (2 channels).
    - One branch processes **wrist PPG** (2 channels).
  - Each branch:
    - First uses **1‑D convolutional layers** to detect local waveform patterns.
    - Then uses a small **Transformer‑style module** to capture longer‑range temporal relationships (how patterns evolve over time).
    - Finally uses an **attention pooling layer** that automatically focuses on the most informative time points instead of averaging everything.
  - The HR feature is passed through a small fully‑connected network.
  - At the end, the outputs of the two PPG branches and the HR branch are **concatenated** and passed to a final fully‑connected “head” that predicts one number: the SBP.
- **Training strategy**:
  - The script `train_march_sbp_torch.py` trains this model with:
    - A **Huber loss**, which is more robust to occasional outliers than simple squared error.
    - An **AdamW optimizer** with a cosine learning‑rate schedule.
    - Gradient clipping, to keep the training stable.
  - Because the dataset is small, we:
    - Use a relatively compact model size.
    - Use early stopping based on the validation performance to avoid overfitting.
  - The final reported metrics are:
    - **MAE** (mean absolute error)
    - **RMSE** (root mean squared error)
    - **R²** (coefficient of determination)

### 3.3 Posture / planking classification model

In addition to predicting SBP, we use essentially the **same PPG + HR backbone** to classify posture, especially to detect **planking**.

- **Shared backbone**:
  - The state classification model in `train_march_state_torch.py` reuses the same idea:
    - Dual PPG branches (finger + wrist).
    - Convolution + Transformer‑style temporal modeling.
    - Attention pooling over time.
    - HR as an extra feature.
- **Two label modes**:
  - **Three‑class mode**: `sit`, `lay`, `plank`.
  - **Binary mode**: `rest(sit+lay)` vs `plank`.
    - This is useful when we care mainly about “is the subject planking or not”.
- **Strict train / validation / test split with balanced classes**:
  - We explicitly split the dataset into:
    - Training set (used to fit the model).
    - Validation set (used to choose the best model).
    - Test set (held out for the final evaluation).
  - All splits are done with **stratification by label**:
    - The fraction of `plank` vs `rest` samples is kept as close as possible in all three sets.
- **Imbalanced data handling (plank is rare)**:
  - In this dataset, planking sessions are fewer than resting sessions.
  - If we simply train on the raw counts, the model tends to ignore `plank`.
  - To fix this, we combine two ideas:
    - **Class‑weighted loss**:
      - The training loss (`CrossEntropyLoss`) gives slightly higher weight to **plank** errors than to rest errors.
      - The weights are derived from the inverse of the class frequencies but then softened, so we do not over‑penalize.
    - **Class‑balanced sampling**:
      - For each training epoch we oversample minority classes (especially `plank`) so that each class contributes roughly the same number of training examples.
      - This forces the model to see planking patterns many times, even if there are fewer raw recordings.
- **Outputs and evaluation**:
  - The model outputs **class probabilities** (via softmax); we choose the class with the highest probability.
  - We report:
    - **Validation accuracy and macro‑average F1** (to monitor training).
    - **Test accuracy and macro‑average F1** (final result).
    - A **confusion matrix** on the test set, which shows:
      - How many resting sessions were correctly/incorrectly classified.
      - How many planking sessions were correctly/incorrectly classified.
  - The script can also save a **PNG image of the confusion matrix**, which is easy to show in slides to non‑technical collaborators.

### 3.4 Model architecture and training details (CS‑oriented)

This subsection gives a more formal, implementation‑level description of the core models.

- **Input representation**
  - `X ∈ ℝ^{N×T×4}` with `T = 600`:
    - Channel order: `[finger_IR, finger_red, wrist_IR, wrist_red]`.
  - `hr ∈ ℝ^{N×1}`: scalar HR per sample (global z‑score).
  - For all models, inputs are stored and fed as `float32`.

- **PPG branch (`PpgBranch`)**
  - Each branch sees 2 channels (either finger or wrist): `x ∈ ℝ^{N×T×2}`.
  - Convolutional stem (implemented as `Conv1d` on `(N, C, T)`):
    - `Conv1d(2 → d_model, kernel_size=9, padding=4)` + GELU + `BatchNorm1d(d_model)`
    - `Conv1d(d_model → d_model, kernel_size=5, padding=2)` + GELU + `BatchNorm1d(d_model)`
  - Positional encoding:
    - Learnable tensor `pos ∈ ℝ^{1×T_max×d_model}`, with `T_max = 600`.
    - Added element‑wise to the time dimension after the stem.
  - Temporal encoder: stack of `n_layers` Transformer blocks:
    - Each block uses pre‑norm **Multi‑Head Self‑Attention** and a 2‑layer feed‑forward network:
      - LayerNorm(d_model)
      - `MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)`
      - Residual connection + Dropout
      - LayerNorm(d_model)
      - `Linear(d_model → d_model * ff_mult)` + GELU + Dropout + `Linear(d_model * ff_mult → d_model)`
      - Second residual connection + Dropout
  - Temporal aggregation (attention pooling):
    - A single linear layer `score: ℝ^{d_model} → ℝ` applied at each time step.
    - Attention weights via softmax over time, used to form a weighted sum:
      - `w_t = softmax(score(h_t))`
      - `z = Σ_t w_t h_t ∈ ℝ^{d_model}`.

- **SBP regression head (`train_march_sbp_torch.py`)**
  - Two PPG branches:
    - `z_finger, z_wrist ∈ ℝ^{d_model}` from the finger and wrist branches.
  - HR sub‑network:
    - `hr ∈ ℝ^{N×1}` → `Linear(1 → 32)` → GELU → `Linear(32 → 32)` → `h_hr ∈ ℝ^{N×32}`.
  - Concatenation:
    - `z = concat(z_finger, z_wrist, h_hr) ∈ ℝ^{N×(2*d_model+32)}`.
  - Regression head:
    - LayerNorm(2*d_model + 32)
    - `Linear(2*d_model + 32 → 128)` → GELU → Dropout
    - `Linear(128 → 1)` → output SBP (scalar per sample).
  - Default hyper‑parameters:
    - `d_model = 96`
    - `n_layers = 3`
    - `n_heads = 4`
    - `ff_mult = 4`
    - `dropout ≈ 0.20`

- **SBP training configuration**
  - Loss: `HuberLoss(delta = 1.0)` between predicted and standardized SBP.
  - Optimizer: `AdamW(lr = 2e-4, weight_decay = 1e-3)`.
  - Learning‑rate schedule: `CosineAnnealingLR(T_max = epochs)`.
  - Batch size: 8, with gradient clipping at `max_norm = 1.0`.
  - Early stopping: patience ≈ 35 epochs based on validation error.
  - Metrics: MAE, RMSE, and R² computed after de‑standardizing SBP.

- **State / planking classifier (`train_march_state_torch.py`)**
  - Backbone:
    - Reuses exactly the same dual‑branch PPG encoder and HR sub‑network as SBP regression.
  - Output head:
    - Same concatenation `z ∈ ℝ^{N×(2*d_model+32)}`.
    - LayerNorm(2*d_model + 32)
    - `Linear(2*d_model + 32 → 128)` → GELU → Dropout
    - `Linear(128 → K)` where `K = 2` (binary `rest` vs `plank`) or `K = 3` (sit / lay / plank).
    - Softmax is applied implicitly by `CrossEntropyLoss`.
  - Train/val/test split:
    - Single‑shot split on sample indices:
      - First, stratified `train+val` vs `test` with `test_ratio` (default 0.2).
      - Then, stratified split of `train+val` into `train` and `val` using a relative `val_ratio`.
    - Stratification uses the class labels so that all three sets preserve class proportions.
  - Class imbalance handling:
    - Class weights:
      - Compute inverse‑frequency weights `inv_freq` from label counts and soften with exponent `γ = 0.5`.
      - Normalize so that the average weight is approximately 1.
      - Pass these weights to `CrossEntropyLoss(weight=...)`.
    - Balanced sampling on the training set:
      - For each epoch, oversample each class (especially the minority `plank`) to the size of the largest class.
      - Construct a balanced index list for mini‑batch training.
  - Optimization:
    - Same optimizer and cosine schedule as SBP.
    - Loss: `CrossEntropyLoss` with class weights.
    - Metrics:
      - Validation: accuracy and macro‑F1.
      - Test: accuracy, macro‑F1, and confusion matrices for both validation and test sets.

