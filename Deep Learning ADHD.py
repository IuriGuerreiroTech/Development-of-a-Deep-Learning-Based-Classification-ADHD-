# train_fusion_lstm.py
import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# ====================== CONFIG ======================
@dataclass
class Config:
    # Manifests
    fusion_manifest: str = "C:/Users/iurig/Desktop/Dissertation/BALLADEER-Dataset/data_clean/dl_fusion_manifest.csv"
    eeg_only_manifest: Optional[str] = "C:/Users/iurig/Desktop/Dissertation/BALLADEER-Dataset/data_clean/dl_eeg_manifest.csv"
    eye_only_manifest: Optional[str] = "C:/Users/iurig/Desktop/Dissertation/BALLADEER-Dataset/data_clean/dl_eye_manifest.csv"

    # Run mode
    run_mode: str = "fusion"            # "fusion" | "eeg" | "eye"

    # filter rows by task_context (regex) r"Slackline" for CGX; r"(Cognifit|AttentionRobots)" for EPOC
    task_include_regex: Optional[str] = None

    # Column selection
    eeg_keep_regex = eeg_keep_regex = r"(?i)^POW\.(?:AF3|F7|F3|FC5|T7|P7|O1|O2|P8|T8|FC6|F4|F8|AF4)\.(?:Theta|Alpha|BetaL|BetaH|Gamma)$"
    eye_drop_regex: Optional[str] = r"(?i)\bweight\b"
    eye_keep_regex: Optional[str] = r"(?i)^(?:looked_col|looked_row)$"


    # Windowing
    window_len = 256
    window_stride = 128     # 50% overlap
    min_len_keep = 256     # discard sessions shorter than this

    # Training
    epochs: int = 25
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lstm_hidden: int = 128
    lstm_layers: int = 1
    dropout: float = 0.3
    bidirectional: bool = False

    # CV
    n_splits: int = 5
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)

# ====================== UTILS ======================
def seq_to_windows(arr: np.ndarray, L: int, stride: int) -> List[np.ndarray]:
    if arr is None or arr.shape[0] < L:
        return []
    out = []
    for start in range(0, arr.shape[0] - L + 1, stride):
        out.append(arr[start:start+L])
    return out

def pick_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    num_df = df.apply(pd.to_numeric, errors="coerce")
    num_df = num_df.dropna(axis=1, how="all")
    return num_df

def filter_columns_by_regex(df: pd.DataFrame, drop_regex: Optional[str]=None, keep_regex: Optional[str]=None) -> pd.DataFrame:
    cols = df.columns
    keep_mask = np.ones(len(cols), dtype=bool)
    if drop_regex:
        keep_mask &= ~cols.str.contains(drop_regex, regex=True)
    if keep_regex:
        keep_mask &= cols.str.contains(keep_regex, regex=True)
    return df.loc[:, cols[keep_mask]]

def compute_class_weight(labels: np.ndarray) -> float:
    pos = labels.sum()
    neg = len(labels) - pos
    return float(neg / max(pos, 1)) if pos > 0 else 1.0

def validate_data(data: np.ndarray, name: str = "data") -> bool:
    """Check for NaN, inf, and extreme values"""
    if np.any(np.isnan(data)):
        print(f"WARNING: {name} contains NaN values")
        return False
    if np.any(np.isinf(data)):
        print(f"WARNING: {name} contains inf values")
        return False
    if np.max(np.abs(data)) > 1e6:
        print(f"WARNING: {name} contains extreme values (max abs: {np.max(np.abs(data))})")
        return False
    return True

def safe_normalize(data: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply scaling with validation"""
    if len(data) == 0:
        return data
    normalized = scaler.transform(data)
    validate_data(normalized, "normalized data")
    return normalized

# ====================== DATASET ======================
class FusionSeqDataset(Dataset):
    """
    Reads a manifest with columns (depending on mode):
      - fusion: participant_id, label, eeg_path, eye_path
      - eeg:    participant_id, label, eeg_path
      - eye:    participant_id, label, eye_path

    Creates fixed-length windows with stride.
    Uses per-fold schemas to align feature columns (consistent dims).
    """
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        run_mode: str,
        L: int,
        stride: int,
        min_len_keep: int,
        eeg_keep_regex: Optional[str],
        eye_drop_regex: Optional[str],
        eye_keep_regex: Optional[str],
        eeg_schema: Optional[List[str]],
        eye_schema: Optional[List[str]],
        eeg_scaler: Optional[StandardScaler] = None,
        eye_scaler: Optional[StandardScaler] = None,
    ):
        self.run_mode = run_mode
        self.L = L
        self.stride = stride
        self.samples: List[Dict] = []
        self.eeg_schema = eeg_schema
        self.eye_schema = eye_schema
        self.eeg_scaler = eeg_scaler
        self.eye_scaler = eye_scaler

        for _, row in manifest_df.iterrows():
            y = int(row["label"])

            eeg_windows = None
            if run_mode in ("fusion", "eeg"):
                eeg_df = pd.read_csv(row["eeg_path"])
                if eeg_keep_regex:
                    eeg_df = filter_columns_by_regex(eeg_df, drop_regex=None, keep_regex=eeg_keep_regex)
                eeg_df = pick_numeric_df(eeg_df)
                # Align to schema
                if self.eeg_schema is not None:
                    eeg_df = eeg_df.reindex(columns=self.eeg_schema, fill_value=0.0)
                eeg_np = eeg_df.to_numpy(dtype=np.float32)
                # Apply scaling if scaler is provided
                if self.eeg_scaler is not None:
                    eeg_np = safe_normalize(eeg_np, self.eeg_scaler)
                eeg_windows = seq_to_windows(eeg_np, L, stride) if eeg_np.shape[0] >= min_len_keep else []

            eye_windows = None
            if run_mode in ("fusion", "eye"):
                eye_df = pd.read_csv(row["eye_path"])
                if eye_drop_regex or eye_keep_regex:
                    eye_df = filter_columns_by_regex(eye_df, drop_regex=eye_drop_regex, keep_regex=eye_keep_regex)

                # engineer eye features from looked_col/row 
                if {"looked_col", "looked_row"}.issubset(eye_df.columns):
                    col = pd.to_numeric(eye_df["looked_col"], errors="coerce")
                    row_ = pd.to_numeric(eye_df["looked_row"], errors="coerce")
                    dx = col.diff().fillna(0.0)
                    dy = row_.diff().fillna(0.0)
                    speed = (dx.pow(2) + dy.pow(2)).pow(0.5)
                    accel = speed.diff().fillna(0.0)
                    eye_df["dx"] = dx
                    eye_df["dy"] = dy
                    eye_df["speed"] = speed
                    eye_df["accel"] = accel
                # -------------------------------------------

                eye_df = pick_numeric_df(eye_df)
                if self.eye_schema is not None:
                    eye_df = eye_df.reindex(columns=self.eye_schema, fill_value=0.0)
                eye_np = eye_df.to_numpy(dtype=np.float32)
                if self.eye_scaler is not None:
                    eye_np = safe_normalize(eye_np, self.eye_scaler)
                eye_windows = seq_to_windows(eye_np, L, stride) if eye_np.shape[0] >= min_len_keep else []

            # Pack windows
            if self.run_mode == "fusion":
                n = min(len(eeg_windows), len(eye_windows))
                for i in range(n):
                    self.samples.append(dict(
                        eeg=torch.from_numpy(eeg_windows[i]),
                        eye=torch.from_numpy(eye_windows[i]),
                        y=torch.tensor(y, dtype=torch.float32),
                        pid=row["participant_id"],
                    ))
            elif self.run_mode == "eeg":
                for w in eeg_windows:
                    self.samples.append(dict(
                        eeg=torch.from_numpy(w),
                        y=torch.tensor(y, dtype=torch.float32),
                        pid=row["participant_id"],
                    ))
            else:  # eye only
                for w in eye_windows:
                    self.samples.append(dict(
                        eye=torch.from_numpy(w),
                        y=torch.tensor(y, dtype=torch.float32),
                        pid=row["participant_id"],
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fusion(batch):
    ys = torch.stack([b["y"] for b in batch], dim=0)
    pids = [b.get("pid", "") for b in batch]

    eeg_seqs = [b["eeg"] for b in batch] if "eeg" in batch[0] else None
    eye_seqs = [b["eye"] for b in batch] if "eye" in batch[0] else None

    eeg_pad = pad_sequence(eeg_seqs, batch_first=True) if eeg_seqs else None
    eye_pad = pad_sequence(eye_seqs, batch_first=True) if eye_seqs else None

    eeg_mask = (torch.sum(torch.abs(eeg_pad), dim=-1) != 0) if eeg_pad is not None else None
    eye_mask = (torch.sum(torch.abs(eye_pad), dim=-1) != 0) if eye_pad is not None else None

    return dict(y=ys, eeg=eeg_pad, eye=eye_pad, eeg_mask=eeg_mask, eye_mask=eye_mask, pid=pids)

# ====================== MODEL ======================
class SeqEncoder(nn.Module):
    def __init__(self, input_dim, hidden, layers=1, dropout=0.0, bidir=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=(dropout if layers > 1 else 0.0),
            bidirectional=bidir
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden * (2 if bidir else 1)

    def forward(self, x, mask=None):
        out, (h, c) = self.lstm(x)
        h_last = h[-1]           # (B, H) last layer
        return self.dropout(h_last)

class FusionHead(nn.Module):
    def __init__(self, in_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, z):
        return self.net(z).squeeze(1)

class TwoStreamLSTM(nn.Module):
    def __init__(self, eeg_dim: Optional[int], eye_dim: Optional[int], cfg: Config):
        super().__init__()
        self.has_eeg = eeg_dim is not None
        self.has_eye = eye_dim is not None

        if self.has_eeg:
            self.eeg_enc = SeqEncoder(eeg_dim, cfg.lstm_hidden, cfg.lstm_layers, cfg.dropout, cfg.bidirectional)
        if self.has_eye:
            self.eye_enc = SeqEncoder(eye_dim, cfg.lstm_hidden, cfg.lstm_layers, cfg.dropout, cfg.bidirectional)

        fused_in = (self.eeg_enc.out_dim if self.has_eeg else 0) + (self.eye_enc.out_dim if self.has_eye else 0)
        self.head = FusionHead(fused_in, cfg.dropout)

    def forward(self, eeg=None, eye=None, eeg_mask=None, eye_mask=None):
        zs = []
        if self.has_eeg and eeg is not None:
            zs.append(self.eeg_enc(eeg, eeg_mask))
        if self.has_eye and eye is not None:
            zs.append(self.eye_enc(eye, eye_mask))
        z = torch.cat(zs, dim=1) if len(zs) > 1 else zs[0]
        return self.head(z)

# ====================== METRICS ======================
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.45) -> Dict[str, float]:
    y_pred = (y_prob >= thresh).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    sens = tp / (tp + fn) if (tp + fn) else 0.0   # TPR
    spec = tn / (tn + fp) if (tn + fp) else 0.0   # TNR
    ba   = 0.5 * (sens + spec)
    acc  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = float("nan")

    return dict(roc=roc, sens=sens, spec=spec, ba=ba, acc=acc, pr_auc=pr_auc)

def compute_metrics_by_participant(true_list, prob_list, pid_list, thresh=0.45):
    """
    Aggregates window-level predictions to participant-level predictions by averaging probabilities.
    """
    df = pd.DataFrame({'true': true_list, 'prob': prob_list, 'pid': pid_list})
    participant_df = df.groupby('pid').agg({'true': 'first', 'prob': 'mean'}).reset_index() # Avg probability per participant
    y_true = participant_df['true'].values
    y_prob = participant_df['prob'].values
    return compute_metrics(y_true, y_prob, thresh)

# ====================== SCHEMAS & TRAIN ======================
# get_schema (majority vote ≥60%, union fallback), and include engineered eye cols
def get_schema(df: pd.DataFrame, path_col: str,
               drop_regex: Optional[str]=None,
               keep_regex: Optional[str]=None,
               sample_cap: int = 100) -> List[str]:
    """
    Returns a schema of numeric columns (after regex filters) from up to
    `sample_cap` files. Uses majority vote (>=60% of files) with union fallback.
    For eye files: if looked_col/row exist, also include engineered ['dx','dy','speed','accel'].
    """
    from collections import Counter

    col_sets = []
    n = 0
    for p in df[path_col].tolist()[:sample_cap]:
        try:
            d = pd.read_csv(p)
            if drop_regex or keep_regex:
                d = filter_columns_by_regex(d, drop_regex=drop_regex, keep_regex=keep_regex)
            cols = list(d.columns)
            if "looked_col" in cols and "looked_row" in cols:
                cols = cols + ["dx", "dy", "speed", "accel"]
            # -------------------------------------------------------------------------
            d = pick_numeric_df(d)
            cols_numeric = [c for c in cols if c in d.columns or c in ["dx","dy","speed","accel"]]
            if cols_numeric:
                col_sets.append(set(cols_numeric))
                n += 1
        except Exception:
            continue

    if n == 0:
        return []

    counter = Counter()
    for s in col_sets:
        counter.update(s)

    thresh = max(1, int(0.6 * n))  # 60% majority
    majority = [c for c, k in counter.items() if k >= thresh]
    if majority:
        return sorted(majority)

    # Fallback: union to avoid empty schema
    return sorted(set().union(*col_sets))

def fit_scaler(df: pd.DataFrame, path_col: str, schema: List[str], sample_cap: int = 100) -> StandardScaler:
    """
    Fits a StandardScaler on a sample of data from the manifest.
    """
    scaler = StandardScaler()
    all_data = []
    for p in df[path_col].tolist()[:sample_cap]:
        try:
            d = pd.read_csv(p)
            d = d.reindex(columns=schema, fill_value=0.0)
            d_np = d.to_numpy(dtype=np.float32)
            
            if not validate_data(d_np, f"raw data from {p}"):
                d_np = np.nan_to_num(d_np, nan=0.0, posinf=0.0, neginf=0.0)
            all_data.append(d_np)
        except Exception as e:
            print(f"Error reading {p}: {e}")
            continue
    
    if all_data:
        all_data = np.vstack(all_data)
        if not validate_data(all_data, "concatenated training data"):
            all_data = np.nan_to_num(all_data, nan=0.0, posinf=0.0, neginf=0.0)
        scaler.fit(all_data)
    
    return scaler


def run_fold(train_df, val_df, cfg: Config, eeg_schema=None, eye_schema=None):
    # Fit scalers on the training data only
    eeg_scaler, eye_scaler = None, None
    if cfg.run_mode in ("fusion","eeg") and eeg_schema:
        eeg_scaler = fit_scaler(train_df, "eeg_path", eeg_schema)
    if cfg.run_mode in ("fusion","eye") and eye_schema:
        eye_scaler = fit_scaler(train_df, "eye_path", eye_schema)

    # Build datasets/loaders
    ds_tr = FusionSeqDataset(
        train_df, cfg.run_mode, cfg.window_len, cfg.window_stride, cfg.min_len_keep,
        cfg.eeg_keep_regex, cfg.eye_drop_regex, cfg.eye_keep_regex,
        eeg_schema=eeg_schema, eye_schema=eye_schema,
        eeg_scaler=eeg_scaler, eye_scaler=eye_scaler
    )
    ds_va = FusionSeqDataset(
        val_df, cfg.run_mode, cfg.window_len, cfg.window_stride, cfg.min_len_keep,
        cfg.eeg_keep_regex, cfg.eye_drop_regex, cfg.eye_keep_regex,
        eeg_schema=eeg_schema, eye_schema=eye_schema,
        eeg_scaler=eeg_scaler, eye_scaler=eye_scaler
    )
    if len(ds_tr) == 0 or len(ds_va) == 0:
        raise RuntimeError("Empty dataset after windowing. Adjust window_len/stride/min_len_keep or filters.")

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,  collate_fn=collate_fusion)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fusion)

    # Infer input dims from schema
    eeg_dim = len(eeg_schema) if eeg_schema is not None else None
    eye_dim = len(eye_schema) if eye_schema is not None else None

    model = TwoStreamLSTM(eeg_dim, eye_dim, cfg).to(cfg.device)
    pos_w = compute_class_weight(train_df["label"].values)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=cfg.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best = dict(ba=-1, state=None)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        for batch in dl_tr:
            y = batch["y"].to(cfg.device)
            eeg = batch.get("eeg")
            eye = batch.get("eye")
            eeg = eeg.to(cfg.device) if eeg is not None else None
            eye = eye.to(cfg.device) if eye is not None else None

            logits = model(eeg=eeg, eye=eye, eeg_mask=batch.get("eeg_mask"), eye_mask=batch.get("eye_mask"))
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += loss.item() * y.size(0)
        tr_loss /= len(ds_tr)

        # Eval
        model.eval()
        ys, ps, pids = [], [], []
        with torch.no_grad():
            for batch in dl_va:
                y = batch["y"].to(cfg.device)
                eeg = batch.get("eeg")
                eye = batch.get("eye")
                eeg = eeg.to(cfg.device) if eeg is not None else None
                eye = eye.to(cfg.device) if eye is not None else None
                logits = model(eeg=eeg, eye=eye, eeg_mask=batch.get("eeg_mask"), eye_mask=batch.get("eye_mask"))
                prob = torch.sigmoid(logits)
                ys.append(y.cpu().numpy())
                ps.append(prob.cpu().numpy())
                pids.extend(batch['pid'])
        ys = np.concatenate(ys); ps = np.concatenate(ps)
        metrics = compute_metrics(ys, ps, thresh=0.45)
        participant_metrics = compute_metrics_by_participant(ys, ps, pids, thresh=0.45)

        scheduler.step(participant_metrics["ba"])

        # Use participant-level BA for model selection
        if participant_metrics["ba"] > best["ba"]:
            best = dict(ba=participant_metrics["ba"], state=model.state_dict())

        print(f"[Epoch {epoch:02d}] loss={tr_loss:.4f} | "
              f"ROC={metrics['roc']:.3f} BA={metrics['ba']:.3f} ACC={metrics['acc']:.3f} "
              f"Sens={metrics['sens']:.3f} Spec={metrics['spec']:.3f} PR-AUC={metrics['pr_auc']:.3f}")
        print(f"         [PARTICIPANT-LEVEL] BA={participant_metrics['ba']:.3f} ACC={participant_metrics['acc']:.3f}")

    # Restore best by participant-level BA
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    # Final val metrics on participant level
    model.eval()
    ys, ps, pids = [], [], []
    with torch.no_grad():
        for batch in dl_va:
            y = batch["y"].to(cfg.device)
            eeg = batch.get("eeg")
            eye = batch.get("eye")
            eeg = eeg.to(cfg.device) if eeg is not None else None
            eye = eye.to(cfg.device) if eye is not None else None
            logits = model(eeg=eeg, eye=eye, eeg_mask=batch.get("eeg_mask"), eye_mask=batch.get("eye_mask"))
            prob = torch.sigmoid(logits)
            ys.append(y.cpu().numpy())
            ps.append(prob.cpu().numpy())
            pids.extend(batch['pid'])
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    final_metrics = compute_metrics_by_participant(ys, ps, pids)
    return final_metrics

# compute schema per fold using train_df
def stratified_group_kfold_train(manifest_path: str, run_mode: str, cfg: Config):
    df = pd.read_csv(manifest_path)

    # task filter
    if cfg.task_include_regex and "task_context" in df.columns:
        df = df[df["task_context"].astype(str).str.contains(cfg.task_include_regex, regex=True, na=False)]
        if len(df) == 0:
            raise ValueError(f"No rows matched task filter: {cfg.task_include_regex}")

    # Required columns
    need = ["participant_id", "label"]
    if run_mode in ("fusion","eeg"):
        need += ["eeg_path"]
    if run_mode in ("fusion","eye"):
        need += ["eye_path"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Manifest missing column: {c}")

    # Use StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    groups = df["participant_id"].values
    y = df["label"].values

    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(df, y, groups)):
        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        va_df = df.iloc[va_idx].reset_index(drop=True)
        
        # per-fold schema derived from *training* split only
        eeg_schema_fold = None
        eye_schema_fold = None
        if cfg.run_mode in ("fusion", "eeg"):
            eeg_schema_fold = get_schema(tr_df, "eeg_path", drop_regex=None, keep_regex=cfg.eeg_keep_regex)
            if not eeg_schema_fold:
                raise RuntimeError("EEG schema is empty in this fold. Adjust eeg_keep_regex or data.")
        if cfg.run_mode in ("fusion", "eye"):
            eye_schema_fold = get_schema(tr_df, "eye_path", drop_regex=cfg.eye_drop_regex, keep_regex=cfg.eye_keep_regex)
            if not eye_schema_fold:
                raise RuntimeError("Eye schema is empty in this fold. Adjust eye_keep_regex/eye_drop_regex or data.")

        print(f"\n===== Fold {fold+1}/{cfg.n_splits} (train n={len(tr_df)}, val n={len(va_df)}) =====")
        print(f"Train class balance: {tr_df['label'].value_counts().to_dict()}")
        print(f"Val class balance: {va_df['label'].value_counts().to_dict()}")
        
        try:
            m = run_fold(tr_df, va_df, cfg, eeg_schema_fold, eye_schema_fold)
            print(f"Fold {fold+1} -> ROC={m['roc']:.3f} BA={m['ba']:.3f} ACC={m['acc']:.3f} "
                  f"Sens={m['sens']:.3f} Spec={m['spec']:.3f} PR-AUC={m['pr_auc']:.3f}")
            fold_metrics.append(m)
        except Exception as e:
            print(f"Error in fold {fold+1}: {e}")
            continue

    if not fold_metrics:
        raise RuntimeError("All folds failed!")
    
    agg = {k: float(np.nanmean([m[k] for m in fold_metrics])) for k in fold_metrics[0].keys()}
    print("\n===== CV MEAN (PARTICIPANT-LEVEL) =====")
    print(f"ROC={agg['roc']:.3f} BA={agg['ba']:.3f} ACC={agg['acc']:.3f} "
          f"Sens={agg['sens']:.3f} Spec={agg['spec']:.3f} PR-AUC={agg['pr_auc']:.3f}")
    return agg

# ====================== MAIN ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fusion","eeg","eye"], default=cfg.run_mode)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--window", type=int, default=cfg.window_len)
    parser.add_argument("--stride", type=int, default=cfg.window_stride)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--task_filter", type=str, default=None,
                        help="Regex on task_context to subset (e.g., 'Slackline' or '(Cognifit|AttentionRobots)')")
    args = parser.parse_args()

    cfg.run_mode = args.mode

    # Guard against typos like "egg"
    if cfg.run_mode not in {"fusion","eeg","eye"}:
        raise ValueError(f"Invalid run_mode='{cfg.run_mode}'. Use one of ['eeg','eye','fusion'].")

    cfg.window_len = args.window
    cfg.window_stride = args.stride
    cfg.epochs = args.epochs
    if args.task_filter:
        cfg.task_include_regex = args.task_filter

    if cfg.run_mode == "fusion":
        manifest = args.manifest or cfg.fusion_manifest
    elif cfg.run_mode == "eeg":
        manifest = args.manifest or cfg.eeg_only_manifest
    else:
        manifest = args.manifest or cfg.eye_only_manifest

    print(f"Device: {cfg.device} | Mode: {cfg.run_mode}")
    stratified_group_kfold_train(manifest, cfg.run_mode, cfg)

    
