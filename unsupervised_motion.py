#!/usr/bin/env python3
"""Unsupervised motion / non-motion classification — no manual thresholds.

1. Extract per-second features from PVDF + PR.
2. Isolation Forest finds outliers (= motion) automatically.
3. Save segments + plot overview.
"""

from __future__ import annotations

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quality_gating import (
    GateConfig,
    load_dual_channel_csv,
    channel_motion_features,
    moving_average,
    abs_rate,
    robust_positive_z,
    mask_runs,
)


_FEATURE_NAMES = [
    "pvdf_volt_rate_p50", "pvdf_volt_rate_p90", "pvdf_volt_rate_max",
    "pvdf_env_rate_p50",  "pvdf_env_rate_p90",  "pvdf_env_rate_max",
    "pr_volt_rate_p50",   "pr_volt_rate_p90",   "pr_volt_rate_max",
    "pr_env_rate_p50",    "pr_env_rate_p90",    "pr_env_rate_max",
    "pvdf_std",           "pr_std",
]


def build_feature_table(
    pvdf: np.ndarray,
    pr: np.ndarray,
    cfg: GateConfig,
    win_sec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-sample motion features into per-window feature vectors."""
    fs = cfg.fs
    win_samples = int(round(win_sec * fs))
    n_windows = len(pvdf) // win_samples
    if n_windows < 10:
        raise ValueError("Signal too short for unsupervised learning.")

    # per-sample raw rates (no z-score — we'll scale per-file later)
    pvdf_smooth = moving_average(pvdf, int(round(0.3 * fs)))
    pvdf_trend = moving_average(pvdf, int(round(8.0 * fs)))
    pvdf_env = moving_average(np.abs(pvdf - pvdf_trend), int(round(1.0 * fs)))
    pvdf_volt_rate = abs_rate(pvdf_smooth, fs)
    pvdf_env_rate = abs_rate(pvdf_env, fs)

    pr_smooth = moving_average(pr, int(round(0.3 * fs)))
    pr_trend = moving_average(pr, int(round(8.0 * fs)))
    pr_env = moving_average(np.abs(pr - pr_trend), int(round(1.0 * fs)))
    pr_volt_rate = abs_rate(pr_smooth, fs)
    pr_env_rate = abs_rate(pr_env, fs)

    # z-score per file (robust reference)
    pvdf_vz = robust_positive_z(pvdf_volt_rate)
    pvdf_ez = robust_positive_z(pvdf_env_rate)
    pr_vz = robust_positive_z(pr_volt_rate)
    pr_ez = robust_positive_z(pr_env_rate)

    features = np.zeros((n_windows, len(_FEATURE_NAMES)), dtype=np.float64)
    t_mid = np.zeros(n_windows, dtype=np.float64)

    for i in range(n_windows):
        s = i * win_samples
        e = s + win_samples
        chunk = slice(s, e)

        features[i, 0] = np.percentile(pvdf_vz[chunk], 50)
        features[i, 1] = np.percentile(pvdf_vz[chunk], 90)
        features[i, 2] = np.max(pvdf_vz[chunk])
        features[i, 3] = np.percentile(pvdf_ez[chunk], 50)
        features[i, 4] = np.percentile(pvdf_ez[chunk], 90)
        features[i, 5] = np.max(pvdf_ez[chunk])
        features[i, 6] = np.percentile(pr_vz[chunk], 50)
        features[i, 7] = np.percentile(pr_vz[chunk], 90)
        features[i, 8] = np.max(pr_vz[chunk])
        features[i, 9] = np.percentile(pr_ez[chunk], 50)
        features[i, 10] = np.percentile(pr_ez[chunk], 90)
        features[i, 11] = np.max(pr_ez[chunk])
        features[i, 12] = np.std(pvdf[chunk])
        features[i, 13] = np.std(pr[chunk])

        t_mid[i] = (s + e) / 2 / fs

    return features, t_mid


def classify(features: np.ndarray, contamination: float) -> np.ndarray:
    """Isolation Forest: -1 = anomaly (motion), 1 = normal (non-motion)."""
    X = RobustScaler().fit_transform(features)
    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    labels = clf.fit_predict(X)          # 1=normal, -1=anomaly
    scores = clf.score_samples(X)        # higher = more normal
    return labels, scores, clf


def labels_to_segments(
    labels: np.ndarray,
    t_mid: np.ndarray,
    win_sec: float,
    motion_score: np.ndarray,
    pvdf: np.ndarray,
    pr: np.ndarray,
    cfg: GateConfig,
) -> pd.DataFrame:
    """Convert per-window labels to contiguous segments."""
    fs = cfg.fs
    win_samples = int(round(win_sec * fs))

    rows = []
    seg_id = 0
    idx = 0
    while idx < len(labels):
        val = labels[idx]
        end = idx + 1
        while end < len(labels) and labels[end] == val:
            end += 1
        s_sample = idx * win_samples
        e_sample = min(end * win_samples, len(pvdf))
        label = "motion" if val == -1 else "non_motion"
        rows.append({
            "segment_id": seg_id,
            "start_sec": round(float(idx * win_sec), 2),
            "end_sec": round(float(min(end * win_sec, len(pvdf) / fs)), 2),
            "duration_sec": round(float(e_sample - s_sample) / fs, 3),
            "label": label,
            "mean_anomaly_score": round(float(np.mean(motion_score[idx:end])), 4),
            "pvdf_std_v": round(float(np.nanstd(pvdf[s_sample:e_sample])), 4),
            "pr_std_v": round(float(np.nanstd(pr[s_sample:e_sample])), 4),
        })
        idx = end
        seg_id += 1
    return pd.DataFrame(rows)


def plot_result(
    out_path: Path,
    t: np.ndarray,
    pvdf: np.ndarray,
    pr: np.ndarray,
    labels: np.ndarray,
    t_mid: np.ndarray,
    anomaly_scores: np.ndarray,
    title: str = "",
):
    """2-row overview: signal + anomaly score scatter."""
    max_pts = 40000
    stride = max(1, int(np.ceil(len(t) / max_pts)))
    sl = slice(None, None, stride)

    # normalize signals for plotting
    pvdf_norm = (pvdf - np.median(pvdf)) / (np.percentile(pvdf, 95) - np.percentile(pvdf, 5) + 1e-12)
    pr_norm = (pr - np.median(pr)) / (np.percentile(pr, 95) - np.percentile(pr, 5) + 1e-12)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t[sl], pvdf_norm[sl], linewidth=0.6, label="PVDF", color="tab:blue")
    axes[0].plot(t[sl], pr_norm[sl], linewidth=0.5, label="PR", color="tab:orange", alpha=0.7)
    axes[0].set_ylabel("norm. signal")
    axes[0].legend(loc="upper right")
    axes[0].set_title(title or "Unsupervised motion detection")

    # anomaly score
    axes[1].plot(t_mid, anomaly_scores, linewidth=0.5, color="tab:purple")
    axes[1].set_ylabel("anomaly score")
    axes[1].axhline(np.percentile(anomaly_scores, 2), color="red", linestyle="--", linewidth=0.8, alpha=0.6)

    # motion mask
    motion = (labels == -1).astype(float)
    axes[2].fill_between(t_mid, 0, motion, step="mid", color="tab:red", alpha=0.4)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("motion")
    axes[2].set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Unsupervised motion / non-motion classifier.")
    p.add_argument("--input", type=Path, required=True, help="CSV with pvdf_adc, pr_adc columns")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/unsupervised"))
    p.add_argument("--win-sec", type=float, default=1.0, help="Feature window length (seconds)")
    p.add_argument("--contamination", type=float, default=0.08,
                   help="Expected fraction of motion (0.0-0.5)")
    args = p.parse_args()

    cfg = GateConfig()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.input}")
    data = load_dual_channel_csv(args.input, cfg)
    pvdf, pr, t_all = data["pvdf"], data["pr"], data["t"]

    print(f"Building {args.win_sec}s features ...")
    features, t_mid = build_feature_table(pvdf, pr, cfg, win_sec=args.win_sec)

    print(f"Running Isolation Forest (contamination={args.contamination}) ...")
    labels, scores, model = classify(features, contamination=args.contamination)

    n_anomaly = int(np.sum(labels == -1))
    n_normal = int(np.sum(labels == 1))
    print(f"  {n_anomaly} motion windows  ({100*n_anomaly/len(labels):.1f} %)")
    print(f"  {n_normal} normal windows ({100*n_normal/len(labels):.1f} %)")

    # top feature importances (mean |SHAP-like| proxy)
    from sklearn.inspection import permutation_importance
    r = permutation_importance(
        model, RobustScaler().fit_transform(features), labels,
        n_repeats=5, random_state=42, scoring="accuracy",
    )
    print("\nTop feature importances:")
    order = np.argsort(r.importances_mean)[::-1]
    for rank, idx in enumerate(order[:8], 1):
        print(f"  {rank}. {_FEATURE_NAMES[idx]:30s}  {r.importances_mean[idx]:.4f}")

    # segments
    segments = labels_to_segments(labels, t_mid, args.win_sec, scores, pvdf, pr, cfg)
    motion_segs = segments[segments["label"] == "motion"]

    # summary
    total_dur = len(pvdf) / cfg.fs
    motion_dur = float(motion_segs["duration_sec"].sum())
    summary = {
        "input_csv": str(args.input),
        "method": "IsolationForest",
        "contamination": args.contamination,
        "window_sec": args.win_sec,
        "total_duration_sec": round(total_dur, 2),
        "motion_duration_sec": round(motion_dur, 2),
        "motion_pct": round(100.0 * motion_dur / total_dur, 2) if total_dur else 0,
        "num_motion_segments": len(motion_segs),
        "num_non_motion_segments": len(segments) - len(motion_segs),
    }

    segments.to_csv(out_dir / "segments.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_result(
        out_dir / "overview.png",
        t_all, pvdf, pr, labels, t_mid, scores,
        title=f"{args.input.stem}  (motion={summary['motion_pct']:.1f}%)",
    )

    print(f"\nSaved: {out_dir}")
    print(f"  segments.csv  — {len(segments)} segments")
    print(f"  overview.png  — signal + anomaly score + mask")


if __name__ == "__main__":
    main()
