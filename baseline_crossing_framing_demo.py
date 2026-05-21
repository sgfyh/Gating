from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from quality_gating import (
    DEFAULT_CSV,
    OUTPUT_DIR,
    USER_CONFIG,
    GateConfig,
    build_segments,
    compute_motion_gate,
    load_dual_channel_csv,
    moving_average,
    robust_scale,
)


DEMO_OUTPUT_DIR = OUTPUT_DIR / "ml_peak_detector"
DEMO_SPAN_SEC = 100.0
BASELINE_SEC = 10.0
SMOOTH_SEC = 0.25
PEAK_LABEL_RADIUS_SEC = 0.12
MIN_PEAK_DISTANCE_SEC = 2.2
MIN_CLEAN_SEGMENT_SEC = 60.0
NEGATIVE_TO_POSITIVE_RATIO = 10
RANDOM_STATE = 42
PEAK_HEIGHT_PERCENTILE = 65.0


def respiratory_component(pvdf: np.ndarray, cfg: GateConfig) -> tuple[np.ndarray, np.ndarray]:
    baseline = moving_average(pvdf, int(round(BASELINE_SEC * cfg.fs)))
    resp = pvdf - baseline
    resp = moving_average(resp, int(round(SMOOTH_SEC * cfg.fs)))
    return baseline, resp


def rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    win = max(int(win), 1)
    return pd.Series(x).rolling(win, center=True, min_periods=1).std().fillna(0.0).to_numpy()


def extract_point_features(resp: np.ndarray, cfg: GateConfig) -> tuple[np.ndarray, list[str]]:
    r = np.asarray(resp, dtype=float)
    d1 = np.gradient(r) * cfg.fs
    d2 = np.gradient(d1) * cfg.fs

    w_short = int(round(0.5 * cfg.fs))
    w_mid = int(round(1.5 * cfg.fs))
    w_long = int(round(3.0 * cfg.fs))

    mean_short = moving_average(r, w_short)
    mean_mid = moving_average(r, w_mid)
    std_short = rolling_std(r, w_short)
    std_mid = rolling_std(r, w_mid)
    smooth_long = moving_average(r, w_long)

    local_max = pd.Series(r).rolling(w_mid, center=True, min_periods=1).max().to_numpy()
    local_min = pd.Series(r).rolling(w_mid, center=True, min_periods=1).min().to_numpy()
    local_range = local_max - local_min
    peak_prom_proxy = r - local_min
    is_local_top_proxy = local_max - r

    features = np.column_stack(
        [
            r,
            np.abs(r),
            d1,
            d2,
            mean_short,
            mean_mid,
            std_short,
            std_mid,
            smooth_long,
            local_range,
            peak_prom_proxy,
            is_local_top_proxy,
        ]
    )
    names = [
        "resp",
        "abs_resp",
        "d1",
        "d2",
        "mean_0p5s",
        "mean_1p5s",
        "std_0p5s",
        "std_1p5s",
        "smooth_3s",
        "local_range_1p5s",
        "peak_prom_proxy",
        "distance_to_local_max",
    ]
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), names


def pseudo_peak_labels(resp: np.ndarray, clean_mask: np.ndarray, cfg: GateConfig) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros(resp.size, dtype=np.uint8)
    _, scale = robust_scale(resp[clean_mask])
    distance = int(round(MIN_PEAK_DISTANCE_SEC * cfg.fs))
    positive_resp = resp[clean_mask & (resp > 0)]
    percentile_height = float(np.nanpercentile(positive_resp, PEAK_HEIGHT_PERCENTILE)) if positive_resp.size else 0.0
    prominence = max(0.35 * scale, 1e-6)
    height = max(percentile_height, 0.35 * scale, 1e-6)

    peak_indices: list[int] = []
    for start, end in mask_runs(clean_mask):
        if (end - start) / cfg.fs < MIN_CLEAN_SEGMENT_SEC:
            continue
        local_peaks, _ = signal.find_peaks(
            resp[start:end],
            distance=distance,
            prominence=prominence,
            height=height,
        )
        peak_indices.extend((local_peaks + start).tolist())

    peak_indices_arr = np.asarray(sorted(set(peak_indices)), dtype=int)
    radius = int(round(PEAK_LABEL_RADIUS_SEC * cfg.fs))
    for idx in peak_indices_arr:
        left = max(0, idx - radius)
        right = min(labels.size, idx + radius + 1)
        labels[left:right] = 1
    labels[~clean_mask] = 0
    return labels, peak_indices_arr


def mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    mask = np.asarray(mask, dtype=bool)
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < len(mask) and mask[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def sample_training_indices(labels: np.ndarray, clean_mask: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_STATE)
    positives = np.flatnonzero((labels == 1) & clean_mask)
    negatives = np.flatnonzero((labels == 0) & clean_mask)
    if positives.size == 0:
        raise ValueError("No pseudo peaks found. Try lowering the find_peaks prominence.")
    n_neg = min(negatives.size, positives.size * NEGATIVE_TO_POSITIVE_RATIO)
    sampled_neg = rng.choice(negatives, size=n_neg, replace=False)
    indices = np.r_[positives, sampled_neg]
    rng.shuffle(indices)
    return indices


def train_peak_classifier(features: np.ndarray, labels: np.ndarray, clean_mask: np.ndarray) -> tuple[RandomForestClassifier, dict[str, object]]:
    indices = sample_training_indices(labels, clean_mask)
    x = features[indices]
    y = labels[indices]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary",
        zero_division=0,
    )
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    clf.fit(x, y)
    summary = {
        "training_samples": int(indices.size),
        "positive_samples": int(np.sum(y == 1)),
        "negative_samples": int(np.sum(y == 0)),
        "holdout_precision": float(precision),
        "holdout_recall": float(recall),
        "holdout_f1": float(f1),
        "classification_report": report,
    }
    return clf, summary


def predict_peaks(
    clf: RandomForestClassifier,
    features: np.ndarray,
    resp: np.ndarray,
    clean_mask: np.ndarray,
    cfg: GateConfig,
) -> tuple[np.ndarray, np.ndarray]:
    prob = clf.predict_proba(features)[:, 1]
    prob[~clean_mask] = 0.0

    distance = int(round(MIN_PEAK_DISTANCE_SEC * cfg.fs))
    _, scale = robust_scale(resp[clean_mask])
    positive_resp = resp[clean_mask & (resp > 0)]
    percentile_height = float(np.nanpercentile(positive_resp, PEAK_HEIGHT_PERCENTILE)) if positive_resp.size else 0.0
    min_peak_height = max(percentile_height, 0.35 * scale, 1e-6)
    raw_peaks, _ = signal.find_peaks(prob, height=0.35, distance=distance)

    aligned_peaks: list[int] = []
    align_radius = int(round(0.30 * cfg.fs))
    for idx in raw_peaks:
        left = max(0, idx - align_radius)
        right = min(resp.size, idx + align_radius + 1)
        if right <= left or not np.any(clean_mask[left:right]):
            continue
        local = resp[left:right]
        aligned = left + int(np.nanargmax(local))
        if clean_mask[aligned] and prob[aligned] >= 0.35 and resp[aligned] >= min_peak_height:
            aligned_peaks.append(aligned)

    aligned_peaks = sorted(set(aligned_peaks))
    filtered: list[int] = []
    min_gap = distance
    for idx in aligned_peaks:
        if not filtered or idx - filtered[-1] >= min_gap:
            filtered.append(idx)
        elif resp[idx] > resp[filtered[-1]]:
            filtered[-1] = idx
    return prob, np.asarray(filtered, dtype=int)


def choose_demo_window(clean_segments: pd.DataFrame, cfg: GateConfig) -> tuple[float, float]:
    usable = clean_segments[clean_segments["duration_sec"] >= DEMO_SPAN_SEC].copy()
    if usable.empty:
        usable = clean_segments.sort_values("duration_sec", ascending=False).head(1).copy()
    else:
        usable = usable.sort_values("duration_sec", ascending=False).head(1)

    row = usable.iloc[0]
    start = float(row.start_sec)
    end = float(row.end_sec)
    if end - start <= DEMO_SPAN_SEC:
        return start, end
    center = 0.5 * (start + end)
    half = 0.5 * DEMO_SPAN_SEC
    return center - half, center + half


def peak_table(t: np.ndarray, resp: np.ndarray, prob: np.ndarray, peaks: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, idx in enumerate(peaks):
        rows.append(
            {
                "peak_id": i,
                "sample_idx": int(idx),
                "time_sec": float(t[idx]),
                "resp_value": float(resp[idx]),
                "peak_probability": float(prob[idx]),
            }
        )
    return pd.DataFrame(rows)


def plot_demo(
    out_path: Path,
    data: dict[str, np.ndarray],
    gate: dict[str, np.ndarray | float],
    baseline: np.ndarray,
    resp: np.ndarray,
    pseudo_peaks: np.ndarray,
    prob: np.ndarray,
    ml_peaks: np.ndarray,
    start_sec: float,
    end_sec: float,
) -> None:
    t = data["t"]
    sl = (t >= start_sec) & (t <= end_sec)

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(t[sl], data["pvdf"][sl], linewidth=0.8, label="PVDF")
    axes[0].plot(t[sl], baseline[sl], linewidth=1.0, label="10s baseline")
    axes[0].fill_between(
        t[sl],
        np.nanmin(data["pvdf"][sl]),
        np.nanmax(data["pvdf"][sl]),
        np.asarray(gate["motion_mask"], dtype=float)[sl],
        color="tab:red",
        alpha=0.12,
        label="motion",
    )
    axes[0].set_ylabel("V")
    axes[0].legend(loc="upper right")

    axes[1].plot(t[sl], resp[sl], color="tab:blue", linewidth=0.8, label="detrended PVDF")
    local_pseudo = pseudo_peaks[(t[pseudo_peaks] >= start_sec) & (t[pseudo_peaks] <= end_sec)]
    axes[1].scatter(t[local_pseudo], resp[local_pseudo], color="tab:gray", s=22, label="pseudo peaks", zorder=4)
    axes[1].set_ylabel("resp")
    axes[1].legend(loc="upper right")

    axes[2].plot(t[sl], prob[sl], color="tab:green", linewidth=0.8, label="ML peak probability")
    local_ml = ml_peaks[(t[ml_peaks] >= start_sec) & (t[ml_peaks] <= end_sec)]
    axes[2].scatter(t[local_ml], prob[local_ml], color="tab:red", s=24, label="ML peaks", zorder=4)
    axes[2].axhline(0.35, color="black", linestyle="--", linewidth=0.8, label="prob threshold")
    axes[2].set_ylabel("P(peak)")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(loc="upper right")

    axes[3].plot(t[sl], resp[sl], color="tab:blue", linewidth=0.75)
    axes[3].scatter(t[local_ml], resp[local_ml], color="tab:red", s=28, label="predicted peaks", zorder=4)
    for idx in local_ml:
        axes[3].axvline(t[idx], color="tab:red", alpha=0.15, linewidth=1)
    axes[3].set_ylabel("peaks")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    cfg = GateConfig(**USER_CONFIG)
    data = load_dual_channel_csv(DEFAULT_CSV, cfg)
    gate = compute_motion_gate(data["pvdf"], data["pr"], cfg)
    _, clean_segments = build_segments(data, gate, cfg)
    clean_mask = ~np.asarray(gate["motion_mask"], dtype=bool)
    clean_mask &= np.asarray(data["bad_fraction"], dtype=float) <= cfg.bad_fraction_max

    baseline, resp = respiratory_component(data["pvdf"], cfg)
    features, feature_names = extract_point_features(resp, cfg)
    labels, pseudo_peaks = pseudo_peak_labels(resp, clean_mask, cfg)
    clf, train_summary = train_peak_classifier(features, labels, clean_mask)
    prob, ml_peaks = predict_peaks(clf, features, resp, clean_mask, cfg)
    start_sec, end_sec = choose_demo_window(clean_segments, cfg)

    DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    peaks_path = DEMO_OUTPUT_DIR / "ml_predicted_peaks.csv"
    plot_path = DEMO_OUTPUT_DIR / "ml_peak_detector_demo.png"
    summary_path = DEMO_OUTPUT_DIR / "ml_peak_detector_summary.json"

    peaks = peak_table(data["t"], resp, prob, ml_peaks)
    peaks.to_csv(peaks_path, index=False, encoding="utf-8-sig")
    plot_demo(plot_path, data, gate, baseline, resp, pseudo_peaks, prob, ml_peaks, start_sec, end_sec)

    summary = {
        "input_csv": str(DEFAULT_CSV),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": end_sec - start_sec,
        "pseudo_peaks_total": int(pseudo_peaks.size),
        "ml_peaks_total": int(ml_peaks.size),
        "feature_names": feature_names,
        "train_summary": train_summary,
        "peaks_csv": str(peaks_path),
        "plot_png": str(plot_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
