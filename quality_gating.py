from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

try:
    import pywt
except ImportError:  # pragma: no cover - only used on machines without PyWavelets.
    pywt = None


DEFAULT_CSV = Path(r"F:\双通道睡眠实验\2026_0413晚\数据\20260414_033313.csv")


@dataclass
class GateConfig:
    fs_raw: float = 500.0
    downsample_q: int = 10
    adc_max: float = 4095.0
    adc_vref: float = 3.3
    window_sec: float = 30.0
    step_sec: float = 5.0
    motion_dilate_sec: float = 2.0
    motion_threshold_z: float = 3.0
    pvdf_weight: float = 0.5
    resp_low_hz: float = 0.10
    resp_high_hz: float = 0.60
    total_low_hz: float = 0.05
    total_high_hz: float = 1.50
    resp_peak_min_dist_sec: float = 2.5
    resp_peak_prom_ratio: float = 0.15
    rr_min_bpm: float = 6.0
    rr_max_bpm: float = 24.0
    edge_guard_sec: float = 3.0
    bad_fraction_max: float = 0.05
    motion_reject_fraction: float = 0.25
    motion_warn_fraction: float = 0.08
    min_resp_band_ratio: float = 0.35
    min_pass_quality: float = 60.0

    @property
    def fs(self) -> float:
        return self.fs_raw / self.downsample_q


def clean_adc(values: pd.Series, cfg: GateConfig) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    bad = (~np.isfinite(x)) | (x < 0) | (x > cfg.adc_max)
    if np.all(bad):
        raise ValueError("All ADC samples are invalid.")

    x = x.copy()
    x[bad] = np.nan
    x = pd.Series(x).interpolate(limit_direction="both").to_numpy(dtype=float)
    x = np.clip(x, 0, cfg.adc_max)
    return x / cfg.adc_max * cfg.adc_vref, bad


def downsample_bad_mask(bad: np.ndarray, q: int, target_len: int) -> np.ndarray:
    target_raw_len = target_len * q
    if len(bad) < target_raw_len:
        bad = np.pad(bad, (0, target_raw_len - len(bad)), constant_values=False)
    trimmed = bad[:target_raw_len]
    return trimmed.reshape(target_len, q).mean(axis=1)


def load_dual_channel_csv(csv_path: Path, cfg: GateConfig) -> dict[str, np.ndarray]:
    header = pd.read_csv(csv_path, nrows=0)
    required = {"pvdf_adc", "pr_adc"}
    missing = required.difference(header.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    usecols = ["pvdf_adc", "pr_adc"]
    if "sample_index" in header.columns:
        usecols.insert(0, "sample_index")

    df = pd.read_csv(csv_path, usecols=usecols)
    pvdf, bad_pvdf = clean_adc(df["pvdf_adc"], cfg)
    pr, bad_pr = clean_adc(df["pr_adc"], cfg)

    pvdf_ds = signal.decimate(pvdf, q=cfg.downsample_q, ftype="fir", zero_phase=True)
    pr_ds = signal.decimate(pr, q=cfg.downsample_q, ftype="fir", zero_phase=True)
    n = min(len(pvdf_ds), len(pr_ds))
    pvdf_ds = pvdf_ds[:n]
    pr_ds = pr_ds[:n]

    bad_ds = np.maximum(
        downsample_bad_mask(bad_pvdf, cfg.downsample_q, n),
        downsample_bad_mask(bad_pr, cfg.downsample_q, n),
    )

    t = np.arange(n, dtype=float) / cfg.fs
    return {
        "t": t,
        "pvdf": pvdf_ds,
        "pr": pr_ds,
        "bad_fraction": bad_ds,
        "raw_samples": np.array([len(df)], dtype=int),
    }


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    win = max(int(win), 1)
    if win <= 1:
        return np.asarray(x, dtype=float).copy()
    return pd.Series(x).rolling(win, center=True, min_periods=1).mean().to_numpy()


def robust_scale(x: np.ndarray) -> tuple[float, float]:
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    return med, 1.4826 * mad + 1e-12


def robust_positive_z(x: np.ndarray) -> np.ndarray:
    med, scale = robust_scale(x)
    return np.clip((x - med) / scale, 0.0, None)


def abs_rate(x: np.ndarray, fs: float) -> np.ndarray:
    return np.abs(np.diff(x, prepend=x[0])) * fs


def reconstruct_single(coeffs: list[np.ndarray], keep_index: int, wavelet: str, out_len: int) -> np.ndarray:
    selected = [np.zeros_like(c) for c in coeffs]
    selected[keep_index] = coeffs[keep_index]
    return pywt.waverec(selected, wavelet)[:out_len]


def bandpass(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    high_hz = min(high_hz, 0.95 * nyq)
    if low_hz <= 0 or high_hz <= low_hz:
        raise ValueError("Invalid bandpass limits.")
    sos = signal.butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def extract_pvdf_respiration(pvdf: np.ndarray, cfg: GateConfig) -> np.ndarray:
    if pywt is None:
        return bandpass(pvdf, cfg.fs, cfg.resp_low_hz, cfg.resp_high_hz)

    wavelet = "db5"
    level = min(5, pywt.dwt_max_level(len(pvdf), pywt.Wavelet(wavelet).dec_len))
    coeffs = pywt.wavedec(pvdf, wavelet=wavelet, level=level)

    cD1 = coeffs[-1]
    noise_sigma = np.median(np.abs(cD1)) / 0.6745 + 1e-12
    threshold = noise_sigma * np.sqrt(2 * np.log(len(pvdf)))
    coeffs_th = [coeffs[0].copy()]
    for c in coeffs[1:]:
        coeffs_th.append(pywt.threshold(c, value=threshold, mode="soft"))

    resp = reconstruct_single(coeffs_th, 0, wavelet, len(pvdf))
    return bandpass(resp, cfg.fs, cfg.resp_low_hz, cfg.resp_high_hz)


def channel_motion_score(x: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    smooth = moving_average(x, int(round(0.3 * fs)))
    trend = moving_average(x, int(round(8.0 * fs)))
    ac = x - trend
    env = moving_average(np.abs(ac), int(round(1.0 * fs)))

    env_rate_z = robust_positive_z(abs_rate(env, fs))
    voltage_rate_z = robust_positive_z(abs_rate(smooth, fs))
    score = 0.5 * env_rate_z + 0.5 * voltage_rate_z
    return {
        "score": score,
        "env": env,
        "env_rate_z": env_rate_z,
        "voltage_rate_z": voltage_rate_z,
    }


def expand_mask(mask: np.ndarray, radius_samples: int) -> np.ndarray:
    if radius_samples <= 0:
        return mask.astype(bool)
    kernel = np.ones(2 * radius_samples + 1, dtype=int)
    return signal.convolve(mask.astype(int), kernel, mode="same") > 0


def compute_motion_gate(pvdf: np.ndarray, pr: np.ndarray, cfg: GateConfig) -> dict[str, np.ndarray | float]:
    pvdf_motion = channel_motion_score(pvdf, cfg.fs)
    pr_motion = channel_motion_score(pr, cfg.fs)
    fused = cfg.pvdf_weight * pvdf_motion["score"] + (1.0 - cfg.pvdf_weight) * pr_motion["score"]
    fused = moving_average(fused, int(round(1.0 * cfg.fs)))

    edge = int(round(cfg.edge_guard_sec * cfg.fs))
    if edge > 0 and len(fused) > 2 * edge:
        for arr in (pvdf_motion["score"], pr_motion["score"], fused):
            arr[:edge] = 0.0
            arr[-edge:] = 0.0

    med, scale = robust_scale(fused)
    adaptive_th = med + cfg.motion_threshold_z * scale
    threshold = max(cfg.motion_threshold_z, adaptive_th)
    raw_motion = fused > threshold
    motion = expand_mask(raw_motion, int(round(cfg.motion_dilate_sec * cfg.fs)))
    return {
        "pvdf_score": pvdf_motion["score"],
        "pr_score": pr_motion["score"],
        "motion_score": fused,
        "motion_threshold": float(threshold),
        "raw_motion_mask": raw_motion,
        "motion_mask": motion,
    }


def spectral_resp_features(x: np.ndarray, cfg: GateConfig) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = signal.detrend(x, type="linear")
    if len(x) < int(8 * cfg.fs):
        return {"fft_rr_bpm": np.nan, "resp_band_ratio": np.nan, "dominance": np.nan}

    freqs, psd = signal.welch(
        x,
        fs=cfg.fs,
        window="hann",
        nperseg=len(x),
        noverlap=0,
        detrend=False,
        scaling="density",
    )
    total_mask = (freqs >= cfg.total_low_hz) & (freqs <= cfg.total_high_hz)
    resp_mask = (freqs >= cfg.resp_low_hz) & (freqs <= cfg.resp_high_hz)
    if not np.any(resp_mask) or not np.any(total_mask):
        return {"fft_rr_bpm": np.nan, "resp_band_ratio": np.nan, "dominance": np.nan}

    total_power = float(np.trapz(psd[total_mask], freqs[total_mask])) + 1e-12
    resp_power = float(np.trapz(psd[resp_mask], freqs[resp_mask]))
    resp_psd = psd[resp_mask]
    resp_freqs = freqs[resp_mask]
    peak_idx = int(np.argmax(resp_psd))
    rr_bpm = float(resp_freqs[peak_idx] * 60.0)
    dominance = float(resp_psd[peak_idx] / (np.median(resp_psd) + 1e-12))
    return {
        "fft_rr_bpm": rr_bpm,
        "resp_band_ratio": resp_power / total_power,
        "dominance": dominance,
    }


def peak_resp_features(x: np.ndarray, cfg: GateConfig) -> dict[str, float]:
    x = signal.detrend(np.asarray(x, dtype=float), type="linear")
    amp = float(np.nanpercentile(x, 95) - np.nanpercentile(x, 5))
    if len(x) < int(8 * cfg.fs) or amp <= 1e-12:
        return {"peak_rr_bpm": np.nan, "peak_count": 0.0, "ibi_cv": np.nan}

    min_distance = max(1, int(round(cfg.resp_peak_min_dist_sec * cfg.fs)))
    prominence = cfg.resp_peak_prom_ratio * amp
    candidates = []
    for sign in (1.0, -1.0):
        peaks, props = signal.find_peaks(sign * x, distance=min_distance, prominence=prominence)
        if len(peaks) < 3:
            continue
        periods = np.diff(peaks) / cfg.fs
        rr_bpm = 60.0 / float(np.median(periods))
        ibi_cv = float(np.std(periods) / (np.mean(periods) + 1e-12))
        med_prom = float(np.median(props.get("prominences", [0.0])) / (amp + 1e-12))
        score = med_prom - ibi_cv + 0.03 * len(peaks)
        candidates.append((score, rr_bpm, len(peaks), ibi_cv))

    if not candidates:
        return {"peak_rr_bpm": np.nan, "peak_count": 0.0, "ibi_cv": np.nan}

    _, rr_bpm, peak_count, ibi_cv = max(candidates, key=lambda item: item[0])
    return {"peak_rr_bpm": float(rr_bpm), "peak_count": float(peak_count), "ibi_cv": float(ibi_cv)}


def respiration_features(x: np.ndarray, cfg: GateConfig) -> dict[str, float]:
    spec = spectral_resp_features(x, cfg)
    peaks = peak_resp_features(x, cfg)
    rr_bpm = peaks["peak_rr_bpm"]
    rr_method = 1.0
    if not np.isfinite(rr_bpm):
        rr_bpm = spec["fft_rr_bpm"]
        rr_method = 0.0
    return {
        "rr_bpm": float(rr_bpm),
        "rr_method_peak": rr_method,
        **spec,
        **peaks,
    }


def iter_windows(n: int, cfg: GateConfig) -> list[tuple[int, int]]:
    win = int(round(cfg.window_sec * cfg.fs))
    step = int(round(cfg.step_sec * cfg.fs))
    if n < win:
        return [(0, n)]
    starts = list(range(0, n - win + 1, step))
    if starts[-1] + win < n:
        starts.append(n - win)
    return [(s, min(s + win, n)) for s in starts]


def classify_window(row: dict[str, float], cfg: GateConfig) -> tuple[str, bool, float]:
    rr_ok = np.isfinite(row["pvdf_rr_bpm"]) and cfg.rr_min_bpm <= row["pvdf_rr_bpm"] <= cfg.rr_max_bpm
    quality = 50.0
    quality += 30.0 * np.clip((row["pvdf_resp_band_ratio"] - 0.20) / 0.60, 0.0, 1.0)
    quality += 15.0 * np.clip((row["pvdf_dominance"] - 1.5) / 8.0, 0.0, 1.0)
    if np.isfinite(row["pvdf_ibi_cv"]):
        quality += 10.0 * np.clip((0.35 - row["pvdf_ibi_cv"]) / 0.35, 0.0, 1.0)
    quality -= 80.0 * row["motion_fraction"]
    quality -= 120.0 * row["bad_fraction"]

    if np.isfinite(row["rr_abs_diff_bpm"]):
        quality -= min(20.0, 1.5 * row["rr_abs_diff_bpm"])

    quality = float(np.clip(quality, 0.0, 100.0))

    if row["bad_fraction"] > cfg.bad_fraction_max:
        label = "invalid"
    elif row["motion_fraction"] > cfg.motion_reject_fraction:
        label = "motion"
    elif (not rr_ok) or row["pvdf_resp_band_ratio"] < cfg.min_resp_band_ratio:
        label = "low_quality"
    elif row["motion_fraction"] > cfg.motion_warn_fraction:
        label = "usable"
    else:
        label = "good"

    pass_gate = label in {"good", "usable"} and quality >= cfg.min_pass_quality
    return label, pass_gate, quality


def build_window_table(
    data: dict[str, np.ndarray],
    resp: np.ndarray,
    gate: dict[str, np.ndarray | float],
    cfg: GateConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    for start, end in iter_windows(len(data["t"]), cfg):
        pvdf_feat = respiration_features(resp[start:end], cfg)
        pr_resp = bandpass(data["pr"][start:end], cfg.fs, cfg.resp_low_hz, cfg.resp_high_hz)
        pr_feat = respiration_features(pr_resp, cfg)

        rr_abs_diff = np.nan
        if (
            np.isfinite(pvdf_feat["rr_bpm"])
            and np.isfinite(pr_feat["rr_bpm"])
            and pr_feat["resp_band_ratio"] >= 0.15
        ):
            rr_abs_diff = abs(pvdf_feat["rr_bpm"] - pr_feat["rr_bpm"])

        row: dict[str, float | str | bool] = {
            "start_sec": float(data["t"][start]),
            "end_sec": float(data["t"][end - 1]) if end > start else float(data["t"][start]),
            "bad_fraction": float(np.mean(data["bad_fraction"][start:end])),
            "motion_fraction": float(np.mean(gate["motion_mask"][start:end])),
            "mean_motion_score": float(np.mean(gate["motion_score"][start:end])),
            "pvdf_rr_bpm": pvdf_feat["rr_bpm"],
            "pvdf_fft_rr_bpm": pvdf_feat["fft_rr_bpm"],
            "pvdf_resp_band_ratio": pvdf_feat["resp_band_ratio"],
            "pvdf_dominance": pvdf_feat["dominance"],
            "pvdf_peak_count": pvdf_feat["peak_count"],
            "pvdf_ibi_cv": pvdf_feat["ibi_cv"],
            "pvdf_rr_method_peak": pvdf_feat["rr_method_peak"],
            "pr_rr_bpm": pr_feat["rr_bpm"],
            "pr_fft_rr_bpm": pr_feat["fft_rr_bpm"],
            "pr_resp_band_ratio": pr_feat["resp_band_ratio"],
            "pr_dominance": pr_feat["dominance"],
            "pr_peak_count": pr_feat["peak_count"],
            "pr_ibi_cv": pr_feat["ibi_cv"],
            "pr_rr_method_peak": pr_feat["rr_method_peak"],
            "rr_abs_diff_bpm": rr_abs_diff,
        }
        label, pass_gate, quality = classify_window(row, cfg)
        row["label"] = label
        row["pass_gate"] = pass_gate
        row["quality_score"] = quality
        rows.append(row)

    return pd.DataFrame(rows)


def normalized_for_plot(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    lo, hi = np.nanpercentile(x, [5, 95])
    scale = hi - lo
    if scale <= 1e-12:
        scale = np.nanstd(x) + 1e-12
    return (x - med) / scale


def set_robust_ylim(ax: plt.Axes, y: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> None:
    finite = np.asarray(y, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    lo, hi = np.nanpercentile(finite, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)


def plot_overview(
    out_path: Path,
    data: dict[str, np.ndarray],
    resp: np.ndarray,
    gate: dict[str, np.ndarray | float],
    windows: pd.DataFrame,
    cfg: GateConfig,
) -> None:
    t = data["t"]
    max_points = 40000
    stride = max(1, int(np.ceil(len(t) / max_points)))
    sl = slice(None, None, stride)

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(t[sl], normalized_for_plot(resp)[sl], linewidth=0.8, label="PVDF respiration")
    axes[0].plot(t[sl], normalized_for_plot(data["pr"])[sl], linewidth=0.6, alpha=0.75, label="PR")
    set_robust_ylim(axes[0], np.r_[normalized_for_plot(resp)[sl], normalized_for_plot(data["pr"])[sl]])
    axes[0].set_ylabel("norm.")
    axes[0].legend(loc="upper right")

    axes[1].plot(t[sl], gate["pvdf_score"][sl], linewidth=0.6, label="PVDF score")
    axes[1].plot(t[sl], gate["pr_score"][sl], linewidth=0.6, label="PR score")
    set_robust_ylim(axes[1], np.r_[gate["pvdf_score"][sl], gate["pr_score"][sl]], 0.0, 99.0)
    axes[1].set_ylabel("z")
    axes[1].legend(loc="upper right")

    axes[2].plot(t[sl], gate["motion_score"][sl], color="tab:red", linewidth=0.8, label="fused motion score")
    axes[2].axhline(gate["motion_threshold"], color="black", linestyle="--", linewidth=1.0, label="threshold")
    set_robust_ylim(axes[2], gate["motion_score"][sl], 0.0, 99.5)
    y_min, y_max = axes[2].get_ylim()
    motion_t = t[gate["motion_mask"]]
    if len(motion_t) > 0:
        axes[2].scatter(
            motion_t[:: max(1, len(motion_t) // 2500)],
            np.full_like(motion_t[:: max(1, len(motion_t) // 2500)], y_min + 0.05 * (y_max - y_min)),
            s=4,
            color="tab:green",
            label="motion mask",
        )
    axes[2].set_ylabel("score")
    axes[2].legend(loc="upper right")

    colors = windows["pass_gate"].map({True: "tab:blue", False: "tab:red"}).to_numpy()
    mids = 0.5 * (windows["start_sec"].to_numpy() + windows["end_sec"].to_numpy())
    axes[3].scatter(mids, windows["quality_score"], c=colors, s=14)
    axes[3].axhline(cfg.min_pass_quality, color="black", linestyle="--", linewidth=1.0)
    axes[3].set_ylabel("SQI")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylim(-2, 102)

    fig.suptitle("Dual-channel quality gating overview", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_detail(
    out_path: Path,
    data: dict[str, np.ndarray],
    resp: np.ndarray,
    gate: dict[str, np.ndarray | float],
    windows: pd.DataFrame,
    cfg: GateConfig,
    span_sec: float = 120.0,
) -> None:
    t = data["t"]
    if len(t) == 0:
        return

    center = int(np.argmax(gate["motion_score"]))
    half = int(round(0.5 * span_sec * cfg.fs))
    start = max(0, center - half)
    end = min(len(t), center + half)
    sl = slice(start, end)

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t[sl], normalized_for_plot(resp[sl]), linewidth=1.0, label="PVDF respiration")
    axes[0].plot(t[sl], normalized_for_plot(data["pr"][sl]), linewidth=0.8, alpha=0.8, label="PR")
    axes[0].set_ylabel("norm.")
    axes[0].legend(loc="upper right")

    axes[1].plot(t[sl], gate["pvdf_score"][sl], linewidth=0.8, label="PVDF score")
    axes[1].plot(t[sl], gate["pr_score"][sl], linewidth=0.8, label="PR score")
    axes[1].plot(t[sl], gate["motion_score"][sl], color="tab:red", linewidth=1.0, label="fused")
    axes[1].axhline(gate["motion_threshold"], color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("score")
    axes[1].legend(loc="upper right")

    local = windows[(windows["end_sec"] >= t[start]) & (windows["start_sec"] <= t[end - 1])]
    if len(local):
        mids = 0.5 * (local["start_sec"].to_numpy() + local["end_sec"].to_numpy())
        colors = local["pass_gate"].map({True: "tab:blue", False: "tab:red"}).to_numpy()
        axes[2].scatter(mids, local["quality_score"], c=colors, s=28)
    axes[2].axhline(cfg.min_pass_quality, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("SQI")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(-2, 102)

    fig.suptitle("Detail around strongest detected motion", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_summary(
    csv_path: Path,
    data: dict[str, np.ndarray],
    gate: dict[str, np.ndarray | float],
    windows: pd.DataFrame,
    cfg: GateConfig,
) -> dict[str, object]:
    passed = windows[windows["pass_gate"]]
    return {
        "input_csv": str(csv_path),
        "fs_raw": cfg.fs_raw,
        "fs_processed": cfg.fs,
        "raw_samples": int(data["raw_samples"][0]),
        "processed_samples": int(len(data["t"])),
        "duration_sec": float(data["t"][-1] - data["t"][0]) if len(data["t"]) else 0.0,
        "window_sec": cfg.window_sec,
        "step_sec": cfg.step_sec,
        "motion_threshold": float(gate["motion_threshold"]),
        "motion_seconds": float(np.sum(gate["motion_mask"]) / cfg.fs),
        "motion_ratio": float(np.mean(gate["motion_mask"])),
        "windows_total": int(len(windows)),
        "windows_passed": int(windows["pass_gate"].sum()),
        "window_pass_ratio": float(windows["pass_gate"].mean()) if len(windows) else 0.0,
        "median_rr_bpm_passed": float(passed["pvdf_rr_bpm"].median()) if len(passed) else None,
        "median_quality_passed": float(passed["quality_score"].median()) if len(passed) else None,
        "label_counts": windows["label"].value_counts().to_dict(),
        "config": asdict(cfg),
    }


def write_text_summary(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "Dual-channel PVDF + piezoresistive quality gating summary",
        f"Input: {summary['input_csv']}",
        f"Duration: {summary['duration_sec']:.1f} s",
        f"Processed sampling rate: {summary['fs_processed']:.1f} Hz",
        f"Motion ratio: {summary['motion_ratio']:.3f}",
        f"Window pass ratio: {summary['window_pass_ratio']:.3f}",
        f"Passed windows: {summary['windows_passed']} / {summary['windows_total']}",
        f"Median RR in passed windows: {summary['median_rr_bpm_passed']}",
        f"Median SQI in passed windows: {summary['median_quality_passed']}",
        f"Label counts: {summary['label_counts']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_csv(csv_path: Path, out_dir: Path, cfg: GateConfig) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_dual_channel_csv(csv_path, cfg)
    resp = extract_pvdf_respiration(data["pvdf"], cfg)
    gate = compute_motion_gate(data["pvdf"], data["pr"], cfg)
    windows = build_window_table(data, resp, gate, cfg)

    windows_path = out_dir / "quality_windows.csv"
    summary_json_path = out_dir / "summary.json"
    summary_txt_path = out_dir / "summary.txt"
    figure_path = out_dir / "quality_gating_overview.png"
    detail_figure_path = out_dir / "quality_gating_detail.png"

    windows.to_csv(windows_path, index=False, encoding="utf-8-sig")
    summary = make_summary(csv_path, data, gate, windows, cfg)
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_text_summary(summary_txt_path, summary)
    plot_overview(figure_path, data, resp, gate, windows, cfg)
    plot_detail(detail_figure_path, data, resp, gate, windows, cfg)

    summary["outputs"] = {
        "windows_csv": str(windows_path),
        "summary_json": str(summary_json_path),
        "summary_txt": str(summary_txt_path),
        "overview_png": str(figure_path),
        "detail_png": str(detail_figure_path),
    }
    return summary


def resolve_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.csv"))
    return [path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PVDF + piezoresistive dual-channel quality gating.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV, help="CSV file or a directory of CSV files.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Directory for generated tables/figures.")
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--step-sec", type=float, default=5.0)
    parser.add_argument("--motion-threshold-z", type=float, default=3.0)
    parser.add_argument("--pvdf-weight", type=float, default=0.5)
    parser.add_argument("--fs-raw", type=float, default=500.0)
    parser.add_argument("--downsample-q", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GateConfig(
        fs_raw=args.fs_raw,
        downsample_q=args.downsample_q,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        motion_threshold_z=args.motion_threshold_z,
        pvdf_weight=args.pvdf_weight,
    )

    csv_paths = resolve_inputs(args.input)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found: {args.input}")

    summaries = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        child_out = args.out_dir / csv_path.stem if len(csv_paths) > 1 else args.out_dir
        print(f"Analyzing {csv_path}")
        summaries.append(analyze_csv(csv_path, child_out, cfg))

    if len(summaries) > 1:
        aggregate_path = args.out_dir / "batch_summary.json"
        aggregate_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        brief_rows = []
        for item in summaries:
            brief_rows.append(
                {
                    "input_csv": Path(item["input_csv"]).name,
                    "duration_sec": item["duration_sec"],
                    "motion_ratio": item["motion_ratio"],
                    "window_pass_ratio": item["window_pass_ratio"],
                    "windows_passed": item["windows_passed"],
                    "windows_total": item["windows_total"],
                    "median_rr_bpm_passed": item["median_rr_bpm_passed"],
                    "median_quality_passed": item["median_quality_passed"],
                }
            )
        brief_path = args.out_dir / "batch_summary.csv"
        pd.DataFrame(brief_rows).to_csv(brief_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {aggregate_path}")
        print(f"Wrote {brief_path}")
    else:
        print(json.dumps(summaries[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
