from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

DATE_NAME = "2026_0413晚"
CSV_NAME = "20260414_030313"
DEFAULT_CSV = Path(f"F:\\双通道睡眠实验\\{DATE_NAME}\\数据\\{CSV_NAME}.csv")
OUTPUT_DIR = Path(f"outputs\\{DATE_NAME}\\{CSV_NAME}_analysis")

# 直接在这里改参数；运行 `python .\quality_gating.py` 即可。
USER_CONFIG = {
    "fs_raw": 500.0, # 原始采样率，单位Hz
    "downsample_q": 10, # 下采样倍数，越大计算越快，但可能丢失短暂运动段
    "motion_threshold_z": 2.0, # 运动得分阈值，单位为 robust z-score，越大越严格
    "pre_motion_dilate_sec": 0.0, # 候选体动先向两边扩张，单位秒
    "motion_merge_gap_sec": 5.0, # 合并相隔不超过这个时间的候选事件，单位秒
    "event_split_min_sec": 25.0, # 只分离超过该时长的长候选事件，单位秒
    "event_split_gap_sec": 2.0, # 长候选事件内部间隔超过该值则切开，单位秒
    "motion_dilate_sec": 2.0, # 最终体动段边界扩张，单位秒
    "pvdf_weight": 0.5, # PVDF信号权重
    "voltage_rate_weight": 0.5, # 电压变化率权重
    "envelope_rate_weight": 0.5, # 包络变化率权重
    "kalman_context_sec": 25.0, # 每个候选段前后用于建立呼吸模型的clean时长，单位秒
    "kalman_residual_threshold_z": 6.0, # 候选段Kalman残差确认阈值，越大越严格
    "pr_contact_threshold_z": 12.0, # PR接触变化确认阈值，越大越严格
    "min_clean_segment_sec": 10.0, # 最小清洁段长度，单位秒
}


@dataclass
class GateConfig:
    fs_raw: float = 500.0
    downsample_q: int = 10
    adc_max: float = 4095.0
    adc_vref: float = 3.3

    motion_threshold_z: float = 3.0
    pre_motion_dilate_sec: float = 1.0
    motion_merge_gap_sec: float = 5.0
    event_split_min_sec: float = 25.0
    event_split_gap_sec: float = 2.0
    motion_dilate_sec: float = 2.0
    edge_guard_sec: float = 3.0

    pvdf_weight: float = 0.6
    voltage_rate_weight: float = 0.5
    envelope_rate_weight: float = 0.5

    kalman_context_sec: float = 25.0
    kalman_residual_threshold_z: float = 4.0
    pr_contact_threshold_z: float = 12.0

    bad_fraction_max: float = 0.05
    min_clean_segment_sec: float = 10.0

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

    return {
        "t": np.arange(n, dtype=float) / cfg.fs,
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


def feature_weights(cfg: GateConfig) -> np.ndarray:
    weights = np.array([cfg.voltage_rate_weight, cfg.envelope_rate_weight], dtype=float)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0:
        return np.array([0.5, 0.5], dtype=float)
    return weights / total


def channel_motion_features(x: np.ndarray, cfg: GateConfig) -> dict[str, np.ndarray]:
    fs = cfg.fs
    smooth = moving_average(x, int(round(0.3 * fs)))
    trend = moving_average(x, int(round(8.0 * fs)))
    ac = x - trend
    env = moving_average(np.abs(ac), int(round(1.0 * fs)))

    return {
        "voltage_rate_z": robust_positive_z(abs_rate(smooth, fs)),
        "env_rate_z": robust_positive_z(abs_rate(env, fs)),
    }


def mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]] + 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def expand_mask(mask: np.ndarray, radius_samples: int) -> np.ndarray:
    if radius_samples <= 0:
        return mask.astype(bool)
    kernel = np.ones(2 * radius_samples + 1, dtype=int)
    return signal.convolve(mask.astype(int), kernel, mode="same") > 0


def merge_close_motion_segments(
    motion_mask: np.ndarray,
    cfg: GateConfig,
    *,
    block_merge_mask: np.ndarray | None = None,
) -> np.ndarray:
    if cfg.motion_merge_gap_sec <= 0:
        return motion_mask.astype(bool)

    runs = mask_runs(motion_mask)
    if not runs:
        return motion_mask.astype(bool)

    merged = np.zeros_like(motion_mask, dtype=bool)
    gap_samples = int(round(cfg.motion_merge_gap_sec * cfg.fs))
    cur_start, cur_end = runs[0]
    for start, end in runs[1:]:
        gap_is_blocked = (
            block_merge_mask is not None
            and start > cur_end
            and np.any(block_merge_mask[cur_end:start])
        )
        if start - cur_end <= gap_samples and not gap_is_blocked:
            cur_end = end
        else:
            merged[cur_start:cur_end] = True
            cur_start, cur_end = start, end
    merged[cur_start:cur_end] = True
    return merged


def merge_runs_by_gap(mask: np.ndarray, gap_samples: int) -> np.ndarray:
    runs = mask_runs(mask)
    if not runs:
        return mask.astype(bool)

    merged = np.zeros_like(mask, dtype=bool)
    cur_start, cur_end = runs[0]
    for start, end in runs[1:]:
        if start - cur_end <= gap_samples:
            cur_end = end
        else:
            merged[cur_start:cur_end] = True
            cur_start, cur_end = start, end
    merged[cur_start:cur_end] = True
    return merged


def otsu_threshold(x: np.ndarray, bins: int = 128) -> float | None:
    finite = np.asarray(x, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 8:
        return None

    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if hi <= lo:
        return None

    hist, edges = np.histogram(finite, bins=bins, range=(lo, hi))
    total = float(hist.sum())
    if total <= 0:
        return None

    centers = (edges[:-1] + edges[1:]) * 0.5
    weight_bg = np.cumsum(hist).astype(float)
    weight_fg = total - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not np.any(valid):
        return None

    sum_total = float(np.sum(hist * centers))
    sum_bg = np.cumsum(hist * centers)
    mean_bg = np.divide(sum_bg, weight_bg, out=np.zeros_like(sum_bg), where=weight_bg > 0)
    mean_fg = np.divide(sum_total - sum_bg, weight_fg, out=np.zeros_like(sum_bg), where=weight_fg > 0)
    between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between_var[~valid] = -np.inf
    return float(centers[int(np.argmax(between_var))])


def adaptive_event_core(local_score: np.ndarray, cfg: GateConfig) -> np.ndarray | None:
    threshold = otsu_threshold(local_score)
    if threshold is None:
        return None

    core = np.asarray(local_score, dtype=float) >= max(float(cfg.motion_threshold_z), threshold)
    finite = np.asarray(local_score, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None

    high = finite[finite >= threshold]
    low = finite[finite < threshold]
    high_ratio = high.size / finite.size
    if high.size == 0 or low.size == 0:
        return None
    if high_ratio < 0.005 or high_ratio > 0.60:
        return None

    _, scale = robust_scale(finite)
    separation = (float(np.nanmean(high)) - float(np.nanmean(low))) / max(scale, 1e-12)
    if separation < 1.5:
        return None

    return core


def separate_long_candidate_events(
    candidate_event: np.ndarray,
    motion_score: np.ndarray,
    pvdf_score: np.ndarray,
    pr_score: np.ndarray,
    cfg: GateConfig,
) -> np.ndarray:
    separated = np.zeros_like(candidate_event, dtype=bool)
    min_len = int(round(cfg.event_split_min_sec * cfg.fs))
    split_gap = int(round(cfg.event_split_gap_sec * cfg.fs))
    pad = int(round(cfg.pre_motion_dilate_sec * cfg.fs))

    for start, end in mask_runs(candidate_event):
        if end - start <= min_len:
            separated[start:end] = True
            continue

        local_cores = []
        for score in (motion_score, pvdf_score, pr_score):
            local_core = adaptive_event_core(score[start:end], cfg)
            if local_core is not None and np.any(local_core):
                local_cores.append(local_core)

        if not local_cores:
            # 长事件如果没有可靠的高分核心，更像体位/接触漂移，不再送入Kalman验证。
            continue

        local_core = np.logical_or.reduce(local_cores)
        local = expand_mask(local_core, pad)
        local = merge_runs_by_gap(local, split_gap)
        separated[start:end] = local & candidate_event[start:end]

    return separated


def kalman_breath_signal(pvdf: np.ndarray, cfg: GateConfig) -> np.ndarray:
    trend = moving_average(pvdf, int(round(8.0 * cfg.fs)))
    resp = pvdf - trend
    return moving_average(resp, int(round(0.15 * cfg.fs)))


def estimate_breath_hz(context: np.ndarray, cfg: GateConfig) -> float:
    min_breath_hz = 0.08
    max_breath_hz = 0.60
    default_breath_hz = 0.25
    x = np.asarray(context, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < max(8, int(round(4.0 * cfg.fs))):
        return default_breath_hz

    x = x - np.nanmedian(x)
    nperseg = min(x.size, max(16, int(round(20.0 * cfg.fs))))
    freqs, power = signal.welch(x, fs=cfg.fs, nperseg=nperseg)
    band = (freqs >= min_breath_hz) & (freqs <= max_breath_hz)
    if not np.any(band):
        return default_breath_hz

    band_power = power[band]
    if band_power.size == 0 or not np.any(np.isfinite(band_power)) or float(np.nanmax(band_power)) <= 0:
        return default_breath_hz
    return float(freqs[band][int(np.nanargmax(band_power))])


def oscillator_transition(freq_hz: float, cfg: GateConfig) -> np.ndarray:
    theta = 2.0 * np.pi * float(freq_hz) / cfg.fs
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, s], [-s, c]], dtype=float)


def kalman_predict(x: np.ndarray, p: np.ndarray, f: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = f @ x
    p = f @ p @ f.T + q
    return x, p


def kalman_update(x: np.ndarray, p: np.ndarray, y: float, r: float) -> tuple[np.ndarray, np.ndarray, float]:
    innovation = float(y - x[0])
    s = float(p[0, 0] + r)
    k = p[:, 0] / max(s, 1e-12)
    x = x + k * innovation
    h = np.array([[1.0, 0.0]], dtype=float)
    p = (np.eye(2) - k[:, None] @ h) @ p
    return x, p, innovation


def kalman_segment_residual(
    resp: np.ndarray,
    candidate_mask: np.ndarray,
    start: int,
    end: int,
    cfg: GateConfig,
) -> tuple[float, np.ndarray]:
    context_radius = int(round(cfg.kalman_context_sec * cfg.fs))
    min_context = int(round(8.0 * cfg.fs))
    pre_start = max(0, start - context_radius)
    post_end = min(len(resp), end + context_radius)

    before_idx = np.arange(pre_start, start, dtype=int)
    after_idx = np.arange(end, post_end, dtype=int)
    before_idx = before_idx[~candidate_mask[before_idx]]
    after_idx = after_idx[~candidate_mask[after_idx]]
    context_idx = np.r_[before_idx, after_idx]

    min_before = max(3, int(round(2.0 * cfg.fs)))
    if context_idx.size < min_context or before_idx.size < min_before:
        fallback_score = max(float(cfg.kalman_residual_threshold_z) + 1.0, 1.0)
        return fallback_score, np.full(end - start, fallback_score, dtype=float)

    context = resp[context_idx]
    center, amp_scale = robust_scale(context)
    y = (resp - center) / max(amp_scale, 1e-12)
    freq_hz = estimate_breath_hz(context, cfg)
    f = oscillator_transition(freq_hz, cfg)
    q = np.eye(2, dtype=float) * 1e-4
    r = 0.05

    x = np.array([float(y[before_idx[0]]), 0.0], dtype=float)
    p = np.eye(2, dtype=float)
    innovations: list[float] = []
    prev = int(before_idx[0])

    for idx in before_idx[1:]:
        steps = max(1, min(int(idx - prev), int(round(2.0 * cfg.fs))))
        for _ in range(steps):
            x, p = kalman_predict(x, p, f, q)
        x, p, innovation = kalman_update(x, p, float(y[idx]), r)
        innovations.append(innovation)
        prev = int(idx)

    if not innovations:
        residual_scale = 1.0
    else:
        _, residual_scale = robust_scale(np.asarray(innovations, dtype=float))
        residual_scale = max(residual_scale, r)

    residual_z = np.zeros(end - start, dtype=float)
    prev = int(before_idx[-1])
    for out_i, idx in enumerate(range(start, end)):
        steps = max(1, min(int(idx - prev), int(round(2.0 * cfg.fs))))
        for _ in range(steps):
            x, p = kalman_predict(x, p, f, q)
        residual_z[out_i] = abs(float(y[idx]) - float(x[0])) / residual_scale
        prev = int(idx)

    score = float(np.nanpercentile(residual_z, 95.0))
    return score, residual_z


def verify_candidate_events(
    candidate_mask: np.ndarray,
    pvdf: np.ndarray,
    pr_score: np.ndarray,
    cfg: GateConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    verified = np.zeros_like(candidate_mask, dtype=bool)
    residual_series = np.zeros_like(pvdf, dtype=float)
    kalman_score_series = np.zeros_like(pvdf, dtype=float)
    pr_score_series = np.zeros_like(pr_score, dtype=float)
    verification_series = np.zeros_like(pvdf, dtype=float)
    resp = kalman_breath_signal(pvdf, cfg)

    for start, end in mask_runs(candidate_mask):
        kalman_score, residual_z = kalman_segment_residual(resp, candidate_mask, start, end, cfg)
        contact_score = float(np.nanpercentile(pr_score[start:end], 95.0))
        kalman_norm = kalman_score / max(float(cfg.kalman_residual_threshold_z), 1e-12)
        pr_norm = contact_score / max(float(cfg.pr_contact_threshold_z), 1e-12)
        verification_score = max(kalman_norm, pr_norm)

        residual_series[start:end] = residual_z
        kalman_score_series[start:end] = kalman_score
        pr_score_series[start:end] = contact_score
        verification_series[start:end] = verification_score
        if verification_score >= 1.0:
            verified[start:end] = True

    return verified, residual_series, kalman_score_series, pr_score_series, verification_series


def compute_motion_gate(pvdf: np.ndarray, pr: np.ndarray, cfg: GateConfig) -> dict[str, np.ndarray | float]:
    pvdf_motion = channel_motion_features(pvdf, cfg)
    pr_motion = channel_motion_features(pr, cfg)

    pvdf_w = float(np.clip(cfg.pvdf_weight, 0.0, 1.0))
    pr_w = 1.0 - pvdf_w
    weights = feature_weights(cfg)

    voltage_score = pvdf_w * pvdf_motion["voltage_rate_z"] + pr_w * pr_motion["voltage_rate_z"]
    envelope_score = pvdf_w * pvdf_motion["env_rate_z"] + pr_w * pr_motion["env_rate_z"]
    motion_score = weights[0] * voltage_score + weights[1] * envelope_score
    motion_score = moving_average(motion_score, int(round(1.0 * cfg.fs)))

    pvdf_score = weights[0] * pvdf_motion["voltage_rate_z"] + weights[1] * pvdf_motion["env_rate_z"]
    pr_score = weights[0] * pr_motion["voltage_rate_z"] + weights[1] * pr_motion["env_rate_z"]

    edge = int(round(cfg.edge_guard_sec * cfg.fs))
    if edge > 0 and len(motion_score) > 2 * edge:
        for arr in (pvdf_score, pr_score, voltage_score, envelope_score, motion_score):
            arr[:edge] = 0.0
            arr[-edge:] = 0.0

    threshold = max(0.0, float(cfg.motion_threshold_z))
    raw_candidate = motion_score > threshold
    pre_dilated_candidate = expand_mask(raw_candidate, int(round(cfg.pre_motion_dilate_sec * cfg.fs)))
    candidate_event = merge_close_motion_segments(pre_dilated_candidate, cfg)
    separated_event = separate_long_candidate_events(candidate_event, motion_score, pvdf_score, pr_score, cfg)
    verified_motion, kalman_residual_series, kalman_score_series, pr_contact_score_series, verification_score = verify_candidate_events(
        separated_event,
        pvdf,
        pr_score,
        cfg,
    )
    final_motion = expand_mask(verified_motion, int(round(cfg.motion_dilate_sec * cfg.fs)))

    return {
        "pvdf_voltage_rate_z": pvdf_motion["voltage_rate_z"],
        "pr_voltage_rate_z": pr_motion["voltage_rate_z"],
        "pvdf_env_rate_z": pvdf_motion["env_rate_z"],
        "pr_env_rate_z": pr_motion["env_rate_z"],
        "pvdf_score": pvdf_score,
        "pr_score": pr_score,
        "voltage_score": voltage_score,
        "envelope_score": envelope_score,
        "motion_score": motion_score,
        "motion_threshold": float(threshold),
        "kalman_residual_z": kalman_residual_series,
        "kalman_segment_score": kalman_score_series,
        "kalman_threshold": float(cfg.kalman_residual_threshold_z),
        "pr_contact_segment_score": pr_contact_score_series,
        "pr_contact_threshold": float(cfg.pr_contact_threshold_z),
        "verification_score": verification_score,
        "verification_threshold": 1.0,
        "raw_candidate_mask": raw_candidate,
        "pre_dilated_candidate_mask": pre_dilated_candidate,
        "candidate_event_mask": candidate_event,
        "separated_event_mask": separated_event,
        "verified_motion_mask": verified_motion,
        "motion_mask": final_motion,
    }


def segment_row(segment_id: int, start: int, end: int, cfg: GateConfig, segment_type: str) -> dict[str, int | float | str | bool]:
    duration = (end - start) / cfg.fs
    return {
        "segment_id": segment_id,
        "segment_type": segment_type,
        "start_idx": int(start),
        "end_idx": int(end),
        "start_sec": float(start / cfg.fs),
        "end_sec": float(end / cfg.fs),
        "duration_sec": float(duration),
        "usable_segment": bool(segment_type == "clean" and duration >= cfg.min_clean_segment_sec),
    }


def segments_from_mask(mask: np.ndarray, cfg: GateConfig, *, segment_type: str) -> pd.DataFrame:
    rows: list[dict[str, int | float | str | bool]] = []
    for segment_id, (start, end) in enumerate(mask_runs(mask)):
        rows.append(segment_row(segment_id, start, end, cfg, segment_type))
    return pd.DataFrame(rows)


def add_segment_diagnostics(
    segments: pd.DataFrame,
    data: dict[str, np.ndarray],
    gate: dict[str, np.ndarray | float],
    cfg: GateConfig,
) -> pd.DataFrame:
    if segments.empty:
        return segments

    out = segments.copy()
    max_motion_score = []
    mean_motion_score = []
    max_kalman_residual_z = []
    mean_kalman_residual_z = []
    kalman_segment_score = []
    pr_contact_segment_score = []
    verification_score = []

    for row in out.itertuples(index=False):
        start = int(row.start_idx)
        end = int(row.end_idx)
        max_motion_score.append(float(np.nanmax(gate["motion_score"][start:end])))
        mean_motion_score.append(float(np.nanmean(gate["motion_score"][start:end])))
        max_kalman_residual_z.append(float(np.nanmax(gate["kalman_residual_z"][start:end])))
        mean_kalman_residual_z.append(float(np.nanmean(gate["kalman_residual_z"][start:end])))
        kalman_segment_score.append(float(np.nanmax(gate["kalman_segment_score"][start:end])))
        pr_contact_segment_score.append(float(np.nanmax(gate["pr_contact_segment_score"][start:end])))
        verification_score.append(float(np.nanmax(gate["verification_score"][start:end])))

    out["kalman_segment_score"] = kalman_segment_score
    out["pr_contact_segment_score"] = pr_contact_segment_score
    out["verification_score"] = verification_score
    out["max_kalman_residual_z"] = max_kalman_residual_z
    out["mean_kalman_residual_z"] = mean_kalman_residual_z
    out["max_motion_score"] = max_motion_score
    out["mean_motion_score"] = mean_motion_score
    return out


def build_segments(data: dict[str, np.ndarray], gate: dict[str, np.ndarray | float], cfg: GateConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    motion = np.asarray(gate["motion_mask"], dtype=bool)
    valid = np.asarray(data["bad_fraction"], dtype=float) <= cfg.bad_fraction_max
    clean = (~motion) & valid
    motion_segments = add_segment_diagnostics(
        segments_from_mask(motion, cfg, segment_type="motion"),
        data,
        gate,
        cfg,
    )
    clean_segments = add_segment_diagnostics(
        segments_from_mask(clean, cfg, segment_type="clean"),
        data,
        gate,
        cfg,
    )
    return motion_segments, clean_segments


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


def plot_overview(out_path: Path, data: dict[str, np.ndarray], gate: dict[str, np.ndarray | float], cfg: GateConfig) -> None:
    t = data["t"]
    max_points = 40000
    stride = max(1, int(np.ceil(len(t) / max_points)))
    sl = slice(None, None, stride)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    # pvdf_norm = normalized_for_plot(data["pvdf"])
    # pr_norm = normalized_for_plot(data["pr"])
    pvdf_norm = data["pvdf"]
    pr_norm = data["pr"]
    axes[0].plot(t[sl], pvdf_norm[sl], linewidth=0.7, label="PVDF")
    axes[0].plot(t[sl], pr_norm[sl], linewidth=0.7, alpha=0.75, label="PR")
    axes[0].fill_between(t[sl], -6, 6, gate["motion_mask"][sl].astype(float), color="tab:red", alpha=0.15)
    set_robust_ylim(axes[0], np.r_[pvdf_norm[sl], pr_norm[sl]], 0.5, 99.5)
    axes[0].set_ylabel("norm.")
    axes[0].legend(loc="upper right")

    axes[1].plot(t[sl], gate["voltage_score"][sl], linewidth=0.7, label="voltage rate")
    axes[1].plot(t[sl], gate["envelope_score"][sl], linewidth=0.7, label="envelope rate")
    axes[1].plot(t[sl], gate["motion_score"][sl], color="tab:red", linewidth=0.9, label="fused")
    axes[1].axhline(gate["motion_threshold"], color="black", linestyle="--", linewidth=1.0, label="threshold")
    set_robust_ylim(axes[1], gate["motion_score"][sl], 0.0, 99.5)
    axes[1].set_ylabel("score")
    axes[1].legend(loc="upper right")

    axes[2].plot(t[sl], gate["kalman_residual_z"][sl], color="tab:purple", linewidth=0.8, label="Kalman residual")
    axes[2].plot(t[sl], gate["pr_score"][sl], color="tab:orange", linewidth=0.7, alpha=0.8, label="PR contact score")
    axes[2].plot(t[sl], gate["verification_score"][sl], color="tab:green", linewidth=0.9, label="verification score")
    if cfg.kalman_residual_threshold_z > 0:
        axes[2].axhline(cfg.kalman_residual_threshold_z, color="black", linestyle="--", linewidth=1.0, label="Kalman threshold")
    if cfg.pr_contact_threshold_z > 0:
        axes[2].axhline(cfg.pr_contact_threshold_z, color="tab:orange", linestyle="--", linewidth=0.9, alpha=0.8, label="PR threshold")
    axes[2].axhline(1.0, color="tab:green", linestyle="--", linewidth=0.9, alpha=0.8, label="verify threshold")
    set_robust_ylim(axes[2], np.r_[gate["kalman_residual_z"][sl], gate["pr_score"][sl], gate["verification_score"][sl]], 0.0, 99.5)
    axes[2].set_ylabel("confirm z")
    axes[2].legend(loc="upper right")

    mask_lanes = [
        ("candidate", "raw_candidate_mask"),
        ("pre_dilate", "pre_dilated_candidate_mask"),
        ("event", "candidate_event_mask"),
        ("separate", "separated_event_mask"),
        ("verify", "verified_motion_mask"),
        ("final", "motion_mask"),
    ]
    for lane, (label, key) in enumerate(mask_lanes):
        y = np.full_like(t[sl], lane, dtype=float)
        mask = gate[key][sl].astype(float)
        axes[3].fill_between(t[sl], y, y + 0.75 * mask, step="post", alpha=0.55, label=label)
    axes[3].set_yticks(np.arange(len(mask_lanes)) + 0.35)
    axes[3].set_yticklabels([label for label, _ in mask_lanes])
    axes[3].set_ylim(-0.2, len(mask_lanes))
    axes[3].set_ylabel("steps")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(loc="upper right", ncol=6)

    fig.suptitle("Motion segment detection overview", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_detail(
    out_path: Path,
    data: dict[str, np.ndarray],
    gate: dict[str, np.ndarray | float],
    cfg: GateConfig,
    span_sec: float = 80.0,
) -> None:
    t = data["t"]
    if len(t) == 0:
        return

    center = int(np.argmax(gate["motion_score"]))
    half = int(round(0.5 * span_sec * cfg.fs))
    start = max(0, center - half)
    end = min(len(t), center + half)
    sl = slice(start, end)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    pvdf_norm = normalized_for_plot(data["pvdf"][sl])
    pr_norm = normalized_for_plot(data["pr"][sl])
    axes[0].plot(t[sl], pvdf_norm, linewidth=1.0, label="PVDF")
    axes[0].plot(t[sl], pr_norm, linewidth=0.8, alpha=0.8, label="PR")
    axes[0].fill_between(t[sl], -6, 6, gate["motion_mask"][sl].astype(float), color="tab:red", alpha=0.15)
    set_robust_ylim(axes[0], np.r_[pvdf_norm, pr_norm], 0.5, 99.5)
    axes[0].set_ylabel("norm.")
    axes[0].legend(loc="upper right")

    axes[1].plot(t[sl], gate["voltage_score"][sl], linewidth=0.8, label="voltage rate")
    axes[1].plot(t[sl], gate["envelope_score"][sl], linewidth=0.8, label="envelope rate")
    axes[1].plot(t[sl], gate["motion_score"][sl], color="tab:red", linewidth=1.0, label="fused")
    axes[1].axhline(gate["motion_threshold"], color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("score")
    axes[1].legend(loc="upper right")

    axes[2].plot(t[sl], gate["kalman_residual_z"][sl], color="tab:purple", linewidth=0.9, label="Kalman residual")
    axes[2].plot(t[sl], gate["pr_score"][sl], color="tab:orange", linewidth=0.8, alpha=0.8, label="PR contact score")
    axes[2].plot(t[sl], gate["verification_score"][sl], color="tab:green", linewidth=0.9, label="verification score")
    if cfg.kalman_residual_threshold_z > 0:
        axes[2].axhline(cfg.kalman_residual_threshold_z, color="black", linestyle="--", linewidth=1.0)
    if cfg.pr_contact_threshold_z > 0:
        axes[2].axhline(cfg.pr_contact_threshold_z, color="tab:orange", linestyle="--", linewidth=0.9, alpha=0.8)
    axes[2].axhline(1.0, color="tab:green", linestyle="--", linewidth=0.9, alpha=0.8)
    axes[2].set_ylabel("confirm z")
    axes[2].legend(loc="upper right")

    mask_lanes = [
        ("candidate", "raw_candidate_mask"),
        ("pre_dilate", "pre_dilated_candidate_mask"),
        ("event", "candidate_event_mask"),
        ("separate", "separated_event_mask"),
        ("verify", "verified_motion_mask"),
        ("final", "motion_mask"),
    ]
    for lane, (label, key) in enumerate(mask_lanes):
        y = np.full_like(t[sl], lane, dtype=float)
        mask = gate[key][sl].astype(float)
        axes[3].fill_between(t[sl], y, y + 0.75 * mask, step="post", alpha=0.55, label=label)
    axes[3].set_yticks(np.arange(len(mask_lanes)) + 0.35)
    axes[3].set_yticklabels([label for label, _ in mask_lanes])
    axes[3].set_ylim(-0.2, len(mask_lanes))
    axes[3].set_ylabel("steps")
    axes[3].set_xlabel("Time (s)")

    fig.suptitle("Detail around strongest motion score", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_step_table(data: dict[str, np.ndarray], gate: dict[str, np.ndarray | float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_sec": data["t"],
            "motion_score": gate["motion_score"],
            "motion_threshold": np.full_like(data["t"], float(gate["motion_threshold"]), dtype=float),
            "pvdf_score": gate["pvdf_score"],
            "pr_score": gate["pr_score"],
            "voltage_score": gate["voltage_score"],
            "envelope_score": gate["envelope_score"],
            "pvdf_voltage_rate_z": gate["pvdf_voltage_rate_z"],
            "pr_voltage_rate_z": gate["pr_voltage_rate_z"],
            "pvdf_env_rate_z": gate["pvdf_env_rate_z"],
            "pr_env_rate_z": gate["pr_env_rate_z"],
            "kalman_residual_z": gate["kalman_residual_z"],
            "kalman_segment_score": gate["kalman_segment_score"],
            "kalman_threshold": np.full_like(data["t"], float(gate["kalman_threshold"]), dtype=float),
            "pr_contact_segment_score": gate["pr_contact_segment_score"],
            "pr_contact_threshold": np.full_like(data["t"], float(gate["pr_contact_threshold"]), dtype=float),
            "verification_score": gate["verification_score"],
            "verification_threshold": np.full_like(data["t"], float(gate["verification_threshold"]), dtype=float),
            "raw_candidate": np.asarray(gate["raw_candidate_mask"], dtype=np.uint8),
            "after_pre_dilate": np.asarray(gate["pre_dilated_candidate_mask"], dtype=np.uint8),
            "candidate_event": np.asarray(gate["candidate_event_mask"], dtype=np.uint8),
            "after_separate": np.asarray(gate["separated_event_mask"], dtype=np.uint8),
            "after_verify": np.asarray(gate["verified_motion_mask"], dtype=np.uint8),
            "final_motion": np.asarray(gate["motion_mask"], dtype=np.uint8),
        }
    )


def make_summary(
    csv_path: Path,
    data: dict[str, np.ndarray],
    gate: dict[str, np.ndarray | float],
    motion_segments: pd.DataFrame,
    clean_segments: pd.DataFrame,
    cfg: GateConfig,
) -> dict[str, object]:
    duration = len(data["t"]) / cfg.fs if len(data["t"]) else 0.0
    motion_seconds = float(np.sum(gate["motion_mask"]) / cfg.fs)
    clean_seconds = float(clean_segments["duration_sec"].sum()) if not clean_segments.empty else 0.0
    usable_clean = clean_segments[clean_segments["usable_segment"]] if not clean_segments.empty else clean_segments
    return {
        "input_csv": str(csv_path),
        "fs_raw": cfg.fs_raw,
        "fs_processed": cfg.fs,
        "raw_samples": int(data["raw_samples"][0]),
        "processed_samples": int(len(data["t"])),
        "duration_sec": float(duration),
        "motion_threshold": float(gate["motion_threshold"]),
        "kalman_residual_threshold_z": cfg.kalman_residual_threshold_z,
        "pr_contact_threshold_z": cfg.pr_contact_threshold_z,
        "motion_seconds": motion_seconds,
        "motion_ratio": float(motion_seconds / duration) if duration else 0.0,
        "clean_seconds": clean_seconds,
        "clean_ratio": float(clean_seconds / duration) if duration else 0.0,
        "invalid_seconds": float(np.sum(data["bad_fraction"] > cfg.bad_fraction_max) / cfg.fs),
        "motion_segments_total": int(len(motion_segments)),
        "clean_segments_total": int(len(clean_segments)),
        "usable_clean_segments_total": int(len(usable_clean)),
        "config": asdict(cfg),
    }


def write_text_summary(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "Dual-channel PVDF + piezoresistive motion segment summary",
        f"Input: {summary['input_csv']}",
        f"Duration: {summary['duration_sec']:.1f} s",
        f"Processed sampling rate: {summary['fs_processed']:.1f} Hz",
        f"Motion threshold: {summary['motion_threshold']}",
        f"Kalman residual threshold: {summary['kalman_residual_threshold_z']}",
        f"PR contact threshold: {summary['pr_contact_threshold_z']}",
        f"Motion seconds: {summary['motion_seconds']:.1f}",
        f"Motion ratio: {summary['motion_ratio']:.3f}",
        f"Clean seconds: {summary['clean_seconds']:.1f}",
        f"Clean ratio: {summary['clean_ratio']:.3f}",
        f"Motion segments: {summary['motion_segments_total']}",
        f"Clean segments: {summary['clean_segments_total']}",
        f"Usable clean segments: {summary['usable_clean_segments_total']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_csv(csv_path: Path, out_dir: Path, cfg: GateConfig) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_dual_channel_csv(csv_path, cfg)
    gate = compute_motion_gate(data["pvdf"], data["pr"], cfg)
    motion_segments, clean_segments = build_segments(data, gate, cfg)
    step_table = build_step_table(data, gate)

    motion_segments_path = out_dir / "motion_segments.csv"
    clean_segments_path = out_dir / "clean_segments.csv"
    steps_path = out_dir / "motion_steps.csv"
    summary_json_path = out_dir / "summary.json"
    summary_txt_path = out_dir / "summary.txt"
    overview_path = out_dir / "motion_overview.png"
    detail_path = out_dir / "motion_detail.png"

    motion_segments.to_csv(motion_segments_path, index=False, encoding="utf-8-sig")
    clean_segments.to_csv(clean_segments_path, index=False, encoding="utf-8-sig")
    step_table.to_csv(steps_path, index=False, encoding="utf-8-sig")

    summary = make_summary(csv_path, data, gate, motion_segments, clean_segments, cfg)
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_text_summary(summary_txt_path, summary)
    plot_overview(overview_path, data, gate, cfg)
    plot_detail(detail_path, data, gate, cfg)

    summary["outputs"] = {
        "motion_segments_csv": str(motion_segments_path),
        "clean_segments_csv": str(clean_segments_path),
        "motion_steps_csv": str(steps_path),
        "summary_json": str(summary_json_path),
        "summary_txt": str(summary_txt_path),
        "overview_png": str(overview_path),
        "detail_png": str(detail_path),
    }
    return summary


def resolve_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.csv"))
    return [path]


def main() -> None:
    cfg = GateConfig(**USER_CONFIG)

    csv_paths = resolve_inputs(DEFAULT_CSV)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found: {DEFAULT_CSV}")

    summaries = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        child_out = OUTPUT_DIR / csv_path.stem if len(csv_paths) > 1 else OUTPUT_DIR
        print(f"Analyzing {csv_path}")
        summaries.append(analyze_csv(csv_path, child_out, cfg))

    if len(summaries) > 1:
        aggregate_path = OUTPUT_DIR / "batch_summary.json"
        aggregate_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        brief_rows = [
            {
                "input_csv": Path(item["input_csv"]).name,
                "duration_sec": item["duration_sec"],
                "motion_ratio": item["motion_ratio"],
                "clean_ratio": item["clean_ratio"],
                "motion_segments_total": item["motion_segments_total"],
                "clean_segments_total": item["clean_segments_total"],
                "usable_clean_segments_total": item["usable_clean_segments_total"],
            }
            for item in summaries
        ]
        brief_path = OUTPUT_DIR / "batch_summary.csv"
        pd.DataFrame(brief_rows).to_csv(brief_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {aggregate_path}")
        print(f"Wrote {brief_path}")
    else:
        print(json.dumps(summaries[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
