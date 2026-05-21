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

# DATE_NAME = "椅子"
# CSV_NAME = "WAVE(2026.4.23-16.55.42)"
# DEFAULT_CSV = Path(f"C:\\Users\\sgfyh\\Desktop\\{CSV_NAME}.csv")
DATE_NAME = "2026_0423晚"
CSV_NAME = "20260424_051233"
DEFAULT_CSV = Path(f"F:\\双通道睡眠实验\\{DATE_NAME}\\数据\\{CSV_NAME}.csv")
OUTPUT_DIR = Path(f"outputs\\{DATE_NAME}\\{CSV_NAME}_analysis")

# 直接在这里改参数；运行 `python .\quality_gating.py` 即可。
USER_CONFIG = {
    "fs_raw": 500.0, # 原始采样率，单位Hz
    "downsample_q": 10, # 下采样倍数，越大计算越快，但可能丢失短暂运动段
    "motion_threshold_z": 5.0, # 运动得分阈值，单位为 robust z-score，越大越严格
    "pre_motion_dilate_sec": 0.5, # 候选体动先向两边扩张，单位秒
    "motion_merge_gap_sec": 10.0, # 合并相隔不超过这个时间的候选事件，单位秒
    "event_split_min_sec": 20.0, # 只分离超过该时长的长候选事件，单位秒
    "event_split_gap_sec": 2.0, # 长候选事件内部间隔超过该值则切开，单位秒
    "motion_min_duration_sec": 1.0, # 小于该时长的分离核心视为短毛刺，放回clean
    "motion_dilate_sec": 1.0, # 最终体动段边界扩张，单位秒
    "pvdf_weight": 0.0, # PVDF信号权重
    "voltage_rate_weight": 0.5, # 电压变化率权重
    "envelope_rate_weight": 0.5, # 包络变化率权重
    "min_clean_segment_sec": 10.0, # 最小清洁段长度，单位秒
}


@dataclass
class GateConfig:
    fs_raw: float = 500.0
    downsample_q: int = 10
    adc_max: float = 4095.0
    adc_vref: float = 3.3

    motion_threshold_z: float = 5.0
    pre_motion_dilate_sec: float = 0.5
    motion_merge_gap_sec: float = 10.0
    event_split_min_sec: float = 15.0
    event_split_gap_sec: float = 2.0
    motion_min_duration_sec: float = 1.0
    motion_dilate_sec: float = 1.0
    edge_guard_sec: float = 3.0

    pvdf_weight: float = 0.0
    voltage_rate_weight: float = 0.5
    envelope_rate_weight: float = 0.5

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
        "env_level_z": robust_positive_z(env),
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


def reliable_otsu_core(
    local_score: np.ndarray,
    cfg: GateConfig,
    *,
    min_threshold: float | None = None,
    max_high_ratio: float = 0.60,
    upper_threshold: float | None = None,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float]:
    score = np.asarray(local_score, dtype=float)
    stage_mask = np.isfinite(score)
    if upper_threshold is not None:
        stage_mask &= score < upper_threshold
    if valid_mask is not None:
        stage_mask &= np.asarray(valid_mask, dtype=bool)

    stage_values = score[stage_mask]
    threshold = otsu_threshold(stage_values)
    threshold_floor = float(cfg.motion_threshold_z) if min_threshold is None else float(min_threshold)
    if threshold is None or threshold <= threshold_floor:
        return None, np.nan

    high = stage_values[stage_values >= threshold]
    low = stage_values[stage_values < threshold]
    if high.size == 0 or low.size == 0:
        return None, threshold

    high_ratio = high.size / stage_values.size
    if high_ratio < 0.005 or high_ratio > max_high_ratio:
        return None, threshold

    _, scale = robust_scale(stage_values)
    separation = (float(np.nanmean(high)) - float(np.nanmean(low))) / max(scale, 1e-12)
    if separation < 1.5:
        return None, threshold

    core = stage_mask & (score >= threshold)
    return core, threshold


def otsu_union_pass(
    split_scores: tuple[np.ndarray, ...],
    cfg: GateConfig,
    valid_mask: np.ndarray,
    *,
    min_threshold: float | None = None,
    max_high_ratio: float = 0.60,
) -> tuple[np.ndarray | None, list[float]]:
    scores = tuple(np.asarray(score, dtype=float) for score in split_scores)
    if not scores:
        return None, []

    cores = []
    thresholds = []
    for score in scores:
        core, threshold = reliable_otsu_core(
            score,
            cfg,
            valid_mask=valid_mask,
            min_threshold=min_threshold,
            max_high_ratio=max_high_ratio,
        )
        if core is not None and np.any(core):
            cores.append(core)
            thresholds.append(threshold)

    if not cores:
        return None, thresholds
    return np.logical_or.reduce(cores), thresholds


def two_stage_otsu_event_core(split_scores: tuple[np.ndarray, ...], cfg: GateConfig) -> tuple[np.ndarray | None, float]:
    scores = tuple(np.asarray(score, dtype=float) for score in split_scores)
    if not scores:
        return None, np.nan

    valid_mask = np.logical_or.reduce([np.isfinite(score) for score in scores])

    first_core, first_thresholds = otsu_union_pass(scores, cfg, valid_mask)
    if first_core is None or not np.any(first_core):
        return None, np.nan

    remaining = valid_mask & ~first_core
    second_core, second_thresholds = otsu_union_pass(
        scores,
        cfg,
        remaining,
        min_threshold=0.0,
        max_high_ratio=0.20,
    )

    if second_core is not None and np.any(second_core):
        core = first_core | second_core
        thresholds = first_thresholds + second_thresholds
    else:
        core = first_core
        thresholds = first_thresholds

    finite_thresholds = [float(v) for v in thresholds if np.isfinite(v)]
    return core, float(np.nanmin(finite_thresholds)) if finite_thresholds else np.nan


def separate_long_candidate_events(
    candidate_event: np.ndarray,
    split_scores: tuple[np.ndarray, ...],
    cfg: GateConfig,
) -> tuple[np.ndarray, np.ndarray]:
    separated = np.zeros_like(candidate_event, dtype=bool)
    threshold_series = np.full(candidate_event.shape, np.nan, dtype=float)
    min_len = int(round(cfg.event_split_min_sec * cfg.fs))
    split_gap = int(round(cfg.event_split_gap_sec * cfg.fs))
    pad = int(round(cfg.pre_motion_dilate_sec * cfg.fs))

    for start, end in mask_runs(candidate_event):
        if end - start <= min_len:
            separated[start:end] = True
            continue

        local_scores = tuple(score[start:end] for score in split_scores)

        local_core, local_threshold = two_stage_otsu_event_core(
            local_scores,
            cfg,
        )
        if local_core is None or not np.any(local_core):
            # Long candidates that cannot be split reliably are returned to clean.
            continue

        if np.isfinite(local_threshold):
            threshold_series[start:end] = local_threshold

        local_core = expand_mask(local_core, pad)
        local = merge_runs_by_gap(local_core, split_gap)
        separated[start:end] = local & candidate_event[start:end]

    return separated, threshold_series


def segment_percentile_score(mask: np.ndarray, score: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    series = np.zeros_like(score, dtype=float)
    for start, end in mask_runs(mask):
        series[start:end] = float(np.nanpercentile(score[start:end], percentile))
    return series


def remove_short_motion_cores(mask: np.ndarray, cfg: GateConfig) -> np.ndarray:
    if cfg.motion_min_duration_sec <= 0:
        return mask.astype(bool)

    min_len = int(round(cfg.motion_min_duration_sec * cfg.fs))
    filtered = np.zeros_like(mask, dtype=bool)
    for start, end in mask_runs(mask):
        if end - start >= min_len:
            filtered[start:end] = True
    return filtered


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
    candidate_score = moving_average(pr_score, int(round(1.0 * cfg.fs)))
    split_pvdf_impulse_score = moving_average(pvdf_motion["voltage_rate_z"], int(round(1.0 * cfg.fs)))
    split_score = split_pvdf_impulse_score.copy()

    edge = int(round(cfg.edge_guard_sec * cfg.fs))
    if edge > 0 and len(motion_score) > 2 * edge:
        for arr in (
            pvdf_score,
            pr_score,
            voltage_score,
            envelope_score,
            motion_score,
            candidate_score,
            split_pvdf_impulse_score,
            split_score,
        ):
            arr[:edge] = 0.0
            arr[-edge:] = 0.0

    threshold = max(0.0, float(cfg.motion_threshold_z))
    raw_candidate = candidate_score > threshold
    pre_dilated_candidate = expand_mask(raw_candidate, int(round(cfg.pre_motion_dilate_sec * cfg.fs)))
    candidate_event = merge_close_motion_segments(pre_dilated_candidate, cfg)
    separated_event, split_threshold = separate_long_candidate_events(
        candidate_event,
        (split_pvdf_impulse_score,),
        cfg,
    )
    verified_motion = remove_short_motion_cores(separated_event, cfg)
    event_contact_score = segment_percentile_score(verified_motion, pr_score)
    event_split_score = segment_percentile_score(verified_motion, split_score)
    final_motion = expand_mask(verified_motion, int(round(cfg.motion_dilate_sec * cfg.fs)))

    return {
        "pvdf_voltage_rate_z": pvdf_motion["voltage_rate_z"],
        "pr_voltage_rate_z": pr_motion["voltage_rate_z"],
        "pvdf_env_rate_z": pvdf_motion["env_rate_z"],
        "pr_env_level_z": pr_motion["env_level_z"],
        "pr_env_rate_z": pr_motion["env_rate_z"],
        "pvdf_score": pvdf_score,
        "pr_score": pr_score,
        "voltage_score": voltage_score,
        "envelope_score": envelope_score,
        "motion_score": motion_score,
        "candidate_score": candidate_score,
        "split_pvdf_impulse_score": split_pvdf_impulse_score,
        "split_score": split_score,
        "motion_threshold": float(threshold),
        "split_threshold": split_threshold,
        "event_contact_score": event_contact_score,
        "event_split_score": event_split_score,
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
    max_candidate_score = []
    mean_candidate_score = []
    max_motion_score = []
    mean_motion_score = []
    event_contact_score = []

    for row in out.itertuples(index=False):
        start = int(row.start_idx)
        end = int(row.end_idx)
        max_candidate_score.append(float(np.nanmax(gate["candidate_score"][start:end])))
        mean_candidate_score.append(float(np.nanmean(gate["candidate_score"][start:end])))
        max_motion_score.append(float(np.nanmax(gate["motion_score"][start:end])))
        mean_motion_score.append(float(np.nanmean(gate["motion_score"][start:end])))
        event_contact_score.append(float(np.nanmax(gate["event_contact_score"][start:end])))

    out["event_contact_score"] = event_contact_score
    out["max_candidate_score"] = max_candidate_score
    out["mean_candidate_score"] = mean_candidate_score
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

    axes[1].plot(t[sl], gate["candidate_score"][sl], color="tab:orange", linewidth=0.9, label="PR candidate score")
    axes[1].plot(t[sl], gate["pvdf_score"][sl], color="tab:blue", linewidth=0.7, alpha=0.65, label="PVDF reference score")
    axes[1].plot(t[sl], gate["motion_score"][sl], color="tab:red", linewidth=0.7, alpha=0.55, label="fused reference")
    axes[1].axhline(gate["motion_threshold"], color="black", linestyle="--", linewidth=1.0, label="candidate threshold")
    score_plot = np.r_[gate["candidate_score"][sl], gate["pvdf_score"][sl], gate["motion_score"][sl]]
    set_robust_ylim(axes[1], score_plot, 0.0, 99.5)
    axes[1].set_ylabel("score")
    axes[1].legend(loc="upper right")

    axes[2].plot(t[sl], gate["split_pvdf_impulse_score"][sl], color="tab:blue", linewidth=0.8, alpha=0.8, label="PVDF impulse split score")
    axes[2].plot(t[sl], gate["event_split_score"][sl], color="tab:green", linewidth=0.8, alpha=0.8, label="event P95 split score")
    axes[2].plot(t[sl], gate["split_threshold"][sl], color="tab:purple", linewidth=0.9, alpha=0.8, label="Otsu split threshold")
    split_plot = np.r_[gate["split_score"][sl], gate["event_split_score"][sl], gate["split_threshold"][sl]]
    set_robust_ylim(axes[2], split_plot, 0.0, 99.5)
    axes[2].set_ylabel("split score")
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

    center = int(np.argmax(gate["candidate_score"]))
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

    axes[1].plot(t[sl], gate["candidate_score"][sl], color="tab:orange", linewidth=1.0, label="PR candidate score")
    axes[1].plot(t[sl], gate["pvdf_score"][sl], color="tab:blue", linewidth=0.8, alpha=0.65, label="PVDF reference score")
    axes[1].plot(t[sl], gate["motion_score"][sl], color="tab:red", linewidth=0.8, alpha=0.55, label="fused reference")
    axes[1].axhline(gate["motion_threshold"], color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("score")
    axes[1].legend(loc="upper right")

    axes[2].plot(t[sl], gate["split_pvdf_impulse_score"][sl], color="tab:blue", linewidth=0.9, alpha=0.8, label="PVDF impulse split score")
    axes[2].plot(t[sl], gate["event_split_score"][sl], color="tab:green", linewidth=0.8, alpha=0.8, label="event P95 split score")
    axes[2].plot(t[sl], gate["split_threshold"][sl], color="tab:purple", linewidth=0.9, alpha=0.8, label="Otsu split threshold")
    axes[2].set_ylabel("split score")
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

    fig.suptitle("Detail around strongest PR candidate score", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_step_table(data: dict[str, np.ndarray], gate: dict[str, np.ndarray | float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_sec": data["t"],
            "motion_score": gate["motion_score"],
            "candidate_score": gate["candidate_score"],
            "split_pvdf_impulse_score": gate["split_pvdf_impulse_score"],
            "split_score": gate["split_score"],
            "motion_threshold": np.full_like(data["t"], float(gate["motion_threshold"]), dtype=float),
            "pvdf_score": gate["pvdf_score"],
            "pr_score": gate["pr_score"],
            "voltage_score": gate["voltage_score"],
            "envelope_score": gate["envelope_score"],
            "pvdf_voltage_rate_z": gate["pvdf_voltage_rate_z"],
            "pr_voltage_rate_z": gate["pr_voltage_rate_z"],
            "pvdf_env_rate_z": gate["pvdf_env_rate_z"],
            "pr_env_level_z": gate["pr_env_level_z"],
            "pr_env_rate_z": gate["pr_env_rate_z"],
            "split_threshold": gate["split_threshold"],
            "event_contact_score": gate["event_contact_score"],
            "event_split_score": gate["event_split_score"],
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
        "split_method": "two_pass_otsu_pvdf_impulse_in_pr_candidates",
        "motion_min_duration_sec": float(cfg.motion_min_duration_sec),
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
        f"Split method: {summary['split_method']}",
        f"Motion min duration: {summary['motion_min_duration_sec']} s",
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
