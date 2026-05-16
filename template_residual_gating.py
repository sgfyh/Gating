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

from quality_gating import (
    expand_mask,
    load_dual_channel_csv,
    mask_runs,
    merge_close_motion_segments,
    moving_average,
    normalized_for_plot,
    resolve_inputs,
    set_robust_ylim,
)

DATE_NAME = "2026_0413晚"
CSV_NAME = "20260414_030313"
DEFAULT_CSV = Path(f"F:\\双通道睡眠实验\\{DATE_NAME}\\数据\\{CSV_NAME}.csv")
OUTPUT_DIR = Path(f"outputs\\{DATE_NAME}\\{CSV_NAME}_template_residual")

# 独立实验脚本：局部呼吸模板残差门控。
# 核心思想：体动不是“电压变化大”，而是“PVDF 当前波形不能被局部正常呼吸模板解释”。
# PR 只作为接触/姿态突变证据，不参与呼吸模板建模。
USER_CONFIG = {
    "fs_raw": 500.0,
    "downsample_q": 10,
    "template_context_sec": 45.0, # 用过去多久的PVDF建立局部呼吸模板
    "template_step_sec": 0.5, # 每隔多久更新一次模板预测
    "template_cycles": 4, # 保留为诊断参数，当前模板使用局部谐波拟合
    "min_resp_period_sec": 2.0,
    "max_resp_period_sec": 8.0,
    "default_resp_period_sec": 4.0,
    "resp_band_low_hz": 0.08,
    "resp_band_high_hz": 0.7,
    "harmonics": 2,
    "residual_energy_sec": 1.0, # 残差能量平滑窗口
    "local_background_sec": 60.0, # 残差/PR证据的局部背景窗口
    "residual_candidate_z": 7.0, # 呼吸模板残差候选阈值
    "residual_confirm_z": 10.0, # 事件确认阈值
    "pr_step_candidate_z": 6.0, # PR接触突变候选阈值
    "pr_step_confirm_z": 8.0, # PR接触突变确认阈值
    "template_quality_min": 0.25, # 低于该值说明局部周期估计不可靠，残差证据降级
    "small_event_keep_sec": 1.0, # 很短但残差突出的事件保留
    "motion_merge_gap_sec": 1.0,
    "motion_dilate_sec": 0.3,
    "edge_guard_sec": 8.0,
    "bad_fraction_max": 0.05,
    "min_clean_segment_sec": 10.0,
}


@dataclass
class ResidualConfig:
    fs_raw: float = 500.0
    downsample_q: int = 10
    adc_max: float = 4095.0
    adc_vref: float = 3.3

    template_context_sec: float = 45.0
    template_step_sec: float = 0.5
    template_cycles: int = 4
    min_resp_period_sec: float = 2.0
    max_resp_period_sec: float = 8.0
    default_resp_period_sec: float = 4.0
    resp_band_low_hz: float = 0.08
    resp_band_high_hz: float = 0.7
    harmonics: int = 2
    residual_energy_sec: float = 1.0
    local_background_sec: float = 60.0

    residual_candidate_z: float = 7.0
    residual_confirm_z: float = 10.0
    pr_step_candidate_z: float = 6.0
    pr_step_confirm_z: float = 8.0
    template_quality_min: float = 0.25
    small_event_keep_sec: float = 1.0
    motion_merge_gap_sec: float = 1.0
    motion_dilate_sec: float = 0.3
    edge_guard_sec: float = 8.0
    bad_fraction_max: float = 0.05
    min_clean_segment_sec: float = 10.0

    @property
    def fs(self) -> float:
        return self.fs_raw / self.downsample_q


MOTION_TYPE_TO_CODE = {
    "clean": 0,
    "local_motion": 1,
    "gross_motion": 2,
    "contact_change": 3,
    "uncertain": 4,
}


def abs_rate(x: np.ndarray, fs: float) -> np.ndarray:
    return np.abs(np.diff(x, prepend=x[0])) * fs


def causal_positive_z(x: np.ndarray, cfg: ResidualConfig, window_sec: float | None = None) -> np.ndarray:
    win_sec = cfg.local_background_sec if window_sec is None else window_sec
    win = max(int(round(win_sec * cfg.fs)), 5)
    min_periods = max(3, win // 5)
    values = pd.Series(np.asarray(x, dtype=float))
    med = values.rolling(win, min_periods=min_periods).median().shift(1)
    med = med.bfill().ffill().to_numpy()
    dev = np.abs(values.to_numpy() - med)
    mad = pd.Series(dev).rolling(win, min_periods=min_periods).median().shift(1)
    mad = mad.bfill().ffill().to_numpy()
    scale = 1.4826 * mad + 1e-12
    return np.clip((values.to_numpy() - med) / scale, 0.0, None)


def preprocess_respiration(pvdf: np.ndarray, cfg: ResidualConfig) -> np.ndarray:
    fs = cfg.fs
    nyq = 0.5 * fs
    low = max(cfg.resp_band_low_hz / nyq, 1e-4)
    high = min(cfg.resp_band_high_hz / nyq, 0.99)
    if high <= low:
        trend = moving_average(pvdf, int(round(8.0 * fs)))
        return pvdf - trend
    sos = signal.butter(3, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(pvdf, dtype=float))


def estimate_period_from_past(past: np.ndarray, cfg: ResidualConfig) -> tuple[float, float]:
    fs = cfg.fs
    min_lag = int(round(cfg.min_resp_period_sec * fs))
    max_lag = int(round(cfg.max_resp_period_sec * fs))
    if len(past) < max_lag * 2:
        return cfg.default_resp_period_sec, 0.0

    x = np.asarray(past, dtype=float)
    x = x - np.nanmedian(x)
    scale = np.nanstd(x) + 1e-12
    x = x / scale
    corr = signal.correlate(x, x, mode="full")
    corr = corr[len(corr) // 2 :]
    if corr[0] <= 1e-12:
        return cfg.default_resp_period_sec, 0.0

    max_lag = min(max_lag, len(corr) - 1)
    lags = np.arange(min_lag, max_lag + 1)
    local_corr = corr[lags] / (corr[0] + 1e-12)
    best_idx = int(np.nanargmax(local_corr))
    best_lag = int(lags[best_idx])
    quality = float(local_corr[best_idx])
    return best_lag / fs, quality


def predict_from_template(past: np.ndarray, period_sec: float, block_len: int, cfg: ResidualConfig) -> np.ndarray:
    fs = cfg.fs
    cycle_len = max(int(round(period_sec * fs)), 2)
    n_cycles = min(cfg.template_cycles, len(past) // cycle_len)
    if n_cycles < 2:
        fill = float(past[-1]) if len(past) else 0.0
        return np.full(block_len, fill, dtype=float)

    recent = np.asarray(past[-n_cycles * cycle_len :], dtype=float)
    cycles = recent.reshape(n_cycles, cycle_len)
    template = np.nanmedian(cycles, axis=0)
    phase = np.arange(block_len) % cycle_len
    return template[phase]


def harmonic_design(sample_idx: np.ndarray, period_samples: float, harmonics: int) -> np.ndarray:
    cols = [np.ones_like(sample_idx, dtype=float)]
    phase = 2.0 * np.pi * sample_idx / max(period_samples, 1.0)
    for harmonic in range(1, harmonics + 1):
        cols.append(np.sin(harmonic * phase))
        cols.append(np.cos(harmonic * phase))
    return np.column_stack(cols)


def predict_with_harmonic_template(
    resp: np.ndarray,
    start: int,
    end: int,
    period_sec: float,
    cfg: ResidualConfig,
) -> np.ndarray:
    context = int(round(cfg.template_context_sec * cfg.fs))
    past_start = max(0, start - context)
    if start - past_start < max(20, int(round(2.0 * period_sec * cfg.fs))):
        return predict_from_template(resp[past_start:start], period_sec, end - start, cfg)

    period_samples = max(period_sec * cfg.fs, 1.0)
    past_idx = np.arange(past_start, start, dtype=float)
    future_idx = np.arange(start, end, dtype=float)
    y = np.asarray(resp[past_start:start], dtype=float)
    x = harmonic_design(past_idx, period_samples, max(1, int(cfg.harmonics)))
    try:
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        return predict_from_template(resp[past_start:start], period_sec, end - start, cfg)
    return harmonic_design(future_idx, period_samples, max(1, int(cfg.harmonics))) @ coef


def build_template_prediction(resp: np.ndarray, cfg: ResidualConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(resp)
    prediction = np.zeros(n, dtype=float)
    period_series = np.full(n, cfg.default_resp_period_sec, dtype=float)
    quality_series = np.zeros(n, dtype=float)

    context = int(round(cfg.template_context_sec * cfg.fs))
    step = max(int(round(cfg.template_step_sec * cfg.fs)), 1)
    for start in range(0, n, step):
        end = min(n, start + step)
        past_start = max(0, start - context)
        past = resp[past_start:start]
        period_sec, quality = estimate_period_from_past(past, cfg)
        prediction[start:end] = predict_with_harmonic_template(resp, start, end, period_sec, cfg)
        period_series[start:end] = period_sec
        quality_series[start:end] = quality

    return prediction, period_series, quality_series


def compute_residual_features(pvdf: np.ndarray, pr: np.ndarray, cfg: ResidualConfig) -> dict[str, np.ndarray]:
    fs = cfg.fs
    pvdf_resp = preprocess_respiration(pvdf, cfg)
    prediction, local_period, template_quality = build_template_prediction(pvdf_resp, cfg)
    residual = pvdf_resp - prediction

    residual_energy = moving_average(residual**2, int(round(cfg.residual_energy_sec * fs)))
    resp_energy = moving_average(pvdf_resp**2, int(round(cfg.local_background_sec * fs)))
    normalized_residual_energy = residual_energy / (resp_energy + 1e-12)
    residual_z = causal_positive_z(normalized_residual_energy, cfg)

    pr_smooth = moving_average(pr, int(round(0.5 * fs)))
    pr_step_energy = moving_average(abs_rate(pr_smooth, fs), int(round(0.5 * fs)))
    pr_step_z = causal_positive_z(pr_step_energy, cfg)

    return {
        "pvdf_resp": pvdf_resp,
        "template_prediction": prediction,
        "local_period_sec": local_period,
        "template_quality": template_quality,
        "template_residual": residual,
        "residual_energy": residual_energy,
        "normalized_residual_energy": normalized_residual_energy,
        "residual_z": residual_z,
        "pr_step_z": pr_step_z,
    }


def classify_candidate_events(
    candidate_mask: np.ndarray,
    features: dict[str, np.ndarray],
    cfg: ResidualConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows: list[dict[str, object]] = []
    confirmed = np.zeros_like(candidate_mask, dtype=bool)
    type_code = np.zeros_like(candidate_mask, dtype=int)

    for event_id, (start, end) in enumerate(mask_runs(candidate_mask)):
        duration = (end - start) / cfg.fs
        max_residual_z = float(np.nanmax(features["residual_z"][start:end]))
        max_pr_step_z = float(np.nanmax(features["pr_step_z"][start:end]))
        mean_period = float(np.nanmean(features["local_period_sec"][start:end]))
        mean_quality = float(np.nanmean(features["template_quality"][start:end]))
        max_residual_energy = float(np.nanmax(features["normalized_residual_energy"][start:end]))

        residual_reliable = mean_quality >= cfg.template_quality_min
        residual_confirmed = max_residual_z >= cfg.residual_confirm_z and residual_reliable
        pr_confirmed = max_pr_step_z >= cfg.pr_step_confirm_z
        short_residual_event = duration <= cfg.small_event_keep_sec and max_residual_z >= cfg.residual_candidate_z and residual_reliable

        if residual_confirmed and pr_confirmed:
            motion_type = "gross_motion"
            keep = True
        elif residual_confirmed or short_residual_event:
            motion_type = "local_motion"
            keep = True
        elif pr_confirmed:
            motion_type = "contact_change"
            keep = False
        else:
            motion_type = "uncertain"
            keep = False

        code = MOTION_TYPE_TO_CODE[motion_type]
        type_code[start:end] = code
        if keep:
            confirmed[start:end] = True

        rows.append(
            {
                "event_id": event_id,
                "start_idx": start,
                "end_idx": end,
                "start_sec": start / cfg.fs,
                "end_sec": end / cfg.fs,
                "duration_sec": duration,
                "motion_type": motion_type,
                "final_motion": bool(keep),
                "max_residual_z": max_residual_z,
                "max_pr_step_z": max_pr_step_z,
                "max_normalized_residual_energy": max_residual_energy,
                "mean_local_period_sec": mean_period,
                "mean_template_quality": mean_quality,
                "residual_reliable": bool(residual_reliable),
            }
        )

    return pd.DataFrame(rows), confirmed, type_code


def compute_motion_gate(pvdf: np.ndarray, pr: np.ndarray, cfg: ResidualConfig) -> dict[str, np.ndarray | float | pd.DataFrame]:
    features = compute_residual_features(pvdf, pr, cfg)
    template_reliable = features["template_quality"] >= cfg.template_quality_min
    residual_candidate = (features["residual_z"] >= cfg.residual_candidate_z) & template_reliable
    raw_candidate = residual_candidate | (features["pr_step_z"] >= cfg.pr_step_candidate_z)

    edge = int(round(cfg.edge_guard_sec * cfg.fs))
    if edge > 0 and len(raw_candidate) > 2 * edge:
        raw_candidate[:edge] = False
        raw_candidate[-edge:] = False

    merged_candidate = merge_close_motion_segments(raw_candidate, cfg)
    candidate_events, confirmed, motion_type_code = classify_candidate_events(merged_candidate, features, cfg)
    merged_motion = merge_close_motion_segments(confirmed, cfg)
    final_motion = expand_mask(merged_motion, int(round(cfg.motion_dilate_sec * cfg.fs)))

    return {
        **features,
        "raw_candidate_mask": raw_candidate,
        "template_reliable_mask": template_reliable,
        "residual_candidate_mask": residual_candidate,
        "merged_candidate_mask": merged_candidate,
        "confirmed_motion_mask": confirmed,
        "merged_motion_mask": merged_motion,
        "motion_mask": final_motion,
        "motion_type_code": motion_type_code,
        "motion_threshold": float(cfg.residual_candidate_z),
        "candidate_events": candidate_events,
    }


def segments_from_mask(mask: np.ndarray, cfg: ResidualConfig, segment_type: str) -> pd.DataFrame:
    rows = []
    for segment_id, (start, end) in enumerate(mask_runs(mask)):
        duration = (end - start) / cfg.fs
        rows.append(
            {
                "segment_id": segment_id,
                "segment_type": segment_type,
                "start_idx": start,
                "end_idx": end,
                "start_sec": start / cfg.fs,
                "end_sec": end / cfg.fs,
                "duration_sec": duration,
                "usable_segment": bool(segment_type == "clean" and duration >= cfg.min_clean_segment_sec),
            }
        )
    return pd.DataFrame(rows)


def add_segment_diagnostics(segments: pd.DataFrame, gate: dict[str, np.ndarray | float | pd.DataFrame], cfg: ResidualConfig) -> pd.DataFrame:
    if segments.empty:
        return segments
    out = segments.copy()
    diag_names = ["residual_z", "pr_step_z", "normalized_residual_energy", "local_period_sec", "template_quality"]
    for name in diag_names:
        values = []
        for row in out.itertuples(index=False):
            seg = np.asarray(gate[name])[int(row.start_idx) : int(row.end_idx)]
            if name in {"local_period_sec", "template_quality"}:
                values.append(float(np.nanmean(seg)))
            else:
                values.append(float(np.nanmax(seg)))
        prefix = "mean" if name in {"local_period_sec", "template_quality"} else "max"
        out[f"{prefix}_{name}"] = values

    motion_type = []
    for row in out.itertuples(index=False):
        seg_code = np.asarray(gate["motion_type_code"])[int(row.start_idx) : int(row.end_idx)]
        seg_code = seg_code[seg_code > 0]
        if len(seg_code) == 0:
            motion_type.append("clean")
        else:
            code = int(pd.Series(seg_code).mode().iloc[0])
            motion_type.append(next(k for k, v in MOTION_TYPE_TO_CODE.items() if v == code))
    out["motion_type"] = motion_type
    return out


def build_segments(data: dict[str, np.ndarray], gate: dict[str, np.ndarray | float | pd.DataFrame], cfg: ResidualConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    motion = np.asarray(gate["motion_mask"], dtype=bool)
    valid = np.asarray(data["bad_fraction"], dtype=float) <= cfg.bad_fraction_max
    clean = (~motion) & valid
    motion_segments = add_segment_diagnostics(segments_from_mask(motion, cfg, "motion"), gate, cfg)
    clean_segments = add_segment_diagnostics(segments_from_mask(clean, cfg, "clean"), gate, cfg)
    return motion_segments, clean_segments


def build_step_table(data: dict[str, np.ndarray], gate: dict[str, np.ndarray | float | pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_sec": data["t"],
            "pvdf_resp": gate["pvdf_resp"],
            "template_prediction": gate["template_prediction"],
            "template_residual": gate["template_residual"],
            "residual_energy": gate["residual_energy"],
            "normalized_residual_energy": gate["normalized_residual_energy"],
            "residual_z": gate["residual_z"],
            "pr_step_z": gate["pr_step_z"],
            "local_period_sec": gate["local_period_sec"],
            "template_quality": gate["template_quality"],
            "template_reliable": np.asarray(gate["template_reliable_mask"], dtype=np.uint8),
            "residual_candidate": np.asarray(gate["residual_candidate_mask"], dtype=np.uint8),
            "raw_candidate": np.asarray(gate["raw_candidate_mask"], dtype=np.uint8),
            "after_first_merge": np.asarray(gate["merged_candidate_mask"], dtype=np.uint8),
            "after_event_classification": np.asarray(gate["confirmed_motion_mask"], dtype=np.uint8),
            "after_second_merge": np.asarray(gate["merged_motion_mask"], dtype=np.uint8),
            "final_motion": np.asarray(gate["motion_mask"], dtype=np.uint8),
            "motion_type_code": gate["motion_type_code"],
        }
    )


def plot_result(
    out_path: Path,
    data: dict[str, np.ndarray],
    gate: dict[str, np.ndarray | float | pd.DataFrame],
    cfg: ResidualConfig,
    *,
    detail: bool,
) -> None:
    t = data["t"]
    if len(t) == 0:
        return

    if detail:
        center = int(np.nanargmax(gate["residual_z"]))
        half = int(round(50.0 * cfg.fs))
        sl = slice(max(0, center - half), min(len(t), center + half))
        figsize = (13, 9)
    else:
        stride = max(1, int(np.ceil(len(t) / 45000)))
        sl = slice(None, None, stride)
        figsize = (15, 10)

    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)

    pvdf_plot = normalized_for_plot(data["pvdf"][sl]) if detail else data["pvdf"][sl]
    pr_plot = normalized_for_plot(data["pr"][sl]) if detail else data["pr"][sl]
    axes[0].plot(t[sl], pvdf_plot, linewidth=0.8, label="PVDF")
    axes[0].plot(t[sl], pr_plot, linewidth=0.8, alpha=0.75, label="PR")
    axes[0].fill_between(t[sl], -6, 6, np.asarray(gate["motion_mask"])[sl].astype(float), color="tab:red", alpha=0.15)
    set_robust_ylim(axes[0], np.r_[pvdf_plot, pr_plot], 0.5, 99.5)
    axes[0].set_ylabel("signal")
    axes[0].legend(loc="upper right")

    axes[1].plot(t[sl], gate["pvdf_resp"][sl], linewidth=0.8, label="PVDF respiratory component")
    axes[1].plot(t[sl], gate["template_prediction"][sl], linewidth=0.8, label="template prediction")
    axes[1].plot(t[sl], gate["template_residual"][sl], linewidth=0.6, alpha=0.75, label="residual")
    set_robust_ylim(axes[1], np.r_[gate["pvdf_resp"][sl], gate["template_prediction"][sl], gate["template_residual"][sl]], 0.5, 99.5)
    axes[1].set_ylabel("template")
    axes[1].legend(loc="upper right")

    axes[2].plot(t[sl], gate["residual_z"][sl], linewidth=0.8, label="residual z")
    axes[2].plot(t[sl], gate["pr_step_z"][sl], linewidth=0.8, label="PR step z")
    axes[2].axhline(cfg.residual_candidate_z, color="black", linestyle="--", linewidth=1.0, label="candidate")
    axes[2].axhline(cfg.residual_confirm_z, color="tab:red", linestyle="--", linewidth=1.0, label="confirm")
    set_robust_ylim(axes[2], np.r_[gate["residual_z"][sl], gate["pr_step_z"][sl]], 0.0, 99.0)
    axes[2].set_ylabel("evidence")
    axes[2].legend(loc="upper right")

    axes[3].plot(t[sl], gate["local_period_sec"][sl], linewidth=0.8, label="local period")
    axes[3].plot(t[sl], gate["template_quality"][sl], linewidth=0.8, label="template quality")
    axes[3].axhline(cfg.template_quality_min, color="black", linestyle="--", linewidth=1.0)
    axes[3].set_ylabel("model")
    axes[3].legend(loc="upper right")

    lanes = [
        ("candidate", "raw_candidate_mask"),
        ("merge1", "merged_candidate_mask"),
        ("classify", "confirmed_motion_mask"),
        ("merge2", "merged_motion_mask"),
        ("final", "motion_mask"),
    ]
    for lane, (label, key) in enumerate(lanes):
        y = np.full_like(t[sl], lane, dtype=float)
        axes[4].fill_between(t[sl], y, y + 0.75 * np.asarray(gate[key])[sl].astype(float), step="post", alpha=0.55, label=label)
    axes[4].set_yticks(np.arange(len(lanes)) + 0.35)
    axes[4].set_yticklabels([label for label, _ in lanes])
    axes[4].set_ylim(-0.2, len(lanes))
    axes[4].set_ylabel("steps")
    axes[4].set_xlabel("Time (s)")
    axes[4].legend(loc="upper right", ncol=5)

    fig.suptitle("Local respiratory template residual gating", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def analyze_csv(csv_path: Path, out_dir: Path, cfg: ResidualConfig) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_dual_channel_csv(csv_path, cfg)
    gate = compute_motion_gate(data["pvdf"], data["pr"], cfg)
    candidate_events = gate["candidate_events"]
    motion_segments, clean_segments = build_segments(data, gate, cfg)
    step_table = build_step_table(data, gate)

    candidate_events.to_csv(out_dir / "candidate_events.csv", index=False, encoding="utf-8-sig")
    motion_segments.to_csv(out_dir / "motion_segments.csv", index=False, encoding="utf-8-sig")
    clean_segments.to_csv(out_dir / "clean_segments.csv", index=False, encoding="utf-8-sig")
    step_table.to_csv(out_dir / "motion_steps.csv", index=False, encoding="utf-8-sig")
    plot_result(out_dir / "motion_overview.png", data, gate, cfg, detail=False)
    plot_result(out_dir / "motion_detail.png", data, gate, cfg, detail=True)

    duration = len(data["t"]) / cfg.fs if len(data["t"]) else 0.0
    motion_seconds = float(np.sum(gate["motion_mask"]) / cfg.fs)
    type_counts = candidate_events["motion_type"].value_counts().to_dict() if not candidate_events.empty else {}
    summary = {
        "input_csv": str(csv_path),
        "duration_sec": float(duration),
        "fs_processed": cfg.fs,
        "motion_seconds": motion_seconds,
        "motion_ratio": float(motion_seconds / duration) if duration else 0.0,
        "clean_seconds": float(clean_segments["duration_sec"].sum()) if not clean_segments.empty else 0.0,
        "motion_segments_total": int(len(motion_segments)),
        "clean_segments_total": int(len(clean_segments)),
        "usable_clean_segments_total": int(clean_segments["usable_segment"].sum()) if not clean_segments.empty else 0,
        "candidate_events_total": int(len(candidate_events)),
        "candidate_event_type_counts": type_counts,
        "config": asdict(cfg),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.txt").write_text(
        "\n".join(
            [
                "Local respiratory template residual gating summary",
                f"Input: {summary['input_csv']}",
                f"Duration: {summary['duration_sec']:.1f} s",
                f"Motion seconds: {summary['motion_seconds']:.1f}",
                f"Motion ratio: {summary['motion_ratio']:.3f}",
                f"Candidate events: {summary['candidate_events_total']}",
                f"Event types: {summary['candidate_event_type_counts']}",
                f"Motion segments: {summary['motion_segments_total']}",
                f"Clean segments: {summary['clean_segments_total']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary["outputs"] = {
        "candidate_events_csv": str(out_dir / "candidate_events.csv"),
        "motion_segments_csv": str(out_dir / "motion_segments.csv"),
        "clean_segments_csv": str(out_dir / "clean_segments.csv"),
        "motion_steps_csv": str(out_dir / "motion_steps.csv"),
        "overview_png": str(out_dir / "motion_overview.png"),
        "detail_png": str(out_dir / "motion_detail.png"),
    }
    return summary


def main() -> None:
    cfg = ResidualConfig(**USER_CONFIG)
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
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "batch_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(summaries[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
