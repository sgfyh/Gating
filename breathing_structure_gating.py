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
    abs_rate,
    expand_mask,
    load_dual_channel_csv,
    mask_runs,
    merge_close_motion_segments,
    moving_average,
    normalized_for_plot,
    resolve_inputs,
    robust_scale,
    set_robust_ylim,
)

DATE_NAME = "2026_0413晚"
CSV_NAME = "20260414_090314"
DEFAULT_CSV = Path(f"F:\\双通道睡眠实验\\{DATE_NAME}\\数据\\{CSV_NAME}.csv")
OUTPUT_DIR = Path(f"outputs\\{DATE_NAME}\\{CSV_NAME}_breathing_structure")

# 独立实验脚本：呼吸结构保护型体动检测。
# 核心可调参数只有几个：candidate_score_z、event_confirm_z、breathing_coherence_min、
# stable_breathing_max_disorder_z。其余多是窗口长度和边界处理。
USER_CONFIG = {
    "fs_raw": 500.0,
    "downsample_q": 10,
    "candidate_score_z": 3.2, # 局部异常候选阈值，越大越严格
    "event_confirm_z": 3.2, # 事件级确认阈值，越大越少报
    "breathing_coherence_min": 0.35, # 呼吸频带集中度，高于它更像稳定呼吸
    "stable_breathing_max_disorder_z": 2.5, # 稳定呼吸允许的最大杂乱度
    "stable_breathing_max_pr_rate_z": 2.5, # 稳定呼吸允许的最大压阻变化率
    "stable_breathing_max_pvdf_rate_z": 4.0, # 稳定呼吸允许的最大PVDF变化率
    "local_background_sec": 60.0, # 局部背景窗口，适应体位/接触造成的幅度漂移
    "breathing_window_sec": 8.0, # 频域呼吸结构分析窗口
    "score_smooth_sec": 0.3,
    "pre_motion_dilate_sec": 0.5,
    "motion_merge_gap_sec": 2.0,
    "motion_dilate_sec": 0.5,
    "small_event_keep_sec": 2.0,
    "bad_fraction_max": 0.05,
    "min_clean_segment_sec": 10.0,
}


@dataclass
class StructureConfig:
    fs_raw: float = 500.0
    downsample_q: int = 10
    adc_max: float = 4095.0
    adc_vref: float = 3.3

    candidate_score_z: float = 3.2
    event_confirm_z: float = 3.2
    breathing_coherence_min: float = 0.35
    stable_breathing_max_disorder_z: float = 2.5
    stable_breathing_max_pr_rate_z: float = 2.5
    stable_breathing_max_pvdf_rate_z: float = 4.0

    local_background_sec: float = 60.0
    breathing_window_sec: float = 8.0
    score_smooth_sec: float = 0.3
    pre_motion_dilate_sec: float = 0.5
    motion_merge_gap_sec: float = 2.0
    motion_dilate_sec: float = 0.5
    edge_guard_sec: float = 3.0
    small_event_keep_sec: float = 2.0
    bad_fraction_max: float = 0.05
    min_clean_segment_sec: float = 10.0

    @property
    def fs(self) -> float:
        return self.fs_raw / self.downsample_q


def rolling_positive_z(x: np.ndarray, cfg: StructureConfig, window_sec: float | None = None) -> np.ndarray:
    win_sec = cfg.local_background_sec if window_sec is None else window_sec
    win = max(int(round(win_sec * cfg.fs)), 5)
    min_periods = max(3, win // 5)
    values = np.asarray(x, dtype=float)
    med = pd.Series(values).rolling(win, center=True, min_periods=min_periods).median()
    med = med.bfill().ffill().to_numpy()
    mad = pd.Series(np.abs(values - med)).rolling(win, center=True, min_periods=min_periods).median()
    mad = mad.bfill().ffill().to_numpy()
    scale = 1.4826 * mad + 1e-12
    return np.clip((values - med) / scale, 0.0, None)


def spectral_structure(x: np.ndarray, cfg: StructureConfig) -> dict[str, np.ndarray]:
    fs = cfg.fs
    n = len(x)
    fallback = {
        "breathing_coherence": np.zeros(n, dtype=float),
        "spectral_entropy": np.ones(n, dtype=float),
        "high_freq_ratio": np.zeros(n, dtype=float),
    }
    nperseg = max(int(round(cfg.breathing_window_sec * fs)), 32)
    if n < nperseg:
        return fallback

    noverlap = min(nperseg - 1, int(round((cfg.breathing_window_sec - 1.0) * fs)))
    freqs, times, psd = signal.spectrogram(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        mode="psd",
    )
    all_band = (freqs >= 0.05) & (freqs <= 2.5)
    resp_band = (freqs >= 0.08) & (freqs <= 0.7)
    high_band = (freqs > 0.7) & (freqs <= 2.5)
    if not np.any(all_band) or not np.any(resp_band):
        return fallback

    band_psd = psd[all_band, :]
    total = np.sum(band_psd, axis=0) + 1e-12
    resp_psd = psd[resp_band, :]
    high_power = np.sum(psd[high_band, :], axis=0) if np.any(high_band) else np.zeros_like(total)
    coherence = np.max(resp_psd, axis=0) / total
    high_ratio = high_power / total
    p = band_psd / total
    entropy = -np.sum(p * np.log(p + 1e-12), axis=0) / np.log(max(2, band_psd.shape[0]))

    sample_t = np.arange(n, dtype=float) / fs
    return {
        "breathing_coherence": np.interp(sample_t, times, coherence, left=coherence[0], right=coherence[-1]),
        "spectral_entropy": np.interp(sample_t, times, entropy, left=entropy[0], right=entropy[-1]),
        "high_freq_ratio": np.interp(sample_t, times, high_ratio, left=high_ratio[0], right=high_ratio[-1]),
    }


def compute_structure_features(pvdf: np.ndarray, pr: np.ndarray, cfg: StructureConfig) -> dict[str, np.ndarray]:
    fs = cfg.fs
    pvdf_smooth = moving_average(pvdf, int(round(0.2 * fs)))
    pr_smooth = moving_average(pr, int(round(0.5 * fs)))
    trend = moving_average(pvdf, int(round(8.0 * fs)))
    pvdf_ac = pvdf - trend
    envelope = moving_average(np.abs(pvdf_ac), int(round(1.0 * fs)))
    pvdf_fast = pvdf - moving_average(pvdf, int(round(0.8 * fs)))

    pvdf_rate_z = rolling_positive_z(abs_rate(pvdf_smooth, fs), cfg)
    pr_rate_z = rolling_positive_z(abs_rate(pr_smooth, fs), cfg)
    env_rate_z = rolling_positive_z(abs_rate(envelope, fs), cfg)
    roughness_z = rolling_positive_z(moving_average(pvdf_fast**2, int(round(0.5 * fs))), cfg)

    spectral = spectral_structure(pvdf_ac, cfg)
    disorder_raw = spectral["spectral_entropy"] + spectral["high_freq_ratio"] + (1.0 - spectral["breathing_coherence"])
    disorder_z = rolling_positive_z(disorder_raw, cfg)

    return {
        "pvdf_rate_z": pvdf_rate_z,
        "pr_rate_z": pr_rate_z,
        "env_rate_z": env_rate_z,
        "roughness_z": roughness_z,
        "disorder_z": disorder_z,
        **spectral,
    }


def confirm_events(candidate_mask: np.ndarray, features: dict[str, np.ndarray], cfg: StructureConfig) -> tuple[np.ndarray, np.ndarray]:
    confirmed = np.zeros_like(candidate_mask, dtype=bool)
    event_evidence = np.zeros_like(features["motion_score"], dtype=float)

    for start, end in mask_runs(candidate_mask):
        duration_sec = (end - start) / cfg.fs
        max_motion = float(np.nanmax(features["motion_score"][start:end]))
        max_disorder = float(np.nanmax(features["disorder_score"][start:end]))
        max_roughness = float(np.nanmax(features["roughness_z"][start:end]))
        max_pr_rate = float(np.nanmax(features["pr_rate_z"][start:end]))
        max_pvdf_rate = float(np.nanmax(features["pvdf_rate_z"][start:end]))
        mean_coherence = float(np.nanmean(features["breathing_coherence"][start:end]))

        evidence = max(max_motion, max_disorder, max_roughness, max_pr_rate)
        event_evidence[start:end] = evidence
        stable_breathing = (
            mean_coherence >= cfg.breathing_coherence_min
            and max_disorder < cfg.stable_breathing_max_disorder_z
            and max_pr_rate < cfg.stable_breathing_max_pr_rate_z
            and max_pvdf_rate < cfg.stable_breathing_max_pvdf_rate_z
        )
        if duration_sec <= cfg.small_event_keep_sec or (evidence >= cfg.event_confirm_z and not stable_breathing):
            confirmed[start:end] = True

    return confirmed, event_evidence


def compute_motion_gate(pvdf: np.ndarray, pr: np.ndarray, cfg: StructureConfig) -> dict[str, np.ndarray | float]:
    features = compute_structure_features(pvdf, pr, cfg)
    disorder_score = 0.5 * features["roughness_z"] + 0.5 * features["disorder_z"]
    motion_score = (
        0.35 * features["pvdf_rate_z"]
        + 0.20 * features["env_rate_z"]
        + 0.20 * features["roughness_z"]
        + 0.15 * features["pr_rate_z"]
        + 0.10 * features["disorder_z"]
    )
    motion_score = moving_average(motion_score, int(round(cfg.score_smooth_sec * cfg.fs)))
    features["motion_score"] = motion_score
    features["disorder_score"] = disorder_score

    edge = int(round(cfg.edge_guard_sec * cfg.fs))
    if edge > 0 and len(motion_score) > 2 * edge:
        for key in ["pvdf_rate_z", "pr_rate_z", "env_rate_z", "roughness_z", "disorder_z", "disorder_score", "motion_score"]:
            features[key][:edge] = 0.0
            features[key][-edge:] = 0.0

    raw_candidate = motion_score > cfg.candidate_score_z
    stable_breathing = (
        (features["breathing_coherence"] >= cfg.breathing_coherence_min)
        & (features["disorder_score"] < cfg.stable_breathing_max_disorder_z)
        & (features["pr_rate_z"] < cfg.stable_breathing_max_pr_rate_z)
        & (features["pvdf_rate_z"] < cfg.stable_breathing_max_pvdf_rate_z)
    )
    protected_candidate = raw_candidate & ~stable_breathing
    pre_dilate = expand_mask(protected_candidate, int(round(cfg.pre_motion_dilate_sec * cfg.fs)))
    first_merge = merge_close_motion_segments(pre_dilate, cfg)
    confirmed, event_evidence = confirm_events(first_merge, features, cfg)
    second_merge = merge_close_motion_segments(confirmed, cfg)
    final_motion = expand_mask(second_merge, int(round(cfg.motion_dilate_sec * cfg.fs)))

    return {
        **features,
        "voltage_score": 0.7 * features["pvdf_rate_z"] + 0.3 * features["pr_rate_z"],
        "envelope_score": features["env_rate_z"],
        "motion_score": motion_score,
        "motion_threshold": float(cfg.candidate_score_z),
        "event_evidence": event_evidence,
        "raw_candidate_mask": raw_candidate,
        "stable_breathing_mask": stable_breathing,
        "protected_candidate_mask": protected_candidate,
        "pre_dilated_candidate_mask": pre_dilate,
        "merged_candidate_mask": first_merge,
        "confirmed_motion_mask": confirmed,
        "merged_motion_mask": second_merge,
        "motion_mask": final_motion,
    }


def segments_from_mask(mask: np.ndarray, cfg: StructureConfig, segment_type: str) -> pd.DataFrame:
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


def add_diagnostics(segments: pd.DataFrame, gate: dict[str, np.ndarray | float]) -> pd.DataFrame:
    if segments.empty:
        return segments
    out = segments.copy()
    for name in ["motion_score", "event_evidence", "disorder_score", "breathing_coherence", "pvdf_rate_z", "pr_rate_z"]:
        values = []
        for row in out.itertuples(index=False):
            seg = np.asarray(gate[name])[int(row.start_idx) : int(row.end_idx)]
            values.append(float(np.nanmean(seg)) if name == "breathing_coherence" else float(np.nanmax(seg)))
        out[f"{'mean' if name == 'breathing_coherence' else 'max'}_{name}"] = values
    return out


def build_segments(data: dict[str, np.ndarray], gate: dict[str, np.ndarray | float], cfg: StructureConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    motion = np.asarray(gate["motion_mask"], dtype=bool)
    valid = np.asarray(data["bad_fraction"], dtype=float) <= cfg.bad_fraction_max
    clean = (~motion) & valid
    return (
        add_diagnostics(segments_from_mask(motion, cfg, "motion"), gate),
        add_diagnostics(segments_from_mask(clean, cfg, "clean"), gate),
    )


def build_step_table(data: dict[str, np.ndarray], gate: dict[str, np.ndarray | float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_sec": data["t"],
            "motion_score": gate["motion_score"],
            "candidate_threshold": np.full_like(data["t"], float(gate["motion_threshold"]), dtype=float),
            "event_evidence": gate["event_evidence"],
            "pvdf_rate_z": gate["pvdf_rate_z"],
            "pr_rate_z": gate["pr_rate_z"],
            "env_rate_z": gate["env_rate_z"],
            "roughness_z": gate["roughness_z"],
            "disorder_score": gate["disorder_score"],
            "breathing_coherence": gate["breathing_coherence"],
            "spectral_entropy": gate["spectral_entropy"],
            "high_freq_ratio": gate["high_freq_ratio"],
            "raw_candidate": np.asarray(gate["raw_candidate_mask"], dtype=np.uint8),
            "stable_breathing": np.asarray(gate["stable_breathing_mask"], dtype=np.uint8),
            "after_structure_protection": np.asarray(gate["protected_candidate_mask"], dtype=np.uint8),
            "after_pre_dilate": np.asarray(gate["pre_dilated_candidate_mask"], dtype=np.uint8),
            "after_first_merge": np.asarray(gate["merged_candidate_mask"], dtype=np.uint8),
            "after_event_confirm": np.asarray(gate["confirmed_motion_mask"], dtype=np.uint8),
            "after_second_merge": np.asarray(gate["merged_motion_mask"], dtype=np.uint8),
            "final_motion": np.asarray(gate["motion_mask"], dtype=np.uint8),
        }
    )


def plot_result(out_path: Path, data: dict[str, np.ndarray], gate: dict[str, np.ndarray | float], cfg: StructureConfig, *, detail: bool) -> None:
    t = data["t"]
    if len(t) == 0:
        return
    if detail:
        center = int(np.argmax(gate["motion_score"]))
        half = int(round(40.0 * cfg.fs))
        sl = slice(max(0, center - half), min(len(t), center + half))
        figsize = (12, 8)
    else:
        stride = max(1, int(np.ceil(len(t) / 40000)))
        sl = slice(None, None, stride)
        figsize = (14, 9)

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    pvdf_plot = normalized_for_plot(data["pvdf"][sl]) if detail else data["pvdf"][sl]
    pr_plot = normalized_for_plot(data["pr"][sl]) if detail else data["pr"][sl]
    axes[0].plot(t[sl], pvdf_plot, linewidth=0.8, label="PVDF")
    axes[0].plot(t[sl], pr_plot, linewidth=0.8, alpha=0.75, label="PR")
    axes[0].fill_between(t[sl], -6, 6, np.asarray(gate["motion_mask"])[sl].astype(float), color="tab:red", alpha=0.15)
    set_robust_ylim(axes[0], np.r_[pvdf_plot, pr_plot], 0.5, 99.5)
    axes[0].set_ylabel("signal")
    axes[0].legend(loc="upper right")

    axes[1].plot(t[sl], gate["motion_score"][sl], color="tab:red", linewidth=0.9, label="motion score")
    axes[1].plot(t[sl], gate["voltage_score"][sl], linewidth=0.7, label="voltage")
    axes[1].plot(t[sl], gate["envelope_score"][sl], linewidth=0.7, label="envelope")
    axes[1].plot(t[sl], gate["disorder_score"][sl], linewidth=0.7, label="disorder")
    axes[1].axhline(cfg.candidate_score_z, color="black", linestyle="--", linewidth=1.0)
    set_robust_ylim(axes[1], gate["motion_score"][sl], 0.0, 99.5)
    axes[1].set_ylabel("score")
    axes[1].legend(loc="upper right")

    axes[2].plot(t[sl], gate["event_evidence"][sl], color="tab:purple", linewidth=0.8, label="event evidence")
    axes[2].plot(t[sl], gate["breathing_coherence"][sl] * cfg.event_confirm_z, color="tab:green", linewidth=0.8, label="breathing coherence")
    axes[2].axhline(cfg.event_confirm_z, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("event")
    axes[2].legend(loc="upper right")

    lanes = [
        ("raw", "raw_candidate_mask"),
        ("protect", "protected_candidate_mask"),
        ("pre", "pre_dilated_candidate_mask"),
        ("merge1", "merged_candidate_mask"),
        ("confirm", "confirmed_motion_mask"),
        ("merge2", "merged_motion_mask"),
        ("final", "motion_mask"),
    ]
    for lane, (label, key) in enumerate(lanes):
        y = np.full_like(t[sl], lane, dtype=float)
        axes[3].fill_between(t[sl], y, y + 0.75 * np.asarray(gate[key])[sl].astype(float), step="post", alpha=0.55, label=label)
    axes[3].set_yticks(np.arange(len(lanes)) + 0.35)
    axes[3].set_yticklabels([label for label, _ in lanes])
    axes[3].set_ylim(-0.2, len(lanes))
    axes[3].set_ylabel("steps")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(loc="upper right", ncol=7)

    fig.suptitle("Breathing-structure-protected motion detection", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def analyze_csv(csv_path: Path, out_dir: Path, cfg: StructureConfig) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_dual_channel_csv(csv_path, cfg)
    gate = compute_motion_gate(data["pvdf"], data["pr"], cfg)
    motion_segments, clean_segments = build_segments(data, gate, cfg)
    step_table = build_step_table(data, gate)

    motion_segments.to_csv(out_dir / "motion_segments.csv", index=False, encoding="utf-8-sig")
    clean_segments.to_csv(out_dir / "clean_segments.csv", index=False, encoding="utf-8-sig")
    step_table.to_csv(out_dir / "motion_steps.csv", index=False, encoding="utf-8-sig")
    plot_result(out_dir / "motion_overview.png", data, gate, cfg, detail=False)
    plot_result(out_dir / "motion_detail.png", data, gate, cfg, detail=True)

    duration = len(data["t"]) / cfg.fs if len(data["t"]) else 0.0
    motion_seconds = float(np.sum(gate["motion_mask"]) / cfg.fs)
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
        "config": asdict(cfg),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.txt").write_text(
        "\n".join(
            [
                "Breathing-structure-protected motion detection summary",
                f"Input: {summary['input_csv']}",
                f"Duration: {summary['duration_sec']:.1f} s",
                f"Motion seconds: {summary['motion_seconds']:.1f}",
                f"Motion ratio: {summary['motion_ratio']:.3f}",
                f"Motion segments: {summary['motion_segments_total']}",
                f"Clean segments: {summary['clean_segments_total']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary["outputs"] = {
        "motion_segments_csv": str(out_dir / "motion_segments.csv"),
        "clean_segments_csv": str(out_dir / "clean_segments.csv"),
        "motion_steps_csv": str(out_dir / "motion_steps.csv"),
        "overview_png": str(out_dir / "motion_overview.png"),
        "detail_png": str(out_dir / "motion_detail.png"),
    }
    return summary


def main() -> None:
    cfg = StructureConfig(**USER_CONFIG)
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
