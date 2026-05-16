#!/usr/bin/env python3
"""Classify every sample as motion (1) or non-motion (0).

Input:  CSV with pvdf_adc + pr_adc columns.
Output: segments.csv  — one row per contiguous run with label + summary stats.
        motion_mask.npz — full-length boolean mask (motion=1, non-motion=0).
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quality_gating import (
    GateConfig,
    load_dual_channel_csv,
    compute_motion_gate,
    mask_runs,
)


def classify(csv_path: Path, cfg: GateConfig, out_dir: Path) -> dict:
    data = load_dual_channel_csv(csv_path, cfg)
    gate = compute_motion_gate(data["pvdf"], data["pr"], cfg)

    mask = gate["motion_mask"]
    t = data["t"]
    fs = cfg.fs

    # ---- segments table ----
    rows = []
    for seg_id, (start, end) in enumerate(mask_runs(mask)):
        dur = (end - start) / fs
        label = "motion"
        rows.append({
            "segment_id": seg_id,
            "start_sec":  round(float(t[start]), 3),
            "end_sec":    round(float(t[end - 1]), 3) if end > start else 0.0,
            "duration_sec": round(dur, 3),
            "label": label,
            "mean_fused_score": round(float(np.mean(gate["motion_score"][start:end])), 3),
            "max_fused_score":  round(float(np.max(gate["motion_score"][start:end])), 3),
            "pvdf_std_v": round(float(np.nanstd(data["pvdf"][start:end])), 4),
            "pr_std_v":   round(float(np.nanstd(data["pr"][start:end])), 4),
        })

    non_motion_runs = mask_runs(~mask)
    offset = len(rows)
    for seg_id, (start, end) in enumerate(non_motion_runs):
        dur = (end - start) / fs
        rows.append({
            "segment_id": offset + seg_id,
            "start_sec":  round(float(t[start]), 3),
            "end_sec":    round(float(t[end - 1]), 3) if end > start else 0.0,
            "duration_sec": round(dur, 3),
            "label": "non_motion",
            "mean_fused_score": round(float(np.mean(gate["motion_score"][start:end])), 3),
            "max_fused_score":  round(float(np.max(gate["motion_score"][start:end])), 3),
            "pvdf_std_v": round(float(np.nanstd(data["pvdf"][start:end])), 4),
            "pr_std_v":   round(float(np.nanstd(data["pr"][start:end])), 4),
        })

    segments = pd.DataFrame(rows).sort_values("start_sec").reset_index(drop=True)

    # ---- summary ----
    total_dur = len(mask) / fs
    motion_dur = float(mask.sum() / fs)
    motion_segs = segments[segments["label"] == "motion"]
    non_motion_segs = segments[segments["label"] == "non_motion"]

    summary = {
        "input_csv": str(csv_path),
        "total_duration_sec": round(total_dur, 2),
        "motion_duration_sec": round(motion_dur, 2),
        "motion_pct": round(100.0 * motion_dur / total_dur, 2) if total_dur else 0,
        "num_motion_segments": len(motion_segs),
        "num_non_motion_segments": len(non_motion_segs),
        "mean_segment_duration_sec": {
            "motion": round(float(motion_segs["duration_sec"].mean()), 2) if len(motion_segs) else 0,
            "non_motion": round(float(non_motion_segs["duration_sec"].mean()), 2) if len(non_motion_segs) else 0,
        },
        "config": {
            "motion_threshold_z": cfg.motion_threshold_z,
            "pvdf_weight": cfg.pvdf_weight,
            "voltage_rate_weight": cfg.voltage_rate_weight,
            "envelope_rate_weight": cfg.envelope_rate_weight,
            "segment_variance_threshold_v": cfg.segment_variance_threshold_v,
            "segment_variance_min_sec": cfg.segment_variance_min_sec,
            "segment_pr_variance_gain": cfg.segment_pr_variance_gain,
            "motion_merge_gap_sec": cfg.motion_merge_gap_sec,
            "motion_dilate_sec": cfg.motion_dilate_sec,
        },
    }

    # ---- save ----
    out_dir.mkdir(parents=True, exist_ok=True)
    segments.to_csv(out_dir / "segments.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(out_dir / "motion_mask.npz",
                        motion_mask=mask, t=t)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def main():
    p = argparse.ArgumentParser(description="Dual-channel motion / non-motion classifier.")
    p.add_argument("--input", type=Path, required=True, help="CSV with pvdf_adc, pr_adc columns")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/classification"))
    p.add_argument("--motion-threshold-z", type=float, default=2.0)
    p.add_argument("--pvdf-weight", type=float, default=0.7)
    p.add_argument("--voltage-rate-weight", type=float, default=0.5)
    p.add_argument("--envelope-rate-weight", type=float, default=0.5)
    p.add_argument("--variance-threshold-v", type=float, default=2.0)
    p.add_argument("--variance-min-sec", type=float, default=2.0)
    p.add_argument("--variance-gain-pr", type=float, default=4.0)
    p.add_argument("--merge-gap-sec", type=float, default=2.0)
    p.add_argument("--dilate-sec", type=float, default=0.0)
    args = p.parse_args()

    cfg = GateConfig(
        motion_threshold_z=args.motion_threshold_z,
        pvdf_weight=args.pvdf_weight,
        voltage_rate_weight=args.voltage_rate_weight,
        envelope_rate_weight=args.envelope_rate_weight,
        segment_variance_threshold_v=args.variance_threshold_v,
        segment_variance_min_sec=args.variance_min_sec,
        segment_pr_variance_gain=args.variance_gain_pr,
        motion_merge_gap_sec=args.merge_gap_sec,
        motion_dilate_sec=args.dilate_sec,
    )

    summary = classify(args.input, cfg, args.out_dir)

    print(f"  total     {summary['total_duration_sec']:.1f} s")
    print(f"  motion    {summary['motion_duration_sec']:.1f} s  ({summary['motion_pct']:.1f} %)")
    print(f"  segments  {summary['num_motion_segments']} motion + "
          f"{summary['num_non_motion_segments']} non-motion")
    print(f"  saved to  {args.out_dir}")


if __name__ == "__main__":
    main()
