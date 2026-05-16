from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quality_gating import (
    DEFAULT_CSV,
    OUTPUT_DIR,
    USER_CONFIG,
    GateConfig,
    build_step_table,
    compute_motion_gate,
    load_dual_channel_csv,
    mask_runs,
    normalized_for_plot,
    set_robust_ylim,
)

REVIEW_DIR = OUTPUT_DIR.parent / f"{OUTPUT_DIR.name}_review"
EVENT_PAD_SEC = 20.0


def event_union_mask(gate: dict[str, np.ndarray | float]) -> np.ndarray:
    keys = [
        "raw_candidate_mask",
        "merged_candidate_mask",
        "split_candidate_mask",
        "variance_motion_mask",
        "motion_mask",
    ]
    mask = np.zeros_like(np.asarray(gate["motion_mask"], dtype=bool))
    for key in keys:
        mask |= np.asarray(gate[key], dtype=bool)
    return mask


def event_diagnostics(
    event_id: int,
    start: int,
    end: int,
    gate: dict[str, np.ndarray | float],
    cfg: GateConfig,
) -> dict[str, object]:
    sl = slice(start, end)
    duration = (end - start) / cfg.fs
    final_overlap = bool(np.any(np.asarray(gate["motion_mask"])[sl]))
    split_removed = bool(np.any(np.asarray(gate["split_removed_mask"])[sl]))
    variance_kept = bool(np.any(np.asarray(gate["variance_motion_mask"])[sl]))
    return {
        "event_id": event_id,
        "start_idx": start,
        "end_idx": end,
        "start_sec": start / cfg.fs,
        "end_sec": end / cfg.fs,
        "duration_sec": duration,
        "algorithm_final_motion": final_overlap,
        "split_removed_any": split_removed,
        "variance_kept_any": variance_kept,
        "max_motion_score": float(np.nanmax(np.asarray(gate["motion_score"])[sl])),
        "max_voltage_score": float(np.nanmax(np.asarray(gate["voltage_score"])[sl])),
        "max_envelope_score": float(np.nanmax(np.asarray(gate["envelope_score"])[sl])),
        "max_segment_variance_score": float(np.nanmax(np.asarray(gate["segment_variance_score"])[sl])),
        "manual_label": "",
        "review_note": "",
        "label_hint": "motion / clean_breathing / contact_stable / uncertain",
    }


def build_review_events(gate: dict[str, np.ndarray | float], cfg: GateConfig) -> pd.DataFrame:
    rows = []
    for event_id, (start, end) in enumerate(mask_runs(event_union_mask(gate))):
        rows.append(event_diagnostics(event_id, start, end, gate, cfg))
    return pd.DataFrame(rows)


def plot_event(
    out_path: Path,
    event: pd.Series,
    data: dict[str, np.ndarray],
    gate: dict[str, np.ndarray | float],
    cfg: GateConfig,
) -> None:
    t = data["t"]
    pad = int(round(EVENT_PAD_SEC * cfg.fs))
    start = max(0, int(event.start_idx) - pad)
    end = min(len(t), int(event.end_idx) + pad)
    sl = slice(start, end)

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    pvdf_norm = normalized_for_plot(data["pvdf"][sl])
    pr_norm = normalized_for_plot(data["pr"][sl])

    axes[0].plot(t[sl], pvdf_norm, linewidth=0.9, label="PVDF")
    axes[0].plot(t[sl], pr_norm, linewidth=0.8, alpha=0.8, label="PR")
    axes[0].axvspan(event.start_sec, event.end_sec, color="tab:orange", alpha=0.18, label="review event")
    axes[0].fill_between(t[sl], -6, 6, np.asarray(gate["motion_mask"])[sl].astype(float), color="tab:red", alpha=0.12)
    set_robust_ylim(axes[0], np.r_[pvdf_norm, pr_norm], 0.5, 99.5)
    axes[0].set_ylabel("norm.")
    axes[0].legend(loc="upper right")

    axes[1].plot(t[sl], np.asarray(gate["voltage_score"])[sl], linewidth=0.8, label="voltage")
    axes[1].plot(t[sl], np.asarray(gate["envelope_score"])[sl], linewidth=0.8, label="envelope")
    axes[1].plot(t[sl], np.asarray(gate["motion_score"])[sl], color="tab:red", linewidth=1.0, label="fused")
    axes[1].axhline(float(gate["motion_threshold"]), color="black", linestyle="--", linewidth=1.0)
    axes[1].axvspan(event.start_sec, event.end_sec, color="tab:orange", alpha=0.14)
    set_robust_ylim(axes[1], np.asarray(gate["motion_score"])[sl], 0.0, 99.5)
    axes[1].set_ylabel("score")
    axes[1].legend(loc="upper right")

    axes[2].plot(t[sl], np.asarray(gate["segment_variance_score"])[sl], color="tab:purple", linewidth=0.9, label="segment variance")
    if cfg.segment_variance_threshold_v > 0:
        axes[2].axhline(cfg.segment_variance_threshold_v, color="black", linestyle="--", linewidth=1.0)
    axes[2].axvspan(event.start_sec, event.end_sec, color="tab:orange", alpha=0.14)
    axes[2].set_ylabel("std")
    axes[2].legend(loc="upper right")

    lanes = [
        ("raw", "raw_candidate_mask"),
        ("merge1", "merged_candidate_mask"),
        ("split", "split_candidate_mask"),
        ("variance", "variance_motion_mask"),
        ("merge2", "merged_motion_mask"),
        ("final", "motion_mask"),
    ]
    for lane, (label, key) in enumerate(lanes):
        y = np.full_like(t[sl], lane, dtype=float)
        axes[3].fill_between(t[sl], y, y + 0.75 * np.asarray(gate[key])[sl].astype(float), step="post", alpha=0.55, label=label)
    axes[3].axvspan(event.start_sec, event.end_sec, color="tab:orange", alpha=0.12)
    axes[3].set_yticks(np.arange(len(lanes)) + 0.35)
    axes[3].set_yticklabels([label for label, _ in lanes])
    axes[3].set_ylim(-0.2, len(lanes))
    axes[3].set_ylabel("steps")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(loc="upper right", ncol=6)

    fig.suptitle(
        f"Review event {int(event.event_id):03d}: {event.start_sec:.1f}-{event.end_sec:.1f}s, final={bool(event.algorithm_final_motion)}",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_index(review_dir: Path, events: pd.DataFrame) -> None:
    rows = []
    for event in events.itertuples(index=False):
        image = f"event_{int(event.event_id):03d}.png"
        rows.append(
            "<tr>"
            f"<td>{int(event.event_id)}</td>"
            f"<td>{event.start_sec:.1f}</td>"
            f"<td>{event.end_sec:.1f}</td>"
            f"<td>{event.duration_sec:.2f}</td>"
            f"<td>{html.escape(str(event.algorithm_final_motion))}</td>"
            f"<td>{event.max_motion_score:.2f}</td>"
            f"<td><a href='{image}'>{image}</a></td>"
            "</tr>"
        )
    body = "\n".join(rows)
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Motion Review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: right; }}
    th {{ background: #f2f2f2; }}
    td:last-child, th:last-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Motion Review Events</h1>
  <p>先在 <code>review_events.csv</code> 里给 <code>manual_label</code> 标注：motion / clean_breathing / contact_stable / uncertain。</p>
  <table>
    <thead>
      <tr><th>ID</th><th>Start</th><th>End</th><th>Duration</th><th>Final</th><th>Max Score</th><th>Plot</th></tr>
    </thead>
    <tbody>{body}</tbody>
  </table>
</body>
</html>
"""
    (review_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    cfg = GateConfig(**USER_CONFIG)
    data = load_dual_channel_csv(DEFAULT_CSV, cfg)
    gate = compute_motion_gate(data["pvdf"], data["pr"], cfg)

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    events = build_review_events(gate, cfg)
    events.to_csv(REVIEW_DIR / "review_events.csv", index=False, encoding="utf-8-sig")
    build_step_table(data, gate).to_csv(REVIEW_DIR / "motion_steps.csv", index=False, encoding="utf-8-sig")

    for event in events.itertuples(index=False):
        plot_event(REVIEW_DIR / f"event_{int(event.event_id):03d}.png", pd.Series(event._asdict()), data, gate, cfg)

    summary = {
        "input_csv": str(DEFAULT_CSV),
        "review_dir": str(REVIEW_DIR),
        "events_total": int(len(events)),
        "algorithm_final_events": int(events["algorithm_final_motion"].sum()) if not events.empty else 0,
        "config": asdict(cfg),
        "label_instruction": "Fill manual_label with: motion / clean_breathing / contact_stable / uncertain",
    }
    (REVIEW_DIR / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_index(REVIEW_DIR, events)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
