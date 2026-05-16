# PVDF + 压阻双通道体动事件划分

当前脚本只做体动事件划分，并输出去掉体动后的 clean 片段。不做 SQI、低通气/暂停、小波、SampEn 或窗口级判断。

## 方法流程

```text
PVDF + PR
-> 电压变化率 + 包络变化率
-> 双通道融合体动分数 motion_score
-> 阈值得到候选点
-> 候选点扩展并合并为候选事件
-> 过长候选事件按内部候选核心分离
-> 对每个候选事件计算综合验证分数 verification_score
-> 通过验证的事件保留为体动
-> 最终边界扩展
```

`motion_steps.csv` 的关键步骤：

```text
raw_candidate    = motion_score > motion_threshold_z
after_pre_dilate = 候选点向两边扩展 pre_motion_dilate_sec 秒
candidate_event  = 间隔不超过 motion_merge_gap_sec 的候选点合并成事件
after_separate   = 过长候选事件按内部候选核心分离
after_verify     = verification_score >= 1 的候选事件
final_motion     = 最终体动事件，并向两边扩展 motion_dilate_sec 秒
```

## 综合验证分数

长候选事件分离使用 Otsu 在事件内部自动寻找高分核心阈值。Otsu 会分别作用于融合分数、PVDF 单通道分数和 PR 单通道分数，再取高分核心并集。这样 PR 的巨大接触峰不会掩盖 PVDF 的冲击峰，PVDF 的尖峰也不会掩盖 PR 的接触变化。若三路都没有可靠核心，则认为该长事件没有明确体动核心，直接放回 clean，不再进入验证。

对每个候选事件计算两类证据：

```text
kalman_segment_score =
    percentile(abs(PVDF_resp - Kalman_prediction) / clean_residual_scale, 95)

pr_contact_segment_score =
    percentile(PR_contact_score within candidate event, 95)
```

统一验证分数：

```text
verification_score =
    max(
        kalman_segment_score / kalman_residual_threshold_z,
        pr_contact_segment_score / pr_contact_threshold_z
    )

verification_score >= 1 -> 保留为体动事件
```

这样 PVDF 负责判断呼吸轨迹是否被体动破坏，PR 负责提供接触/体位变化证据。两个通道任一通道出现强证据即可确认体动；如果两个通道都只是中等偏离，则不靠相加过关，避免把呼吸幅度变化判成体动。

## 默认参数

```text
motion_threshold_z = 2.0
pre_motion_dilate_sec = 0.0
motion_merge_gap_sec = 5.0
event_split_min_sec = 25.0
event_split_gap_sec = 2.0
motion_dilate_sec = 2.0
pvdf_weight = 0.5
voltage_rate_weight = 0.5
envelope_rate_weight = 0.5
kalman_context_sec = 25.0
kalman_residual_threshold_z = 6.0
pr_contact_threshold_z = 12.0
```

说明：

- `motion_threshold_z`：第一阶段候选阈值，越低候选越多。
- `pre_motion_dilate_sec`：候选点先扩展，补偿体动起止边界。
- `motion_merge_gap_sec`：相邻候选点间隔不超过该值时合并为同一事件。
- `event_split_min_sec`：只对超过该时长的长候选事件做分离，短事件保持完整；长事件若 Otsu 不可靠则视为非体动。
- `event_split_gap_sec`：长候选事件内部 Otsu 高分核心间隔超过该值时切开。
- `motion_dilate_sec`：最终体动事件边界扩展。
- `pvdf_weight`：PVDF 在双通道融合分数中的权重。
- `kalman_context_sec`：候选事件前后用于建立 PVDF 呼吸模型的 clean 时长。
- `kalman_residual_threshold_z`：PVDF Kalman 残差归一化阈值。
- `pr_contact_threshold_z`：PR 接触变化归一化阈值。

## 运行

在 `quality_gating.py` 顶部修改：

```python
DEFAULT_CSV = Path(...)
OUTPUT_DIR = Path("outputs")
USER_CONFIG = {...}
```

然后运行：

```powershell
python .\quality_gating.py
```

## 输出

- `motion_segments.csv`：最终体动事件。
- `clean_segments.csv`：去掉体动后的可用片段。
- `motion_steps.csv`：逐采样点步骤诊断表。
- `motion_overview.png`：整段分步骤可视化。
- `motion_detail.png`：最强体动分数附近的局部图。
- `summary.json` / `summary.txt`：总体统计和参数。
