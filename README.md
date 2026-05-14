# PVDF + 压阻双通道体动片段划分

当前脚本只做体动片段划分：

```text
原始 PVDF + 压阻
-> 电压变化率 + 包络变化率
-> 得到候选体动片段
-> 对每个候选片段计算方差
-> 长候选片段用方差二次确认，短候选片段直接保留
-> 方差大的保留为体动，方差小的放回 clean
```

不再做 SQI、低通气/暂停、小波、SampEn 或窗口级判断。

## 运行

在 `quality_gating.py` 顶部修改：

```python
DEFAULT_CSV = Path(...)
OUTPUT_DIR = Path("outputs")
USER_CONFIG = {...}
```

然后直接运行：

```powershell
python .\quality_gating.py
```

## 输出

- `motion_segments.csv`：最终体动片段。
- `clean_segments.csv`：去掉体动后的可用片段。
- `motion_steps.csv`：逐采样点步骤诊断表。
- `motion_overview.png`：整段分步骤可视化。
- `motion_detail.png`：最强体动分数附近的局部图。
- `summary.json` / `summary.txt`：总体统计和参数。

`motion_steps.csv` 的关键步骤：

```text
raw_candidate  = motion_score > motion_threshold_z
after_pre_dilate = 候选体动先向两边拓展 pre_motion_dilate_sec 秒
after_first_merge = 先把间隔不超过 motion_merge_gap_sec 的候选段合并
after_split    = 长候选段中，只保留有电压变化率支撑或强分数支撑的部分
after_variance = 候选片段方差足够大
after_second_merge = 方差确认后再次合并，参与分离的片段也参与合并
final_motion   = 最终体动片段，并向两边拓展 motion_dilate_sec 秒
```

## 默认参数

```text
motion_threshold_z = 2.2
segment_variance_threshold_v = 0.15
segment_variance_min_sec = 3.0
pre_motion_dilate_sec = 0.5
motion_merge_gap_sec = 3.0
motion_dilate_sec = 2.0
pvdf_weight = 0.6
voltage_rate_weight = 0.5
envelope_rate_weight = 0.5
segment_pr_variance_gain = 4.0
long_candidate_split_min_sec = 10.0
long_candidate_voltage_support_z = 16.0
long_candidate_strong_score_z = 25.0
```

说明：

- `motion_threshold_z` 是固定直接阈值，越低候选越多。
- `segment_variance_threshold_v` 越高，越容易把小幅稳定变化放回 clean。
- `segment_variance_threshold_v <= 0` 会关闭方差过滤，所有候选都保留。
- `segment_variance_min_sec = 3.0` 表示只对持续大于 3 秒的候选片段做方差确认，3 秒及以下的小体动先保留。
- `pre_motion_dilate_sec = 0.5` 表示候选体动先向两边各拓展 0.5 秒，再做第一次合并。
- `motion_merge_gap_sec = 3.0` 表示方差确认完成后，最终保留的体动段间隔不超过 3 秒时合并。
- `motion_dilate_sec = 2.0` 表示最终合并后向两边各拓展 2 秒，用来补回长段分离后被切瘦的体动边界。
- `long_candidate_split_min_sec = 10.0` 表示只对超过 10 秒的长候选段做分离。
- `long_candidate_voltage_support_z` 和 `long_candidate_strong_score_z` 是长段分离用的更高阈值，防止正常幅度变化和两边真实体动粘连后一起进入方差判断。

片段方差分数：

```text
segment_variance_score = max(std(PVDF), segment_pr_variance_gain * std(PR))
```

这样压阻因为接触原因出现“电压变小但稳定”的时段，一般不会因为稳定低电压本身被判成体动；只有候选片段内部波动方差足够大，才会保留为体动。
