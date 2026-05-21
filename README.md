# PVDF + 压阻双通道体动事件划分

当前脚本只做体动事件划分，并输出去掉体动后的 clean 片段。不做 SQI、低通气/暂停、小波、SampEn 或窗口级判断。

## 方法流程

```text
PVDF + PR
-> 每通道计算电压变化率 + 包络变化率
-> PR 候选分数 candidate_score
-> PR 阈值得到候选点
-> 候选点扩展并合并为候选事件
-> 过长 PR 候选事件用 PVDF 冲击变化率做两轮 Otsu 分离
-> 分离后的事件直接作为体动核心
-> 最终边界扩展
```

`motion_steps.csv` 的关键步骤：

```text
raw_candidate    = candidate_score > motion_threshold_z
after_pre_dilate = 候选点向两边扩展 pre_motion_dilate_sec 秒
candidate_event  = 间隔不超过 motion_merge_gap_sec 的候选点合并成事件
after_separate   = 过长 PR 候选事件按 PVDF 冲击变化率的两轮 Otsu 高分核心分离
after_verify     = 去掉持续时间小于 motion_min_duration_sec 的短核心
final_motion     = 最终体动事件，并向两边扩展 motion_dilate_sec 秒
```

## 长事件分离

长候选事件内部不再用 PR 分数分离，而是使用 PVDF 冲击变化率：

```text
split_pvdf_score  = moving_average(PVDF_voltage_rate_z, 1 second)
pass1_core        = Otsu(split_pvdf_score)
pass2_core        = same Otsu union on NOT pass1_core
final_split_core  = pass1_core OR pass2_core
```

这样做是为了避免规则呼吸幅度变大时，电压变化率把每个呼吸上下坡都切成体动；包络变化率负责抓突变边界，包络值只作为补充证据，不能单独把持续高幅平台判为体动。长候选事件内部使用两层 Otsu：

```text
第 1 层：在整个长候选事件内找高分核心
第 2 层：如果第 1 层核心仍然超过 event_split_min_sec，只在这个长核心内部再找更强核心
包络值核心：只有靠近包络变化率核心时才并入，不能单独生成长体动段
```

每一次 Otsu 都要通过可靠性检查：

```text
1. Otsu 阈值必须高于 motion_threshold_z
2. 高分组比例不能过小或过大
3. 高分组和低分组的鲁棒分离度必须足够大
```

短候选事件保持完整。长候选事件如果两阶段 Otsu 找不到可靠核心，则认为“分离不了”，直接放回 clean。

分离后会再做一个持续时间验证：小于 `motion_min_duration_sec` 的孤立短核心视为毛刺，放回 clean。该验证发生在最终边界扩展前。

PVDF 分数和 PR 总分数仍保留在 `motion_steps.csv` 和图中作为参考诊断；当前体动 mask 由 PR 总分生成候选，再由 PVDF 冲击变化率对长候选事件做内部定位。

## 默认参数

```text
motion_threshold_z = 5.0
pre_motion_dilate_sec = 0.5
motion_merge_gap_sec = 10.0
event_split_min_sec = 15.0
event_split_gap_sec = 2.0
motion_min_duration_sec = 1.0
motion_dilate_sec = 1.0
pvdf_weight = 0.0
voltage_rate_weight = 0.5
envelope_rate_weight = 0.5
```

说明：

- `motion_threshold_z`：PR 候选阈值，越低候选越多。
- `pre_motion_dilate_sec`：候选点先扩展，补偿体动起止边界。
- `motion_merge_gap_sec`：相邻候选点间隔不超过该值时合并为同一候选事件。
- `event_split_min_sec`：只对超过该时长的长候选事件做分离，短事件保持完整。
- `event_split_gap_sec`：长候选事件内部高分核心间隔超过该值时切开。
- `motion_min_duration_sec`：分离后小于该时长的孤立短核心放回 clean。
- `motion_dilate_sec`：最终体动事件边界扩展。
- `pvdf_weight`：仅影响图中的融合参考分数；当前候选和分离不使用该权重。

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
- `motion_detail.png`：最强 PR 候选分数附近的局部图。
- `summary.json` / `summary.txt`：总体统计和参数。
