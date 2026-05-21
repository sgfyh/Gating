# 体动事件划分算法说明

## 目标

当前版本只划分体动事件，并输出去掉体动后的 clean 片段。算法不包含 SQI、低通气/暂停、小波、SampEn 或窗口级可用性判断。

## 1. 预处理

1. `pvdf_adc` 和 `pr_adc` 转成电压。
2. 异常 ADC 点插值，同时记录异常比例。
3. 从 500 Hz 降采样到 50 Hz。

## 2. 变化率特征

每个通道计算两类体动候选特征：

```text
voltage_rate_z = 平滑电压的一阶变化率鲁棒 z 分数
env_rate_z     = 去趋势后包络的一阶变化率鲁棒 z 分数
```

电压变化率偏向检测瞬时电压突变；包络变化率偏向检测振幅或接触状态变化。

## 3. PR 候选生成

```text
PR_score =
    voltage_rate_weight  * PR_voltage_rate_z
  + envelope_rate_weight * PR_env_rate_z

candidate_score = moving_average(PR_score, 1 second)
```

`voltage_rate_weight` 和 `envelope_rate_weight` 会自动归一化。`candidate_score` 只来自 PR 通道，用 1 秒移动平均减少单点噪声。PVDF_score 和 motion_score 仍然输出到图和表中，作为对照诊断，不参与当前体动 mask。

候选事件生成：

```text
raw_candidate = candidate_score > motion_threshold_z
pre_dilate    = expand(raw_candidate, pre_motion_dilate_sec)
candidate_event = merge(pre_dilate, gap <= motion_merge_gap_sec)
```

## 4. 两阶段 Otsu 长事件分离

如果候选事件持续时间超过 `event_split_min_sec`，则在事件内部使用 PVDF 冲击变化率作为分离分数：

```text
split_pvdf_score  = moving_average(PVDF_voltage_rate_z, 1 second)
pass1_core        = Otsu(split_pvdf_score)
pass2_core        = same Otsu union on NOT pass1_core
final_split_core  = pass1_core OR pass2_core
```

候选阶段仍用 PR 总分数保证召回；分离阶段不用总分数，是为了降低规则大幅呼吸的电压变化率把长段切碎的风险。包络变化率负责抓突变边界，包络值只作为补充证据，不能独立产生长时间体动 mask。长候选事件内部使用两层 Otsu：第一层先在整个长候选事件内找粗异常核心；如果这个核心仍然过长，第二层只在该核心内部继续找更强异常核心。包络变化率核心可独立成立，包络值核心只有靠近包络变化率核心时才并入。

```text
layer 1:
    在整个长候选事件内做 Otsu，得到高分核心

layer 2:
    如果 layer 1 的高分核心仍然超过 event_split_min_sec，
    只在该长核心内部再做一次 Otsu，得到更强核心

```

每个阶段都必须满足可靠性条件：

```text
threshold > motion_threshold_z
0.005 <= high_ratio <= 0.60
separation = (mean(high) - mean(low)) / robust_scale(score) >= 1.5
```

分离逻辑：

```text
separated_event =
    keep short candidate_event unchanged
    find reliable two-pass Otsu cores from the PVDF impulse score
    expand high core locally within candidate support
    split long candidate_event when high-core gaps > event_split_gap_sec
    reject long candidate_event if no reliable core exists
```

因此，长候选段如果分离不出可靠核心，就直接放回 clean。这是为了避免在单峰或缓慢漂移数据中强行切出体动。

## 5. 持续时间验证与边界修正

分离后的核心先做持续时间验证，小于 `motion_min_duration_sec` 的孤立短核心放回 clean：

```text
after_verify =
    keep separated_event runs with duration >= motion_min_duration_sec
```

通过持续时间验证的事件作为体动核心，再向两边扩展 `motion_dilate_sec` 秒：

```text
final_motion = expand(after_verify, motion_dilate_sec)
```

当前主流程不使用 Kalman 残差，也不再单独做 PR 验证。PR 总分数负责候选，PVDF 冲击变化率负责长事件分离；额外 PR 验证不提供独立信息。Kalman 适合做后续对照实验，不放在主算法里。

## 6. 可视化

`motion_overview.png` 和 `motion_detail.png` 展示：

1. PVDF/PR 波形和最终体动阴影。
2. PR 候选分数、PVDF 参考分数和融合参考分数。
3. PVDF 冲击分离分数、事件 P95 分离分数和 Otsu 分离阈值。
4. `candidate -> pre_dilate -> event -> separate -> verify -> final` 的逐步 mask。

`motion_steps.csv` 保存同样的逐点诊断信息。
