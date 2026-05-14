# 体动片段划分算法说明

## 目标

当前版本只划分体动片段，并输出去掉体动后的 clean 片段。算法不再包含 SQI、低通气/暂停、小波、SampEn 或窗口级可用性判断。

## 1. 预处理

1. `pvdf_adc` 和 `pr_adc` 转成电压。
2. 异常 ADC 点插值，同时记录异常比例。
3. 从 500 Hz 降采样到 50 Hz。

## 2. 两类变化率特征

每个通道只计算两类体动候选特征：

```text
voltage_rate_z = 平滑电压的一阶变化率鲁棒 z 分数
env_rate_z     = 去趋势后包络的一阶变化率鲁棒 z 分数
```

电压变化率偏向检测瞬时电压突变；包络变化率偏向检测振幅/接触状态变化。两者有相关性，但不完全重复。

## 3. 双通道融合

```text
voltage_score =
    pvdf_weight       * PVDF_voltage_rate_z
  + (1 - pvdf_weight) * PR_voltage_rate_z

envelope_score =
    pvdf_weight       * PVDF_env_rate_z
  + (1 - pvdf_weight) * PR_env_rate_z

motion_score =
    voltage_rate_weight  * voltage_score
  + envelope_rate_weight * envelope_score
```

`voltage_rate_weight` 和 `envelope_rate_weight` 会自动归一化。融合后做 1 秒移动平均，减少单点噪声。

## 4. 候选体动片段

```text
raw_candidate = motion_score > motion_threshold_z
```

当前使用固定直接阈值：

```text
raw_candidate = motion_score > motion_threshold_z
```

阈值越低，候选体动越多；阈值越高，候选体动越少。

## 5. 第一次合并和长候选段拆分

候选体动先向两边拓展 `pre_motion_dilate_sec` 秒，再把间隔不超过 `motion_merge_gap_sec` 的候选体动合并，形成候选事件：

```text
raw_candidate -> pre_dilate -> first_merge
```

如果长候选段主要来自包络慢变，容易和两侧真实体动粘连。仅对持续时间超过 `long_candidate_split_min_sec` 的长候选段，在方差判断前做一次拆分：

```text
长候选段内保留：
  voltage_score >= long_candidate_voltage_support_z
  或 motion_score >= long_candidate_strong_score_z
```

这样正常呼吸幅度逐渐变大但缺少尖峰支撑的部分，会先从候选段中切掉，再进入方差判断。

## 6. 片段方差二次确认

对每个候选体动片段单独计算：

```text
segment_variance_score = max(std(PVDF), segment_pr_variance_gain * std(PR))
```

然后判断：

```text
duration <= segment_variance_min_sec -> 直接保留，避免漏掉短小体动
segment_variance_score >= segment_variance_threshold_v -> 保留为体动
segment_variance_score <  segment_variance_threshold_v -> 放回 clean
```

默认 `segment_variance_min_sec = 3.0`，只对大于 3 秒的候选片段做方差确认。如果 `segment_variance_threshold_v <= 0`，则关闭方差过滤，所有候选片段都会保留。

这样做的目的，是把“压阻电压变小但很稳定”的接触变化和真正有明显波动的体动区分开。

## 7. 第二次合并和膨胀

候选片段完成方差确认后，再合并间隔不超过 `motion_merge_gap_sec` 的体动片段。参与长段分离的片段也参与这次最终合并：

```text
motion_merge_gap_sec = 3.0
```

最终合并后再向两边拓展一小段，用来补回长段分离后被切瘦的体动边界：

```text
motion_dilate_sec = 2.0
```

## 8. 可视化

`motion_overview.png` 和 `motion_detail.png` 展示：

1. PVDF/PR 波形和最终体动阴影。
2. 电压变化率、包络变化率和融合体动分数。
3. 候选片段的方差分数和方差阈值。
4. `candidate -> dilate0.5 -> merge1 -> split -> variance -> merge2 -> final` 的逐步 mask。

`motion_steps.csv` 保存同样的逐点诊断信息，用来定位某段体动是在哪一步被保留或剔除的。
