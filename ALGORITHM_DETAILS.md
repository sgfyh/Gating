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

## 3. 双通道候选生成

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

候选事件生成：

```text
raw_candidate = motion_score > motion_threshold_z
pre_dilate    = expand(raw_candidate, pre_motion_dilate_sec)
candidate_event = merge(pre_dilate, gap <= motion_merge_gap_sec)
```

这里 PVDF 主要提供快速冲击/振动证据，PR 主要提供接触压力或体位变化证据。二者在 `motion_score` 中统一融合，不再额外设置单通道候选分支。

如果候选事件持续时间超过 `event_split_min_sec`，则用 Otsu 在事件内部自动寻找高分核心阈值：

```text
separated_event =
    keep short candidate_event unchanged
    high_core = union(
        reliable Otsu core of motion_score within event,
        reliable Otsu core of PVDF_score within event,
        reliable Otsu core of PR_score within event
    )
    split long candidate_event when high_core gaps > event_split_gap_sec
    reject long candidate_event if all Otsu cores are unreliable
```

为避免单峰分布被强行切开，Otsu 阈值只有在高分组比例合理、且高低分组均值分离度足够时才启用。长候选事件按融合分数、PVDF 单通道分数和 PR 单通道分数分别寻找核心，再取并集；否则 PR 的巨大接触峰可能会掩盖 PVDF 的真实冲击。若三路都不可靠，该长候选事件视为没有明确体动核心，直接放回 clean。短候选事件不做 Otsu 分离，保持完整进入验证。

这一步只处理长事件粘连，不承担二阶段确认功能。

## 4. 事件级综合验证

对每个分离后的候选事件计算两个分数。

PVDF Kalman 残差：

```text
候选事件前后 clean PVDF
-> 估计呼吸主频
-> 建立二维呼吸振荡器 Kalman 模型
-> 从候选事件前的 clean 状态开始，只预测、不用候选事件更新
-> 计算候选事件 PVDF 呼吸分量与预测值的残差

kalman_segment_score =
    percentile(abs(PVDF_resp - Kalman_prediction) / clean_residual_scale, 95)
```

PR 接触变化分数：

```text
pr_contact_segment_score =
    percentile(PR_score within candidate_event, 95)
```

统一验证分数：

```text
verification_score =
    max(
        kalman_segment_score / kalman_residual_threshold_z,
        pr_contact_segment_score / pr_contact_threshold_z
    )

verification_score >= 1 -> 保留为体动事件
verification_score <  1 -> 驳回，放回 clean
```

这个形式让 PVDF 和 PR 的贡献在同一个公式中体现：PVDF 负责判断呼吸轨迹是否被破坏，PR 负责提供接触/体位变化证据。任一通道达到确认阈值即可保留；如果两个通道都只是中等偏离，则不通过相加确认，从而减少呼吸幅度变化造成的假阳性。

如果候选事件前后的 clean 上下文不足以建立 Kalman 模型，则保留该候选事件，避免模型不可用导致漏检。

## 5. 边界修正

通过验证的候选事件直接作为体动事件，再向两边扩展 `motion_dilate_sec` 秒：

```text
final_motion = expand(verified_event, motion_dilate_sec)
```

这样体动核心、尾部恢复段和接触恢复段由候选事件整体保留，不再额外做 split、fill 或二次合并。

## 6. 可视化

`motion_overview.png` 和 `motion_detail.png` 展示：

1. PVDF/PR 波形和最终体动阴影。
2. 电压变化率、包络变化率和融合体动分数。
3. Kalman 残差、PR 接触变化分数和统一验证分数。
4. `candidate -> pre_dilate -> event -> separate -> verify -> final` 的逐步 mask。

`motion_steps.csv` 保存同样的逐点诊断信息。
