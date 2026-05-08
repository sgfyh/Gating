# PVDF + 压阻双通道质量门控算法说明

这份文档用于说明当前 `quality_gating.py` 的完整算法过程、变量含义和公式。它适合发给网页版 ChatGPT、导师或合作者，让对方在理解现有实现的基础上继续帮你想实验设计、论文创新点和下一步改进方案。

## 1. 算法目标

床下非接触睡眠监测的难点不是安静片段中完全看不到呼吸，而是整晚数据里会混入翻身、姿态改变、接触状态变化、短时冲击等体动伪迹。当前算法的目标是：

1. 从 PVDF 通道提取较稳定的呼吸候选波形。
2. 同时利用 PVDF 和压阻通道检测体动或低质量片段。
3. 把整段数据切成滑动窗口，给每个窗口计算信号质量指数 SQI。
4. 输出哪些窗口可以用于后续呼吸率统计，哪些窗口应被剔除。

一句话概括：

```text
先用双通道融合判断体动，再用窗口级 SQI 筛出可用于呼吸率估计的高质量片段。
```

## 2. 符号定义

通道定义：

- `P`：PVDF 通道。
- `R`：压阻通道，代码中列名为 `pr_adc`。

原始 ADC 序列：

```text
a_P[n], a_R[n], n = 0, 1, ..., N_raw - 1
```

默认参数：

```text
fs_raw = 500 Hz
q = 10
fs = fs_raw / q = 50 Hz
A_max = 4095
V_ref = 3.3 V
```

降采样后的电压序列：

```text
x_P[k], x_R[k], k = 0, 1, ..., N - 1
```

时间轴：

```text
t[k] = k / fs
```

滑动窗口：

```text
window_sec = 30 s
step_sec = 5 s
```

第 `j` 个窗口记为：

```text
W_j = [s_j, e_j)
```

其中 `s_j` 和 `e_j` 是降采样后的样本索引。

## 3. 总体流程

当前程序的主流程如下：

```text
CSV 数据
  -> ADC 清洗和插值
  -> ADC 转电压
  -> 500 Hz 降采样到 50 Hz
  -> PVDF 呼吸成分提取
  -> PVDF/压阻双通道体动分数
  -> 鲁棒阈值生成体动掩码
  -> 30 s 滑动窗口特征
  -> SQI 评分和窗口标签
  -> 表格、摘要、总览图、局部图
```

对应代码入口：

- `load_dual_channel_csv()`：读取、清洗、降采样。
- `extract_pvdf_respiration()`：提取 PVDF 呼吸波形。
- `compute_motion_gate()`：计算体动分数和体动掩码。
- `build_window_table()`：生成窗口级特征表。
- `classify_window()`：计算 SQI、标签和是否通过门控。

## 4. ADC 清洗和电压转换

对每个通道 `c in {P, R}`，先判断 ADC 坏点：

```text
b_c[n] = 1, if a_c[n] is NaN/Inf or a_c[n] < 0 or a_c[n] > A_max
b_c[n] = 0, otherwise
```

其中：

```text
A_max = 4095
```

坏点被替换为缺失值，然后用线性插值补齐：

```text
ã_c[n] = interpolate(a_c[n])
```

再把 ADC 转成电压：

```text
v_c[n] = clip(ã_c[n], 0, A_max) / A_max * V_ref
```

其中：

```text
V_ref = 3.3 V
```

这样做的原因：

- 插值可以避免个别异常点造成滤波或小波分解炸点。
- 保留坏点掩码 `b_c[n]`，后面仍会统计每个窗口里的坏点比例。
- 电压单位比 ADC 原始整数更适合解释和画图。

## 5. 降采样

原始信号 500 Hz 对呼吸分析来说过高，当前算法使用 FIR 零相位降采样：

```text
x_c[k] = Decimate_q(v_c[n])
```

默认：

```text
q = 10
fs = 500 / 10 = 50 Hz
```

50 Hz 足够覆盖：

- 呼吸频段：0.10-0.60 Hz。
- 呼吸率范围：6-36 bpm。
- 慢变姿态扰动和多数体动特征。

坏点比例也同步降采样。对每个降采样块统计坏点均值：

```text
bad_c[k] = mean(b_c[kq : (k+1)q])
```

双通道坏点比例取更保守的最大值：

```text
bad[k] = max(bad_P[k], bad_R[k])
```

## 6. PVDF 呼吸成分提取

PVDF 对动态压力变化更敏感，因此当前把 PVDF 作为呼吸主信号。目标是从 `x_P[k]` 中得到呼吸候选波形 `r_P[k]`。

### 6.1 小波近似分量

对 PVDF 降采样信号做 DB5 小波分解：

```text
x_P[k] -> {A_L, D_L, D_{L-1}, ..., D_1}
```

当前最大层数：

```text
L <= 5
```

程序会根据数据长度自动选择不超过 5 的合法层数。

代码中使用最高频细节分量 `D_1` 估计噪声水平：

```text
sigma = median(|D_1|) / 0.6745
lambda = sigma * sqrt(2 * ln(N))
```

并对细节分量做软阈值：

```text
D_i' = sign(D_i) * max(|D_i| - lambda, 0)
```

当前实际输出主要保留近似分量 `A_L` 重构：

```text
r_0[k] = waverec(A_L)
```

解释：

- 近似分量保留低频趋势，更接近呼吸周期。
- 高频细节更容易包含冲击、BCG、噪声和床垫振动。
- 代码里保留了细节阈值步骤，是为了后续可以扩展成“近似分量 + 低频细节分量”的版本；当前主输出仍以 `A_L` 为主。

### 6.2 呼吸频带滤波

再对 `r_0[k]` 做 Butterworth 带通滤波：

```text
r_P[k] = BandPass(r_0[k], 0.10 Hz, 0.60 Hz)
```

频段含义：

```text
0.10 Hz = 6 bpm
0.60 Hz = 36 bpm
```

虽然后续窗口分类里默认有效呼吸率范围更保守：

```text
6 bpm <= RR <= 24 bpm
```

但滤波上限放到 36 bpm 可以保留短时偏快或噪声判断所需的信息。

如果本机没有安装 `PyWavelets`，程序会退化为：

```text
r_P[k] = BandPass(x_P[k], 0.10 Hz, 0.60 Hz)
```

## 7. 单通道体动特征

对每个通道 `c in {P, R}`，分别计算三类体动特征：

1. 电压快速变化率：用于捕捉短时冲击、翻身、明显姿态切换。
2. 包络变化率：用于捕捉慢一些的接触状态变化、姿态变化、信号幅度突变。
3. 小波细节能量：用于捕捉非呼吸频段的短时宽带扰动。

### 7.1 平滑信号和趋势

短窗平滑：

```text
u_c[k] = MA_{0.3s}(x_c[k])
```

长窗趋势：

```text
g_c[k] = MA_{8s}(x_c[k])
```

去趋势交流分量：

```text
ac_c[k] = x_c[k] - g_c[k]
```

1 s 平滑包络：

```text
env_c[k] = MA_{1s}(|ac_c[k]|)
```

其中 `MA_T` 表示长度为 `T` 秒的居中移动平均。

### 7.2 变化率

电压变化率：

```text
vRate_c[k] = |u_c[k] - u_c[k-1]| * fs
```

包络变化率：

```text
eRate_c[k] = |env_c[k] - env_c[k-1]| * fs
```

乘以 `fs` 的作用是把一阶差分换成近似“每秒变化率”。

### 7.3 小波细节能量

对降采样后的原始通道信号做 DB5 小波分解，重构细节分量：

```text
d_c[k] = waverec(D_L, D_{L-1}, ..., D_1)
```

再计算 1 s 平滑能量：

```text
Ew_c[k] = MA_{1s}(d_c[k]^2)
```

解释：

- 这里的小波不是用来估计呼吸，而是用于体动门控。
- 呼吸主要集中在 0.10-0.60 Hz，翻身、碰撞、接触突变通常会带来更宽频的瞬态能量。
- 稳定心冲击或背景噪声如果长期存在，经过 MAD 标准化后不会持续给很高分。
- 如果想验证它是否必要，可以把 `wavelet_motion_weight` 设为 0，退化为只用平滑电压变化率和包络变化率。

## 8. 鲁棒标准化

体动分数不能直接使用原始变化率，因为不同通道、不同夜晚、不同姿态的绝对幅度可能差别很大。当前使用中位数和 MAD 做鲁棒标准化。

对任意特征序列 `z[k]`：

```text
median_z = median(z)
MAD_z = median(|z - median_z|)
scale_z = 1.4826 * MAD_z + epsilon
```

鲁棒非负 z 分数：

```text
Z_+(z[k]) = max((z[k] - median_z) / scale_z, 0)
```

其中 `1.4826` 是把 MAD 转成近似标准差尺度的常用系数，`epsilon` 用来避免除零。

这样做的意义：

- 中位数和 MAD 不容易被体动尖峰拉偏。
- 只保留正向异常，正常波动不会贡献负分。
- 不需要手工指定某个固定电压阈值。

## 9. 单通道体动分数

每个通道的体动分数定义为：

```text
S_c[k] = 0.4 * Z_+(vRate_c[k])
       + 0.4 * Z_+(eRate_c[k])
       + 0.2 * Z_+(Ew_c[k])
```

其中：

- `vRate_c` 表示电压变化率。
- `eRate_c` 表示包络变化率。
- `Ew_c` 表示小波细节能量。

解释：

- `vRate_c` 对突然冲击和翻身更敏感。
- `eRate_c` 对呼吸幅度突然变强/变弱、姿态导致的压力变化更敏感。
- `Ew_c` 对短时宽带扰动敏感，但默认只占 0.2，避免把稳定 BCG 或高频噪声误当成主要判据。
- 三者融合，避免只依赖单一特征。

## 10. 双通道融合体动分数

PVDF 和压阻的互补性：

- PVDF 对动态微动和冲击敏感。
- 压阻对接触状态、姿态变化和大体动扰动敏感。

注意：当前硬件是 PVDF 与压阻平行排列的两个通道，不是压力分布阵列，所以压阻不再用于呼吸率估计，只用于体动门控。

双通道融合分数：

```text
S_raw[k] = w * S_P[k] + (1 - w) * S_R[k]
```

默认：

```text
w = pvdf_weight = 0.5
```

再做 1 s 平滑：

```text
S[k] = MA_{1s}(S_raw[k])
```

如果想做消融实验：

```text
PVDF-only: w = 1.0
PR-only:   w = 0.0
Fusion:    w = 0.5
```

## 11. 自适应体动阈值

对融合体动分数 `S[k]` 计算鲁棒尺度：

```text
med_S = median(S)
MAD_S = median(|S - med_S|)
scale_S = 1.4826 * MAD_S + epsilon
```

自适应阈值：

```text
T_adapt = med_S + motion_threshold_z * scale_S
```

当前默认：

```text
motion_threshold_z = 3
```

最终阈值：

```text
T = max(3, T_adapt)
```

为什么要取 `max(3, T_adapt)`：

- 当整段数据非常安静时，MAD 可能很小，纯自适应阈值会过低。
- `3` 作为最低鲁棒 z 分数阈值，避免把普通呼吸波动误判为体动。

原始体动候选：

```text
m_raw[k] = 1, if S[k] > T
m_raw[k] = 0, otherwise
```

文件开头和结尾各 `edge_guard_sec = 3 s` 的体动分数置零，避免零相位滤波边缘效应造成误判。

## 12. 体动掩码扩展

体动不是一个瞬时点，翻身或姿态改变会影响前后一小段信号。当前对体动候选做前后扩展：

```text
m[k] = dilation(m_raw[k], radius = motion_dilate_sec * fs)
```

默认：

```text
motion_dilate_sec = 2 s
```

也就是每个检测到的体动点前后各扩展约 2 秒。

## 13. 窗口级特征

每个窗口 `W_j = [s_j, e_j)` 计算以下特征。

### 13.1 坏点比例

```text
bad_fraction_j = mean(bad[k]), k in W_j
```

### 13.2 体动比例

```text
motion_fraction_j = mean(m[k]), k in W_j
```

### 13.3 平均体动分数

```text
mean_motion_score_j = mean(S[k]), k in W_j
```

这个值主要用于观察和输出，不直接决定最终标签。

## 14. 频谱呼吸特征

对窗口内呼吸信号先去线性趋势，然后用 Welch 方法估计功率谱：

```text
PSD_j(f) = Welch(r_P[k]), k in W_j
```

总能量频段：

```text
F_total = [0.05, 1.50] Hz
```

呼吸频段：

```text
F_resp = [0.10, 0.60] Hz
```

总能量：

```text
P_total = integral PSD_j(f) df, f in F_total
```

呼吸频带能量：

```text
P_resp = integral PSD_j(f) df, f in F_resp
```

呼吸频带能量占比：

```text
resp_band_ratio_j = P_resp / (P_total + epsilon)
```

频谱峰呼吸率：

```text
f_peak = argmax_f PSD_j(f), f in F_resp
fft_rr_bpm_j = 60 * f_peak
```

频谱主峰优势度：

```text
dominance_j = PSD_j(f_peak) / (median(PSD_j(f in F_resp)) + epsilon)
```

解释：

- `resp_band_ratio` 越高，说明能量越集中在呼吸频段。
- `dominance` 越高，说明呼吸频谱主峰越突出。
- 稳定呼吸通常会有更高的呼吸频带占比和更明显的主峰。

## 15. 峰间期呼吸率特征

对窗口内呼吸波形同时尝试正峰和负峰检测：

```text
peaks_pos = find_peaks(r_P)
peaks_neg = find_peaks(-r_P)
```

峰检测参数：

```text
min_distance = resp_peak_min_dist_sec * fs
prominence = resp_peak_prom_ratio * amplitude
```

默认：

```text
resp_peak_min_dist_sec = 2.5 s
resp_peak_prom_ratio = 0.15
```

窗口幅度：

```text
amplitude = percentile_95(r_P) - percentile_5(r_P)
```

若检测到至少 3 个峰，峰间期为：

```text
IBI_i = (peak_{i+1} - peak_i) / fs
```

峰间期呼吸率：

```text
peak_rr_bpm_j = 60 / median(IBI_i)
```

峰间期变异系数：

```text
ibi_cv_j = std(IBI_i) / (mean(IBI_i) + epsilon)
```

正峰和负峰各自计算一个候选分数：

```text
candidate_score = median_prominence / amplitude - ibi_cv + 0.03 * peak_count
```

最终选择候选分数更高的一组峰。

优先级：

```text
如果峰间期呼吸率可用，则 rr_bpm = peak_rr_bpm
否则 rr_bpm = fft_rr_bpm
```

也就是说，当前呼吸率估计“峰间距优先，频谱峰兜底”。

## 16. 压阻通道的角色

当前版本不再对压阻通道估计呼吸率，也不再使用 PVDF/压阻呼吸率差异惩罚 SQI。

原因：

- 当前硬件是 PVDF 与压阻平行排列的两个通道，不是压力分布阵列。
- 压阻通道的呼吸波形不一定稳定，强行估计呼吸率容易把不可靠信息引入 SQI。
- 压阻更适合做体动和接触状态辅助判断，例如翻身、姿态改变、身体压力重新加载。

因此当前分工是：

```text
PVDF：呼吸率估计 + 呼吸质量特征 + 体动辅助
压阻：只参与体动门控，不参与呼吸率估计
```

## 17. SQI 评分公式

当前 SQI 是规则评分，不是训练模型。基础思想是：

```text
基础分 50
+ PVDF 稳定呼吸奖励
- 体动、坏点惩罚
```

### 17.1 呼吸频带能量奖励

```text
B_resp = 30 * clip((resp_band_ratio - 0.20) / 0.60, 0, 1)
```

含义：

- `resp_band_ratio <= 0.20` 时不给奖励。
- `resp_band_ratio >= 0.80` 时拿满 30 分。
- 中间线性增加。

### 17.2 频谱主峰奖励

```text
B_dom = 15 * clip((dominance - 1.5) / 8.0, 0, 1)
```

含义：

- 主峰不明显时不给奖励。
- 主峰远高于呼吸频带中位功率时，最多加 15 分。

### 17.3 峰间期稳定性奖励

如果 `ibi_cv` 有效：

```text
B_stable = 10 * clip((0.35 - ibi_cv) / 0.35, 0, 1)
```

如果 `ibi_cv` 无效：

```text
B_stable = 0
```

含义：

- `ibi_cv` 越小，呼吸周期越稳定。
- `ibi_cv = 0` 时最多加 10 分。
- `ibi_cv >= 0.35` 时不给稳定性奖励。

### 17.4 体动惩罚

```text
P_motion = 80 * motion_fraction
```

例如：

- 体动占 10%，扣 8 分。
- 体动占 25%，扣 20 分。

### 17.5 坏点惩罚

```text
P_bad = 120 * bad_fraction
```

坏点越多扣分越重，因为坏点会影响滤波、峰值和频谱。

### 17.6 最终 SQI

```text
SQI_raw = 50
          + B_resp
          + B_dom
          + B_stable
          - P_motion
          - P_bad
```

最终限制到 0-100：

```text
SQI = clip(SQI_raw, 0, 100)
```

## 18. 窗口标签规则

最终标签不是只看 SQI，而是先用硬规则分类。

### 18.1 呼吸率有效性

PVDF 呼吸率必须在：

```text
rr_min_bpm <= rr_P <= rr_max_bpm
```

默认：

```text
6 bpm <= rr_P <= 24 bpm
```

### 18.2 分类顺序

按以下顺序判断：

```text
if bad_fraction > 0.05:
    label = invalid
elif motion_fraction > 0.25:
    label = motion
elif rr_P not in [6, 24] or resp_band_ratio < 0.35:
    label = low_quality
elif motion_fraction > 0.08:
    label = usable
else:
    label = good
```

标签含义：

- `invalid`：ADC 坏点太多，不可靠。
- `motion`：体动占比太高，不适合呼吸率统计。
- `low_quality`：没有稳定呼吸成分或呼吸率不合理。
- `usable`：有轻微体动，但仍可能用于统计。
- `good`：稳定高质量窗口。

### 18.3 是否通过门控

只有满足下面条件才通过门控：

```text
pass_gate = (label in {good, usable}) and (SQI >= 60)
```

也就是说：

- `good/usable` 是硬规则层面的可用。
- `SQI >= 60` 是质量分数层面的可用。
- 两个条件都满足，窗口才进入后续呼吸率统计。

## 19. 输出文件解释

单文件分析输出：

- `quality_windows.csv`：窗口级结果表。
- `summary.json`：完整摘要、配置参数和输出路径。
- `summary.txt`：简版摘要。
- `quality_gating_overview.png`：全段总览图。
- `quality_gating_detail.png`：最强体动附近局部图。

批量分析额外输出：

- `batch_summary.csv`：每个 CSV 的简明汇总。
- `batch_summary.json`：每个 CSV 的完整摘要列表。

`quality_windows.csv` 里的关键列：

- `start_sec`, `end_sec`：窗口起止时间。
- `bad_fraction`：窗口坏点比例。
- `motion_fraction`：窗口体动比例。
- `mean_motion_score`：窗口平均体动分数。
- `mean_pvdf_motion_score`：PVDF 通道窗口平均体动分数。
- `mean_pr_motion_score`：压阻通道窗口平均体动分数。
- `mean_pvdf_voltage_rate_z`, `mean_pr_voltage_rate_z`：两通道电压变化率鲁棒分数。
- `mean_pvdf_env_rate_z`, `mean_pr_env_rate_z`：两通道包络变化率鲁棒分数。
- `mean_pvdf_wavelet_energy_z`, `mean_pr_wavelet_energy_z`：两通道小波细节能量鲁棒分数。
- `pvdf_rr_bpm`：PVDF 呼吸率估计。
- `pvdf_fft_rr_bpm`：PVDF 频谱峰呼吸率。
- `pvdf_resp_band_ratio`：PVDF 呼吸频带能量占比。
- `pvdf_dominance`：PVDF 呼吸频带主峰优势度。
- `pvdf_ibi_cv`：PVDF 峰间期变异系数。
- `label`：窗口标签。
- `pass_gate`：是否通过门控。
- `quality_score`：SQI 分数。

## 20. 当前结果如何解释

已批量处理：

```text
F:\双通道睡眠实验\2026_0413晚\数据
```

输出目录：

```text
outputs\2026_0413_night_revised
```

整晚结果：

- 总时长约 9.77 h。
- 加权窗口通过率约 79.03%。
- 加权体动占比约 13.25%。
- 单文件通过窗口中位呼吸率的中位数约 14.14 bpm。

稳定片段 `20260414_033313.csv`：

- 窗口通过率 97.75%。
- 体动占比 2.58%。
- 通过窗口中位呼吸率 13.73 bpm。

体动较多片段：

- `20260414_003312.csv`：体动占比 36.90%，窗口通过率 38.87%。
- `20260414_010312.csv`：体动占比 41.45%，窗口通过率 34.37%。

这说明当前门控至少具备一个重要性质：它能区分稳定片段和体动较多片段。正式论文里仍需要人工标注或参考设备来证明 precision、recall 和呼吸率误差改善。

## 21. 可以让 ChatGPT 继续想的方向

把这份文档发给网页版 ChatGPT 后，可以重点让它围绕以下问题想 idea：

1. 这个 SQI 公式是否有更清晰的论文表达方式？
2. 体动分数是否应该加入 PVDF/压阻互相关、相干性或幅度稳定性？
3. 消融实验应该怎么设计，才能证明双通道融合有价值？
4. 人工标注应该按秒级、事件级还是窗口级？
5. 门控前后呼吸率稳定性应该用哪些指标？
6. 如何把当前规则方法升级成半监督或轻量模型，同时保持可解释性？

当前最重要的下一步不是重写算法，而是补评价：

```text
人工体动标注 -> precision/recall/F1/IoU
门控前后对比 -> RR std/CV/异常窗口比例
通道消融 -> PVDF-only vs PR-only vs Fusion
```
