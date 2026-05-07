from pathlib import Path
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt


CSV_PATH = Path(r"F:\双通道睡眠实验\2026_0413晚\数据\20260414_033313.csv")
FS_RAW = 500.0
N_RANGE = (1000, 1200)
DOWNSAMPLE_Q = 10 # 降采样到50
FS = FS_RAW / DOWNSAMPLE_Q # 50 Hz
dt = 1 / FS
def clean_adc(x):
    x = np.asarray(x, dtype=float)
    bad = (x <= 0) | (x > 4095) | ~np.isfinite(x)
    x[bad] = np.nan
    x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    return x, bad


df = pd.read_csv(
    CSV_PATH,
    usecols=["sample_index", "pvdf_adc", "pr_adc"]
)

def reconstruct_single(coeffs, keep_index, wavelet_name="db5", out_len=None):
    single_coeffs = [np.zeros_like(c) for c in coeffs]
    single_coeffs[keep_index] = coeffs[keep_index]
    y = pywt.waverec(single_coeffs, wavelet_name)
    if out_len is not None:
        y = y[:out_len]
    return y


def get_pvdf_resp_a5(x):
    coeffs = pywt.wavedec(x, wavelet="db5", level=5)

    cD1 = coeffs[-1]
    noise_sigma = np.median(np.abs(cD1)) / 0.6745
    threshold = noise_sigma * np.sqrt(2 * np.log(len(x)))

    coeffs_th = [coeffs[0].copy()]
    for c in coeffs[1:]:
        coeffs_th.append(pywt.threshold(c, value=threshold, mode="soft"))

    a5 = reconstruct_single(coeffs_th, 0, "db5", len(x))
    return a5

def rolling_median(x, win):
    return pd.Series(x).rolling(win, center=True, min_periods=1).median().to_numpy()


def rolling_rms(x, win):
    kernel = np.ones(win) / win
    return np.sqrt(signal.convolve(x * x, kernel, mode="same"))

sample_index = df["sample_index"].to_numpy(dtype=float)
t = (sample_index - sample_index[0]) / FS_RAW

# 数据清洗和插值、降采样、切片
pvdf, bad_pvdf = clean_adc(df["pvdf_adc"].to_numpy()) 
pr, bad_pr = clean_adc(df["pr_adc"].to_numpy())
pvdf = pvdf / 4095 * 3.3
pr = pr / 4095 * 3.3
pvdf_ds = signal.decimate(pvdf, q=DOWNSAMPLE_Q, ftype="fir", zero_phase=True)
pr_ds = signal.decimate(pr, q=DOWNSAMPLE_Q, ftype="fir", zero_phase=True)
t_ds = np.arange(len(pvdf_ds)) / FS
start, end = N_RANGE
idx = (t_ds >= start) & (t_ds <= end)
t_seg = t_ds[idx]
pvdf_seg = pvdf_ds[idx]
pr_seg = pr_ds[idx]

pvdf_resp_a5 = get_pvdf_resp_a5(pvdf_seg)   # 小波提取呼吸信号

base_win = int(5 * FS)   # 5 秒窗口
if base_win % 2 == 0:   # 确保窗口大小为奇数，以便 rolling_median 居中对齐
    base_win += 1

pr_base = rolling_median(pr_seg, base_win)

def moving_average(x, win):
    return pd.Series(x).rolling(win, center=True, min_periods=1).mean().to_numpy()


# PR: 用短窗和长窗的差描述基线台阶变化
pr_short = moving_average(pr_seg, win=int(1 * FS))     # 2 s
pr_long = moving_average(pr_seg, win=int(5 * FS))     # 12 s
# pr_shift = pr_short - pr_long
pr_moving = moving_average(pr_seg, win=int(1 * FS))
pr_diff =np.diff(pr_moving,prepend=pr_moving[0])/0.02
# pr_slope = np.abs(np.diff(pr_long, prepend=pr_long[0]))/0.02
pr_center = moving_average(pr_seg, win=int(8 * FS))
pr_detrend = pr_seg - pr_center
pr_slope = moving_average(np.abs(pr_detrend), win=int(1.0 * FS))

# PVDF: 不再看单点幅值，改看包络和包络变化
pvdf_center = moving_average(pvdf_resp_a5, win=int(8 * FS))
pvdf_detrend = pvdf_resp_a5 - pvdf_center
pvdf_env = moving_average(np.abs(pvdf_detrend), win=int(1.0 * FS))
pvdf_env_slope = np.abs(np.diff(pvdf_env, prepend=pvdf_env[0]))/0.02

# 两个通道各自标准化到相近量级
eps = 1e-12
pr_diff_score = np.abs(pr_diff) / (np.percentile(np.abs(pr_diff), 95) + eps)
pr_slope_score = pr_slope / (np.percentile(pr_slope, 95) + eps)
pvdf_env_score = pvdf_env / (np.percentile(pvdf_env, 95) + eps)
pvdf_slope_score = pvdf_env_slope / (np.percentile(pvdf_env_slope, 99.9) + eps)

# # 融合分数：PR 更看趋势，PVDF 更看动态
# motion_score = (
#     0.35 * pr_shift_score +
#     0.15 * pr_slope_score +
#     0.25 * pvdf_env_score +
#     0.25 * pvdf_slope_score
# )

# motion_score_smooth = moving_average(motion_score, win=int(1.5 * FS))
# MOTION_TH = np.percentile(motion_score_smooth, 95)
# # MOTION_TH = 1.0
# motion_mask = motion_score_smooth > MOTION_TH

# def robust_z(x):
#     med = np.median(x)
#     mad = np.median(np.abs(x - med)) + 1e-12
#     return (x - med) / (1.4826 * mad)

# pr_diff_z = robust_z(np.abs(pr_diff))
# pr_slope_z = robust_z(pr_slope)
# pvdf_env_z = robust_z(pvdf_env)
# pvdf_env_slope_z = robust_z(pvdf_env_slope)

# motion_score = (
#     0.35 * pr_diff_z +
#     0.15 * pr_slope_z +
#     0.25 * pvdf_env_z +
#     0.25 * pvdf_env_slope_z
# )

# motion_score_smooth = moving_average(motion_score, win=int(0.1 * FS))

# score_med = np.median(motion_score_smooth)
# score_mad = np.median(np.abs(motion_score_smooth - score_med)) + 1e-12
# MOTION_TH = score_med + 10 * 1.4826 * score_mad

# motion_mask = motion_score_smooth > MOTION_TH


plt.figure(figsize=(14, 12))

plt.subplot(5, 1, 1)
plt.plot(t_seg, pvdf_resp_a5, linewidth=0.8)
plt.title("PVDF respiration")
plt.ylabel("V")

plt.subplot(5, 1, 2)
plt.plot(t_seg, pr_seg, linewidth=0.8, label="PR")
plt.plot(t_seg, pr_short, linewidth=1.0, label="PR short")
plt.plot(t_seg, pr_long, linewidth=1.0, label="PR long")
plt.title("PR trend")
plt.ylabel("V")
plt.legend()

plt.subplot(5, 1, 3)
plt.plot(t_seg, pr_diff, linewidth=0.8, label="PR diff")
plt.plot(t_seg, pr_slope, linewidth=0.8, label="PR slope")
plt.title("PR diff and slope")
plt.ylabel("score")
plt.legend()

# plt.subplot(5, 1, 4)
# plt.plot(t_seg, pvdf_env, linewidth=0.8, label="PVDF env")
# plt.plot(t_seg, pvdf_env_slope, linewidth=0.8, label="PVDF env slope")
# plt.title("PVDF envelope")
# plt.ylabel("score")
# plt.legend()

# plt.subplot(5, 1, 5)
# plt.plot(t_seg, motion_score_smooth, linewidth=1.0, color="tab:red", label="motion score")
# plt.axhline(MOTION_TH, color="black", linestyle="--")
# plt.plot(
#     t_seg[motion_mask],
#     motion_score_smooth[motion_mask],
#     "o",
#     color="green",
#     markersize=3,
#     label="motion mask"
# )
# plt.title("Fused motion score")
# plt.ylabel("score")
# plt.xlabel("Time (s)")
# plt.legend()

# plt.tight_layout()
# plt.show()


def abs_rate(x):
    return np.abs(np.diff(x, prepend=x[0])) / dt

def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad)

# PVDF
pvdf_smooth = moving_average(pvdf_resp_a5, win=int(0.3 * FS))
pvdf_trend = moving_average(pvdf_resp_a5, win=int(8 * FS))
pvdf_ac = pvdf_resp_a5 - pvdf_trend
pvdf_env = moving_average(np.abs(pvdf_ac), win=int(1.0 * FS))
pvdf_env_rate = abs_rate(pvdf_env)
pvdf_voltage_rate = abs_rate(pvdf_smooth)

# PR
pr_smooth = moving_average(pr_seg, win=int(0.3 * FS))
pr_trend = moving_average(pr_seg, win=int(8 * FS))
pr_ac = pr_seg - pr_trend
pr_env = moving_average(np.abs(pr_ac), win=int(1.0 * FS))
pr_env_rate = abs_rate(pr_env)
pr_voltage_rate = abs_rate(pr_smooth)

# 标准化
pvdf_env_rate_z = robust_z(pvdf_env_rate)
pvdf_voltage_rate_z = robust_z(pvdf_voltage_rate)
pr_env_rate_z = robust_z(pr_env_rate)
pr_voltage_rate_z = robust_z(pr_voltage_rate)

# 每个通道内部：包络变化率 + 电压变化率，各一半
pvdf_score = 0.5 * pvdf_env_rate_z + 0.5 * pvdf_voltage_rate_z
pr_score = 0.5 * pr_env_rate_z + 0.5 * pr_voltage_rate_z

# 双通道先做五五开对照
final_score_equal = 0.5 * pvdf_score + 0.5 * pr_score

# 也建议同时保留主从版
final_score_master = 0.8 * pvdf_score + 0.2 * pr_score

def adaptive_threshold(x, k=3.0):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return med + k * 1.4826 * mad

final_score = moving_average(final_score_equal, win=int(1.0 * FS))
FINAL_TH = adaptive_threshold(final_score, k=3.0)
motion_mask = final_score > FINAL_TH

master_score = moving_average(final_score_master, win=int(1.0 * FS))
MASTER_TH = adaptive_threshold(master_score, k=3.0)
motion_mask_master = master_score > MASTER_TH

plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(t_seg, final_score, linewidth=1.0, color="tab:red", label="final score")
plt.axhline(FINAL_TH, color="black", linestyle="--", label="threshold")
plt.plot(
    t_seg[motion_mask],
    final_score[motion_mask],
    "o",
    color="green",
    markersize=3,
    label="motion mask"
) 
plt.title("Final Motion Score")
plt.ylabel("score")
plt.xlabel("Time (s)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_seg, master_score, linewidth=0.8, label="Master Score")
plt.axhline(MASTER_TH, color="black", linestyle="--", label="threshold")
plt.plot(
    t_seg[motion_mask_master],
    master_score[motion_mask_master],
    "o",
    color="green",
    markersize=3,
    label="motion mask"
)
plt.title("Master Motion Score")
plt.ylabel("score")
plt.xlabel("Time (s)")
plt.legend()
plt.tight_layout()
plt.show()