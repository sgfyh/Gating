from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

try:
    import pywt
except ImportError:  # pragma: no cover - only used on machines without PyWavelets.
    pywt = None


DEFAULT_CSV = Path(r"F:\双通道睡眠实验\2026_0413晚\数据\20260414_033313.csv")


@dataclass
class GateConfig:
    """质量门控的全部参数。

    这里的参数不是训练出来的模型权重，而是一套可解释的工程阈值。
    默认值适合当前 500 Hz 床下 PVDF + 压阻双通道数据；正式论文里可以做
    参数敏感性分析，例如比较 20 s、30 s、60 s 窗口。
    """

    # 原始采样率和降采样设置：500 Hz / 10 = 50 Hz，足够覆盖呼吸和体动特征。
    fs_raw: float = 500.0
    downsample_q: int = 10

    # ADC 转电压参数。12 bit ADC 常见范围是 0-4095，参考电压按 3.3 V 计算。
    adc_max: float = 4095.0
    adc_vref: float = 3.3

    # 窗口级 SQI 参数。window_sec 越长越稳定但响应越慢；step_sec 越小结果越密。
    window_sec: float = 30.0
    step_sec: float = 5.0

    # 体动门控参数：超过阈值的点被认为是体动，并向前后扩展 motion_dilate_sec 秒。
    motion_dilate_sec: float = 2.0
    motion_threshold_z: float = 3.0

    # 双通道融合权重。0.5 表示 PVDF 和压阻对体动判断同等重要。
    # 注意：压阻只用于体动门控，不再用于呼吸率估计或 SQI 的呼吸一致性惩罚。
    pvdf_weight: float = 0.5

    # 体动分数的三个特征级权重。
    # voltage_rate/envelope_rate 使用 PVDF+压阻双通道融合；
    # wavelet_motion 只使用 PVDF，因为压阻高频更可能是电路噪声而不是体动。
    voltage_rate_weight: float = 0.40
    envelope_rate_weight: float = 0.40
    wavelet_motion_weight: float = 0.20

    # 呼吸频段。0.10-0.60 Hz 对应 6-36 bpm，用于频谱质量评价和带通滤波。
    resp_low_hz: float = 0.10
    resp_high_hz: float = 0.60

    # 总能量参考频段。呼吸频带能量 / 总能量越高，说明窗口越像稳定呼吸。
    total_low_hz: float = 0.05
    total_high_hz: float = 1.50

    # 呼吸峰检测参数。最小峰间距 2.5 s 等价于不允许超过约 24 bpm 的峰间距估计。
    resp_peak_min_dist_sec: float = 2.5
    resp_peak_prom_ratio: float = 0.15
    rr_min_bpm: float = 6.0
    rr_max_bpm: float = 24.0

    # 文件开头/结尾零相位滤波容易有边缘伪峰，这几秒不参与体动判断。
    edge_guard_sec: float = 3.0

    # 窗口分类阈值。
    bad_fraction_max: float = 0.05       # 异常 ADC 比例超过 5%，窗口直接判 invalid。
    motion_reject_fraction: float = 0.25 # 体动占窗口超过 25%，窗口判 motion。
    motion_warn_fraction: float = 0.08   # 体动占 8%-25%，窗口还能用但标成 usable。
    min_resp_band_ratio: float = 0.35    # 呼吸频带能量占比低于 35%，窗口判 low_quality。
    min_pass_quality: float = 60.0       # SQI 低于 60，即使标签可用也不通过门控。

    # 体动分段和事件候选参数。这里输出的是候选事件，不是临床诊断结论。
    min_quiet_segment_sec: float = 10.0  # 两次体动之间少于 10 s，不够判事件，跳过。
    min_event_sec: float = 10.0          # 呼吸暂停/低通气候选的最短持续时间。
    apnea_drop_fraction: float = 0.90    # 相对基线振幅下降 >=90%，偏向暂停候选。
    hypopnea_drop_fraction: float = 0.30 # 相对基线振幅下降 >=30%，偏向低通气候选。
    baseline_min_frames: int = 5         # 建基线至少希望有多少个正常呼吸帧。

    @property
    def fs(self) -> float:
        return self.fs_raw / self.downsample_q


def clean_adc(values: pd.Series, cfg: GateConfig) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    bad = (~np.isfinite(x)) | (x < 0) | (x > cfg.adc_max)
    if np.all(bad):
        raise ValueError("All ADC samples are invalid.")

    x = x.copy()
    x[bad] = np.nan
    x = pd.Series(x).interpolate(limit_direction="both").to_numpy(dtype=float)
    x = np.clip(x, 0, cfg.adc_max)
    return x / cfg.adc_max * cfg.adc_vref, bad


def downsample_bad_mask(bad: np.ndarray, q: int, target_len: int) -> np.ndarray:
    target_raw_len = target_len * q
    if len(bad) < target_raw_len:
        bad = np.pad(bad, (0, target_raw_len - len(bad)), constant_values=False)
    trimmed = bad[:target_raw_len]
    return trimmed.reshape(target_len, q).mean(axis=1)


def load_dual_channel_csv(csv_path: Path, cfg: GateConfig) -> dict[str, np.ndarray]:
    header = pd.read_csv(csv_path, nrows=0)
    required = {"pvdf_adc", "pr_adc"}
    missing = required.difference(header.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    usecols = ["pvdf_adc", "pr_adc"]
    if "sample_index" in header.columns:
        usecols.insert(0, "sample_index")

    df = pd.read_csv(csv_path, usecols=usecols)
    pvdf, bad_pvdf = clean_adc(df["pvdf_adc"], cfg)
    pr, bad_pr = clean_adc(df["pr_adc"], cfg)

    pvdf_ds = signal.decimate(pvdf, q=cfg.downsample_q, ftype="fir", zero_phase=True)
    pr_ds = signal.decimate(pr, q=cfg.downsample_q, ftype="fir", zero_phase=True)
    n = min(len(pvdf_ds), len(pr_ds))
    pvdf_ds = pvdf_ds[:n]
    pr_ds = pr_ds[:n]

    bad_ds = np.maximum(
        downsample_bad_mask(bad_pvdf, cfg.downsample_q, n),
        downsample_bad_mask(bad_pr, cfg.downsample_q, n),
    )

    t = np.arange(n, dtype=float) / cfg.fs
    return {
        "t": t,
        "pvdf": pvdf_ds,
        "pr": pr_ds,
        "bad_fraction": bad_ds,
        "raw_samples": np.array([len(df)], dtype=int),
    }


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    win = max(int(win), 1)
    if win <= 1:
        return np.asarray(x, dtype=float).copy()
    return pd.Series(x).rolling(win, center=True, min_periods=1).mean().to_numpy()


def robust_scale(x: np.ndarray) -> tuple[float, float]:
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    return med, 1.4826 * mad + 1e-12


def robust_positive_z(x: np.ndarray) -> np.ndarray:
    med, scale = robust_scale(x)
    return np.clip((x - med) / scale, 0.0, None)


def abs_rate(x: np.ndarray, fs: float) -> np.ndarray:
    return np.abs(np.diff(x, prepend=x[0])) * fs


def reconstruct_single(coeffs: list[np.ndarray], keep_index: int, wavelet: str, out_len: int) -> np.ndarray:
    selected = [np.zeros_like(c) for c in coeffs]
    selected[keep_index] = coeffs[keep_index]
    return pywt.waverec(selected, wavelet)[:out_len]


def bandpass(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    high_hz = min(high_hz, 0.95 * nyq)
    if low_hz <= 0 or high_hz <= low_hz:
        raise ValueError("Invalid bandpass limits.")
    sos = signal.butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def extract_pvdf_respiration(pvdf: np.ndarray, cfg: GateConfig) -> np.ndarray:
    if pywt is None:
        return bandpass(pvdf, cfg.fs, cfg.resp_low_hz, cfg.resp_high_hz)

    wavelet = "db5"
    level = min(5, pywt.dwt_max_level(len(pvdf), pywt.Wavelet(wavelet).dec_len))
    coeffs = pywt.wavedec(pvdf, wavelet=wavelet, level=level)

    cD1 = coeffs[-1]
    noise_sigma = np.median(np.abs(cD1)) / 0.6745 + 1e-12
    threshold = noise_sigma * np.sqrt(2 * np.log(len(pvdf)))
    coeffs_th = [coeffs[0].copy()]
    for c in coeffs[1:]:
        coeffs_th.append(pywt.threshold(c, value=threshold, mode="soft"))

    resp = reconstruct_single(coeffs_th, 0, wavelet, len(pvdf))
    return bandpass(resp, cfg.fs, cfg.resp_low_hz, cfg.resp_high_hz)


def motion_feature_weights(cfg: GateConfig) -> np.ndarray:
    weights = np.array(
        [cfg.voltage_rate_weight, cfg.envelope_rate_weight, cfg.wavelet_motion_weight],
        dtype=float,
    )
    weights = np.clip(weights, 0.0, None)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        return np.array([0.5, 0.5, 0.0], dtype=float)
    return weights / weight_sum


def wavelet_motion_energy_z(x: np.ndarray, cfg: GateConfig) -> np.ndarray:
    """用 PVDF 小波细节能量描述非呼吸瞬态扰动。

    它不是用来估计呼吸，而是补充体动门控：稳定心冲击/噪声如果能量长期存在，
    鲁棒 z 分数不会很高；翻身、碰撞、接触状态突变通常会让细节能量短时升高。
    """
    if cfg.wavelet_motion_weight <= 0:
        return np.zeros_like(x, dtype=float)

    if pywt is None:
        high = x - moving_average(x, int(round(1.0 * cfg.fs)))
    else:
        wavelet = "db5"
        level = min(4, pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len))
        if level < 1:
            high = x - moving_average(x, int(round(1.0 * cfg.fs)))
        else:
            coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
            detail_coeffs = [np.zeros_like(coeffs[0])] + [c.copy() for c in coeffs[1:]]
            high = pywt.waverec(detail_coeffs, wavelet)[: len(x)]

    detail_energy = moving_average(high * high, int(round(1.0 * cfg.fs)))
    return robust_positive_z(detail_energy)


def channel_motion_features(x: np.ndarray, cfg: GateConfig, *, include_wavelet: bool) -> dict[str, np.ndarray]:
    """计算单通道体动特征。

    电压变化率和包络变化率有相关性，但不完全重复：
    - 电压变化率更像“一阶导数”，对瞬时冲击敏感；
    - 包络变化率看的是交流幅度变化，对接触状态、姿态改变更敏感；
    - 小波细节能量只在 PVDF 上计算，因为 PVDF 对振动/冲击有高频响应基础。
    """
    fs = cfg.fs
    smooth = moving_average(x, int(round(0.3 * fs)))
    trend = moving_average(x, int(round(8.0 * fs)))
    ac = x - trend
    env = moving_average(np.abs(ac), int(round(1.0 * fs)))

    env_rate_z = robust_positive_z(abs_rate(env, fs))
    voltage_rate_z = robust_positive_z(abs_rate(smooth, fs))
    if include_wavelet:
        wavelet_energy_z = wavelet_motion_energy_z(x, cfg)
    else:
        wavelet_energy_z = np.zeros_like(x, dtype=float)

    return {
        "env": env,
        "env_rate_z": env_rate_z,
        "voltage_rate_z": voltage_rate_z,
        "wavelet_energy_z": wavelet_energy_z,
    }


def expand_mask(mask: np.ndarray, radius_samples: int) -> np.ndarray:
    if radius_samples <= 0:
        return mask.astype(bool)
    kernel = np.ones(2 * radius_samples + 1, dtype=int)
    return signal.convolve(mask.astype(int), kernel, mode="same") > 0


def compute_motion_gate(pvdf: np.ndarray, pr: np.ndarray, cfg: GateConfig) -> dict[str, np.ndarray | float]:
    """计算逐采样点体动门控。

    先计算三个物理含义不同的特征：
    1. PVDF+压阻电压快速变化率，用来抓短时冲击和翻身；
    2. PVDF+压阻包络变化率，用来抓慢一点的姿态/接触状态变化；
    3. PVDF 小波细节能量，用来抓压电材料可感知的高频/宽带扰动。

    压阻不计算小波细节能量，避免把压阻高频电路噪声当成体动信号。
    最后用 median + k*MAD 的鲁棒阈值判体动。
    """
    pvdf_motion = channel_motion_features(pvdf, cfg, include_wavelet=True)
    pr_motion = channel_motion_features(pr, cfg, include_wavelet=False)

    weights = motion_feature_weights(cfg)
    pvdf_w = float(np.clip(cfg.pvdf_weight, 0.0, 1.0))
    pr_w = 1.0 - pvdf_w

    voltage_score = pvdf_w * pvdf_motion["voltage_rate_z"] + pr_w * pr_motion["voltage_rate_z"]
    envelope_score = pvdf_w * pvdf_motion["env_rate_z"] + pr_w * pr_motion["env_rate_z"]
    wavelet_score = pvdf_motion["wavelet_energy_z"]

    fused = weights[0] * voltage_score + weights[1] * envelope_score + weights[2] * wavelet_score
    fused = moving_average(fused, int(round(1.0 * cfg.fs)))

    # Diagnostic channel scores; PR has no wavelet term by design.
    pvdf_score = (
        weights[0] * pvdf_motion["voltage_rate_z"]
        + weights[1] * pvdf_motion["env_rate_z"]
        + weights[2] * pvdf_motion["wavelet_energy_z"]
    )
    pr_score = weights[0] * pr_motion["voltage_rate_z"] + weights[1] * pr_motion["env_rate_z"]

    edge = int(round(cfg.edge_guard_sec * cfg.fs))
    if edge > 0 and len(fused) > 2 * edge:
        for arr in (pvdf_score, pr_score, voltage_score, envelope_score, wavelet_score, fused):
            arr[:edge] = 0.0
            arr[-edge:] = 0.0

    med, scale = robust_scale(fused)
    adaptive_th = med + cfg.motion_threshold_z * scale
    threshold = max(cfg.motion_threshold_z, adaptive_th)
    raw_motion = fused > threshold
    motion = expand_mask(raw_motion, int(round(cfg.motion_dilate_sec * cfg.fs)))
    return {
        "pvdf_score": pvdf_score,
        "pr_score": pr_score,
        "voltage_score": voltage_score,
        "envelope_score": envelope_score,
        "pvdf_wavelet_score": wavelet_score,
        "pvdf_voltage_rate_z": pvdf_motion["voltage_rate_z"],
        "pvdf_env_rate_z": pvdf_motion["env_rate_z"],
        "pvdf_wavelet_energy_z": pvdf_motion["wavelet_energy_z"],
        "pr_voltage_rate_z": pr_motion["voltage_rate_z"],
        "pr_env_rate_z": pr_motion["env_rate_z"],
        "motion_score": fused,
        "motion_threshold": float(threshold),
        "raw_motion_mask": raw_motion,
        "motion_mask": motion,
    }


def spectral_resp_features(x: np.ndarray, cfg: GateConfig) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = signal.detrend(x, type="linear")
    if len(x) < int(8 * cfg.fs):
        return {"fft_rr_bpm": np.nan, "resp_band_ratio": np.nan, "dominance": np.nan}

    freqs, psd = signal.welch(
        x,
        fs=cfg.fs,
        window="hann",
        nperseg=len(x),
        noverlap=0,
        detrend=False,
        scaling="density",
    )
    total_mask = (freqs >= cfg.total_low_hz) & (freqs <= cfg.total_high_hz)
    resp_mask = (freqs >= cfg.resp_low_hz) & (freqs <= cfg.resp_high_hz)
    if not np.any(resp_mask) or not np.any(total_mask):
        return {"fft_rr_bpm": np.nan, "resp_band_ratio": np.nan, "dominance": np.nan}

    total_power = float(np.trapz(psd[total_mask], freqs[total_mask])) + 1e-12
    resp_power = float(np.trapz(psd[resp_mask], freqs[resp_mask]))
    resp_psd = psd[resp_mask]
    resp_freqs = freqs[resp_mask]
    peak_idx = int(np.argmax(resp_psd))
    rr_bpm = float(resp_freqs[peak_idx] * 60.0)
    dominance = float(resp_psd[peak_idx] / (np.median(resp_psd) + 1e-12))
    return {
        "fft_rr_bpm": rr_bpm,
        "resp_band_ratio": resp_power / total_power,
        "dominance": dominance,
    }


def peak_resp_features(x: np.ndarray, cfg: GateConfig) -> dict[str, float]:
    x = signal.detrend(np.asarray(x, dtype=float), type="linear")
    amp = float(np.nanpercentile(x, 95) - np.nanpercentile(x, 5))
    if len(x) < int(8 * cfg.fs) or amp <= 1e-12:
        return {"peak_rr_bpm": np.nan, "peak_count": 0.0, "ibi_cv": np.nan}

    min_distance = max(1, int(round(cfg.resp_peak_min_dist_sec * cfg.fs)))
    prominence = cfg.resp_peak_prom_ratio * amp
    candidates = []
    for sign in (1.0, -1.0):
        peaks, props = signal.find_peaks(sign * x, distance=min_distance, prominence=prominence)
        if len(peaks) < 3:
            continue
        periods = np.diff(peaks) / cfg.fs
        rr_bpm = 60.0 / float(np.median(periods))
        ibi_cv = float(np.std(periods) / (np.mean(periods) + 1e-12))
        med_prom = float(np.median(props.get("prominences", [0.0])) / (amp + 1e-12))
        score = med_prom - ibi_cv + 0.03 * len(peaks)
        candidates.append((score, rr_bpm, len(peaks), ibi_cv))

    if not candidates:
        return {"peak_rr_bpm": np.nan, "peak_count": 0.0, "ibi_cv": np.nan}

    _, rr_bpm, peak_count, ibi_cv = max(candidates, key=lambda item: item[0])
    return {"peak_rr_bpm": float(rr_bpm), "peak_count": float(peak_count), "ibi_cv": float(ibi_cv)}


def respiration_features(x: np.ndarray, cfg: GateConfig) -> dict[str, float]:
    spec = spectral_resp_features(x, cfg)
    peaks = peak_resp_features(x, cfg)
    rr_bpm = peaks["peak_rr_bpm"]
    rr_method = 1.0
    if not np.isfinite(rr_bpm):
        rr_bpm = spec["fft_rr_bpm"]
        rr_method = 0.0
    return {
        "rr_bpm": float(rr_bpm),
        "rr_method_peak": rr_method,
        **spec,
        **peaks,
    }


def iter_windows(n: int, cfg: GateConfig) -> list[tuple[int, int]]:
    win = int(round(cfg.window_sec * cfg.fs))
    step = int(round(cfg.step_sec * cfg.fs))
    if n < win:
        return [(0, n)]
    starts = list(range(0, n - win + 1, step))
    if starts[-1] + win < n:
        starts.append(n - win)
    return [(s, min(s + win, n)) for s in starts]


def classify_window(row: dict[str, float], cfg: GateConfig) -> tuple[str, bool, float]:
    """把一个 30 s 窗口的特征转换成 label、pass_gate 和 SQI。

    SQI 的思路是“先给基础分，再奖励 PVDF 稳定呼吸特征，最后扣除体动/坏点”。
    当前公式：

        SQI = 50
              + 呼吸频带能量奖励，最多 +30
              + 频谱主峰突出程度奖励，最多 +15
              + 呼吸峰间期稳定性奖励，最多 +10
              - 体动占比惩罚，最多按 80 * motion_fraction 扣
              - 坏点占比惩罚，最多按 120 * bad_fraction 扣

    最后把 SQI 限制在 0-100。分类不是只看 SQI：先用硬规则判 invalid/motion/
    low_quality/usable/good，再要求 good 或 usable 且 SQI >= min_pass_quality 才通过。
    压阻不参与这里的呼吸质量评分，因为当前硬件是平行双通道，不是压力分布阵列。
    """
    rr_ok = np.isfinite(row["pvdf_rr_bpm"]) and cfg.rr_min_bpm <= row["pvdf_rr_bpm"] <= cfg.rr_max_bpm

    resp_band_bonus = 30.0 * np.clip((row["pvdf_resp_band_ratio"] - 0.20) / 0.60, 0.0, 1.0)
    dominance_bonus = 15.0 * np.clip((row["pvdf_dominance"] - 1.5) / 8.0, 0.0, 1.0)
    stability_bonus = 0.0
    if np.isfinite(row["pvdf_ibi_cv"]):
        stability_bonus = 10.0 * np.clip((0.35 - row["pvdf_ibi_cv"]) / 0.35, 0.0, 1.0)

    motion_penalty = 80.0 * row["motion_fraction"]
    bad_penalty = 120.0 * row["bad_fraction"]

    quality = (
        50.0
        + resp_band_bonus
        + dominance_bonus
        + stability_bonus
        - motion_penalty
        - bad_penalty
    )
    quality = float(np.clip(quality, 0.0, 100.0))

    if row["bad_fraction"] > cfg.bad_fraction_max:
        label = "invalid"
    elif row["motion_fraction"] > cfg.motion_reject_fraction:
        label = "motion"
    elif (not rr_ok) or row["pvdf_resp_band_ratio"] < cfg.min_resp_band_ratio:
        label = "low_quality"
    elif row["motion_fraction"] > cfg.motion_warn_fraction:
        label = "usable"
    else:
        label = "good"

    pass_gate = label in {"good", "usable"} and quality >= cfg.min_pass_quality
    return label, pass_gate, quality


def build_window_table(
    data: dict[str, np.ndarray],
    resp: np.ndarray,
    gate: dict[str, np.ndarray | float],
    cfg: GateConfig,
) -> pd.DataFrame:
    """生成窗口级结果表。

    每一行对应一个滑动窗口，默认是 30 s 窗、5 s 步长。窗口大小可以放宽：
    - 20 s：更敏感，能更快定位短时体动，但呼吸频率估计会更抖；
    - 30 s：当前折中默认值；
    - 60 s：更稳定，适合整晚统计，但体动定位会更粗。
    """
    rows: list[dict[str, float | str | bool]] = []
    for start, end in iter_windows(len(data["t"]), cfg):
        pvdf_feat = respiration_features(resp[start:end], cfg)

        row: dict[str, float | str | bool] = {
            "start_sec": float(data["t"][start]),
            "end_sec": float(data["t"][end - 1]) if end > start else float(data["t"][start]),
            "bad_fraction": float(np.mean(data["bad_fraction"][start:end])),
            "motion_fraction": float(np.mean(gate["motion_mask"][start:end])),
            "mean_motion_score": float(np.mean(gate["motion_score"][start:end])),
            "mean_pvdf_motion_score": float(np.mean(gate["pvdf_score"][start:end])),
            "mean_pr_motion_score": float(np.mean(gate["pr_score"][start:end])),
            "mean_voltage_score": float(np.mean(gate["voltage_score"][start:end])),
            "mean_envelope_score": float(np.mean(gate["envelope_score"][start:end])),
            "mean_pvdf_wavelet_score": float(np.mean(gate["pvdf_wavelet_score"][start:end])),
            "mean_pvdf_voltage_rate_z": float(np.mean(gate["pvdf_voltage_rate_z"][start:end])),
            "mean_pvdf_env_rate_z": float(np.mean(gate["pvdf_env_rate_z"][start:end])),
            "mean_pvdf_wavelet_energy_z": float(np.mean(gate["pvdf_wavelet_energy_z"][start:end])),
            "mean_pr_voltage_rate_z": float(np.mean(gate["pr_voltage_rate_z"][start:end])),
            "mean_pr_env_rate_z": float(np.mean(gate["pr_env_rate_z"][start:end])),
            "pvdf_rr_bpm": pvdf_feat["rr_bpm"],
            "pvdf_fft_rr_bpm": pvdf_feat["fft_rr_bpm"],
            "pvdf_resp_band_ratio": pvdf_feat["resp_band_ratio"],
            "pvdf_dominance": pvdf_feat["dominance"],
            "pvdf_peak_count": pvdf_feat["peak_count"],
            "pvdf_ibi_cv": pvdf_feat["ibi_cv"],
            "pvdf_rr_method_peak": pvdf_feat["rr_method_peak"],
        }
        label, pass_gate, quality = classify_window(row, cfg)
        row["label"] = label
        row["pass_gate"] = pass_gate
        row["quality_score"] = quality
        rows.append(row)

    return pd.DataFrame(rows)


def build_quiet_segments(gate: dict[str, np.ndarray | float], cfg: GateConfig) -> pd.DataFrame:
    """把逐点体动掩码转换成安静段。

    安静段是两次体动之间的连续非体动区间。它是后续呼吸率时间序列和事件候选
    检测的基本单位，避免固定 30 s 窗口把一个暂停事件切碎。
    """
    motion = np.asarray(gate["motion_mask"], dtype=bool)
    quiet = ~motion
    rows: list[dict[str, int | float | bool]] = []
    start = None
    segment_id = 0
    for i, is_quiet in enumerate(quiet):
        if is_quiet and start is None:
            start = i
        elif (not is_quiet) and start is not None:
            end = i
            duration = (end - start) / cfg.fs
            rows.append(
                {
                    "segment_id": segment_id,
                    "start_idx": int(start),
                    "end_idx": int(end),
                    "start_sec": float(start / cfg.fs),
                    "end_sec": float(end / cfg.fs),
                    "duration_sec": float(duration),
                    "analyzable": bool(duration >= cfg.min_quiet_segment_sec),
                }
            )
            segment_id += 1
            start = None

    if start is not None:
        end = len(quiet)
        duration = (end - start) / cfg.fs
        rows.append(
            {
                "segment_id": segment_id,
                "start_idx": int(start),
                "end_idx": int(end),
                "start_sec": float(start / cfg.fs),
                "end_sec": float(end / cfg.fs),
                "duration_sec": float(duration),
                "analyzable": bool(duration >= cfg.min_quiet_segment_sec),
            }
        )

    return pd.DataFrame(rows)


def attach_window_segments(windows: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    if windows.empty:
        return windows
    out = windows.copy()
    out["segment_id"] = -1
    out["segment_duration_sec"] = np.nan
    if segments.empty:
        return out

    centers = 0.5 * (out["start_sec"].to_numpy() + out["end_sec"].to_numpy())
    for seg in segments.itertuples(index=False):
        mask = (centers >= seg.start_sec) & (centers < seg.end_sec) & bool(seg.analyzable)
        out.loc[mask, "segment_id"] = int(seg.segment_id)
        out.loc[mask, "segment_duration_sec"] = float(seg.duration_sec)
    return out


def window_quality_at_time(windows: pd.DataFrame, time_sec: float) -> tuple[float, bool]:
    if windows.empty:
        return np.nan, False
    mask = (windows["start_sec"] <= time_sec) & (time_sec <= windows["end_sec"])
    if not mask.any():
        return np.nan, False
    local = windows.loc[mask]
    return float(local["quality_score"].max()), bool(local["pass_gate"].any())


def build_breath_frames(resp: np.ndarray, segments: pd.DataFrame, windows: pd.DataFrame, cfg: GateConfig) -> pd.DataFrame:
    """在安静段内按谷到谷建立呼吸帧。

    这里不使用 FFT 兜底。事件检测语境下“找不到可靠谷”本身就是信号异常线索，
    不应为了给出呼吸率而强行凑数。
    """
    rows: list[dict[str, float | int | bool]] = []
    if segments.empty:
        return pd.DataFrame(rows)

    min_distance = max(1, int(round(cfg.resp_peak_min_dist_sec * cfg.fs)))
    for seg in segments.itertuples(index=False):
        if not bool(seg.analyzable):
            continue
        start = int(seg.start_idx)
        end = int(seg.end_idx)
        x = np.asarray(resp[start:end], dtype=float)
        if len(x) < max(3, min_distance):
            continue
        x = signal.detrend(x, type="linear")
        amp_range = float(np.nanpercentile(x, 95) - np.nanpercentile(x, 5))
        if amp_range <= 1e-12:
            continue
        prominence = cfg.resp_peak_prom_ratio * amp_range
        valleys, props = signal.find_peaks(-x, distance=min_distance, prominence=prominence)
        if len(valleys) < 2:
            continue

        abs_valleys = valleys + start
        prominences = props.get("prominences", np.full(len(valleys), np.nan))
        for i in range(len(abs_valleys) - 1):
            frame_start = int(abs_valleys[i])
            frame_end = int(abs_valleys[i + 1])
            if frame_end <= frame_start:
                continue
            y = resp[frame_start : frame_end + 1]
            duration = (frame_end - frame_start) / cfg.fs
            amplitude = float(np.nanmax(y) - np.nanmin(y)) if len(y) else np.nan
            center_sec = 0.5 * (frame_start + frame_end) / cfg.fs
            quality, in_pass_window = window_quality_at_time(windows, center_sec)
            rows.append(
                {
                    "segment_id": int(seg.segment_id),
                    "frame_index": int(len(rows)),
                    "start_sec": float(frame_start / cfg.fs),
                    "end_sec": float(frame_end / cfg.fs),
                    "center_sec": float(center_sec),
                    "duration_sec": float(duration),
                    "rr_bpm": float(60.0 / duration) if duration > 0 else np.nan,
                    "amplitude": amplitude,
                    "left_valley_prominence": float(prominences[i]) if i < len(prominences) else np.nan,
                    "window_quality_score": quality,
                    "in_pass_window": in_pass_window,
                }
            )

    return pd.DataFrame(rows)


def estimate_baseline(frames: pd.DataFrame, windows: pd.DataFrame, cfg: GateConfig) -> dict[str, object]:
    normal_min = 60.0 / cfg.rr_max_bpm
    normal_max = cfg.min_event_sec
    if frames.empty:
        return {
            "baseline_rr_bpm": None,
            "baseline_amplitude": None,
            "baseline_frame_count": 0,
            "baseline_source": "none",
        }

    base = frames[
        (frames["duration_sec"] >= normal_min)
        & (frames["duration_sec"] < normal_max)
        & np.isfinite(frames["amplitude"])
        & (frames["amplitude"] > 0)
        & (frames["in_pass_window"])
    ].copy()
    source = "pass_gate_frames"
    if len(base) < cfg.baseline_min_frames:
        base = frames[
            (frames["duration_sec"] >= normal_min)
            & (frames["duration_sec"] < normal_max)
            & np.isfinite(frames["amplitude"])
            & (frames["amplitude"] > 0)
        ].copy()
        source = "all_normal_duration_frames"

    if base.empty:
        return {
            "baseline_rr_bpm": None,
            "baseline_amplitude": None,
            "baseline_frame_count": 0,
            "baseline_source": "none",
        }

    if "window_quality_score" in base.columns:
        base = base.sort_values("window_quality_score", ascending=False)
        top_n = max(cfg.baseline_min_frames, int(np.ceil(0.3 * len(base))))
        base = base.head(top_n)

    return {
        "baseline_rr_bpm": float(base["rr_bpm"].median()),
        "baseline_amplitude": float(base["amplitude"].median()),
        "baseline_frame_count": int(len(base)),
        "baseline_source": source,
    }


def merge_frame_events(frames: pd.DataFrame, mask: np.ndarray, event_type: str, cfg: GateConfig) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    current: list[pd.Series] = []
    for i, (_, row) in enumerate(frames.iterrows()):
        is_event = bool(mask[i])
        if is_event:
            current.append(row)
        elif current:
            events.extend(summarize_frame_group(current, event_type, cfg))
            current = []
    if current:
        events.extend(summarize_frame_group(current, event_type, cfg))
    return events


def summarize_frame_group(group: list[pd.Series], event_type: str, cfg: GateConfig) -> list[dict[str, object]]:
    start_sec = float(group[0]["start_sec"])
    end_sec = float(group[-1]["end_sec"])
    duration = end_sec - start_sec
    if duration < cfg.min_event_sec:
        return []
    amp_ratio = float(np.nanmedian([g.get("amplitude_ratio", np.nan) for g in group]))
    return [
        {
            "segment_id": int(group[0]["segment_id"]),
            "event_type": event_type,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": float(duration),
            "frame_count": int(len(group)),
            "median_amplitude_ratio": amp_ratio,
            "min_rr_bpm": float(np.nanmin([g["rr_bpm"] for g in group])),
            "rule": "consecutive_low_amplitude_frames",
        }
    ]


def detect_event_candidates(frames: pd.DataFrame, segments: pd.DataFrame, baseline: dict[str, object], cfg: GateConfig) -> pd.DataFrame:
    event_columns = [
        "segment_id",
        "event_type",
        "start_sec",
        "end_sec",
        "duration_sec",
        "frame_count",
        "median_amplitude_ratio",
        "min_rr_bpm",
        "rule",
    ]
    rows: list[dict[str, object]] = []
    baseline_amp = baseline.get("baseline_amplitude")
    if frames.empty or baseline_amp is None or not np.isfinite(float(baseline_amp)) or float(baseline_amp) <= 0:
        return pd.DataFrame(rows, columns=event_columns)

    out_frames = frames.copy()
    out_frames["amplitude_ratio"] = out_frames["amplitude"] / float(baseline_amp)
    apnea_ratio = 1.0 - cfg.apnea_drop_fraction
    hypopnea_ratio = 1.0 - cfg.hypopnea_drop_fraction

    for _, row in out_frames.iterrows():
        if row["duration_sec"] >= cfg.min_event_sec:
            ratio = float(row["amplitude_ratio"])
            event_type = "apnea_gap_candidate" if ratio <= hypopnea_ratio else "long_interval_candidate"
            rows.append(
                {
                    "segment_id": int(row["segment_id"]),
                    "event_type": event_type,
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                    "duration_sec": float(row["duration_sec"]),
                    "frame_count": 1,
                    "median_amplitude_ratio": ratio,
                    "min_rr_bpm": float(row["rr_bpm"]),
                    "rule": "valley_interval_ge_min_event_sec",
                }
            )

    for segment_id, group in out_frames.groupby("segment_id"):
        group = group.sort_values("start_sec").copy()
        apnea_mask = (group["amplitude_ratio"].to_numpy() <= apnea_ratio) & (group["duration_sec"].to_numpy() < cfg.min_event_sec)
        hypopnea_mask = (
            (group["amplitude_ratio"].to_numpy() <= hypopnea_ratio)
            & (group["amplitude_ratio"].to_numpy() > apnea_ratio)
            & (group["duration_sec"].to_numpy() < cfg.min_event_sec)
        )
        rows.extend(merge_frame_events(group, apnea_mask, "apnea_low_amplitude_candidate", cfg))
        rows.extend(merge_frame_events(group, hypopnea_mask, "hypopnea_candidate", cfg))

    return (
        pd.DataFrame(rows, columns=event_columns).sort_values(["start_sec", "end_sec"]).reset_index(drop=True)
        if rows
        else pd.DataFrame(rows, columns=event_columns)
    )


def normalized_for_plot(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    lo, hi = np.nanpercentile(x, [5, 95])
    scale = hi - lo
    if scale <= 1e-12:
        scale = np.nanstd(x) + 1e-12
    return (x - med) / scale


def set_robust_ylim(ax: plt.Axes, y: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> None:
    finite = np.asarray(y, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    lo, hi = np.nanpercentile(finite, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)


def plot_overview(
    out_path: Path,
    data: dict[str, np.ndarray],
    resp: np.ndarray,
    gate: dict[str, np.ndarray | float],
    windows: pd.DataFrame,
    cfg: GateConfig,
) -> None:
    t = data["t"]
    max_points = 40000
    stride = max(1, int(np.ceil(len(t) / max_points)))
    sl = slice(None, None, stride)

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(t[sl], normalized_for_plot(resp)[sl], linewidth=0.8, label="PVDF respiration")
    axes[0].plot(t[sl], normalized_for_plot(data["pr"])[sl], linewidth=0.6, alpha=0.75, label="PR")
    set_robust_ylim(axes[0], np.r_[normalized_for_plot(resp)[sl], normalized_for_plot(data["pr"])[sl]])
    axes[0].set_ylabel("norm.")
    axes[0].legend(loc="upper right")

    axes[1].plot(t[sl], gate["pvdf_score"][sl], linewidth=0.6, label="PVDF score")
    axes[1].plot(t[sl], gate["pr_score"][sl], linewidth=0.6, label="PR score")
    set_robust_ylim(axes[1], np.r_[gate["pvdf_score"][sl], gate["pr_score"][sl]], 0.0, 99.0)
    axes[1].set_ylabel("z")
    axes[1].legend(loc="upper right")

    axes[2].plot(t[sl], gate["motion_score"][sl], color="tab:red", linewidth=0.8, label="fused motion score")
    axes[2].axhline(gate["motion_threshold"], color="black", linestyle="--", linewidth=1.0, label="threshold")
    set_robust_ylim(axes[2], gate["motion_score"][sl], 0.0, 99.5)
    y_min, y_max = axes[2].get_ylim()
    motion_t = t[gate["motion_mask"]]
    if len(motion_t) > 0:
        axes[2].scatter(
            motion_t[:: max(1, len(motion_t) // 2500)],
            np.full_like(motion_t[:: max(1, len(motion_t) // 2500)], y_min + 0.05 * (y_max - y_min)),
            s=4,
            color="tab:green",
            label="motion mask",
        )
    axes[2].set_ylabel("score")
    axes[2].legend(loc="upper right")

    colors = windows["pass_gate"].map({True: "tab:blue", False: "tab:red"}).to_numpy()
    mids = 0.5 * (windows["start_sec"].to_numpy() + windows["end_sec"].to_numpy())
    axes[3].scatter(mids, windows["quality_score"], c=colors, s=14)
    axes[3].axhline(cfg.min_pass_quality, color="black", linestyle="--", linewidth=1.0)
    axes[3].set_ylabel("SQI")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylim(-2, 102)

    fig.suptitle("Dual-channel quality gating overview", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_detail(
    out_path: Path,
    data: dict[str, np.ndarray],
    resp: np.ndarray,
    gate: dict[str, np.ndarray | float],
    windows: pd.DataFrame,
    cfg: GateConfig,
    span_sec: float = 120.0,
) -> None:
    t = data["t"]
    if len(t) == 0:
        return

    center = int(np.argmax(gate["motion_score"]))
    half = int(round(0.5 * span_sec * cfg.fs))
    start = max(0, center - half)
    end = min(len(t), center + half)
    sl = slice(start, end)

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t[sl], normalized_for_plot(resp[sl]), linewidth=1.0, label="PVDF respiration")
    axes[0].plot(t[sl], normalized_for_plot(data["pr"][sl]), linewidth=0.8, alpha=0.8, label="PR")
    axes[0].set_ylabel("norm.")
    axes[0].legend(loc="upper right")

    axes[1].plot(t[sl], gate["pvdf_score"][sl], linewidth=0.8, label="PVDF score")
    axes[1].plot(t[sl], gate["pr_score"][sl], linewidth=0.8, label="PR score")
    axes[1].plot(t[sl], gate["motion_score"][sl], color="tab:red", linewidth=1.0, label="fused")
    axes[1].axhline(gate["motion_threshold"], color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("score")
    axes[1].legend(loc="upper right")

    local = windows[(windows["end_sec"] >= t[start]) & (windows["start_sec"] <= t[end - 1])]
    if len(local):
        mids = 0.5 * (local["start_sec"].to_numpy() + local["end_sec"].to_numpy())
        colors = local["pass_gate"].map({True: "tab:blue", False: "tab:red"}).to_numpy()
        axes[2].scatter(mids, local["quality_score"], c=colors, s=28)
    axes[2].axhline(cfg.min_pass_quality, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("SQI")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(-2, 102)

    fig.suptitle("Detail around strongest detected motion", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_event_overview(
    out_path: Path,
    data: dict[str, np.ndarray],
    resp: np.ndarray,
    gate: dict[str, np.ndarray | float],
    segments: pd.DataFrame,
    events: pd.DataFrame,
    cfg: GateConfig,
) -> None:
    t = data["t"]
    max_points = 40000
    stride = max(1, int(np.ceil(len(t) / max_points)))
    sl = slice(None, None, stride)

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(t[sl], normalized_for_plot(resp)[sl], linewidth=0.8, label="PVDF respiration")
    axes[0].set_ylabel("resp.")
    axes[0].legend(loc="upper right")

    axes[1].plot(t[sl], gate["motion_score"][sl], color="tab:red", linewidth=0.8, label="motion score")
    axes[1].axhline(gate["motion_threshold"], color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("motion")
    axes[1].legend(loc="upper right")

    if not segments.empty:
        for seg in segments.itertuples(index=False):
            if bool(seg.analyzable):
                axes[2].axvspan(seg.start_sec, seg.end_sec, color="tab:blue", alpha=0.08, linewidth=0)

    event_colors = {
        "apnea_gap_candidate": "tab:red",
        "apnea_low_amplitude_candidate": "tab:purple",
        "hypopnea_candidate": "tab:orange",
        "long_interval_candidate": "tab:brown",
    }
    if not events.empty:
        for event in events.itertuples(index=False):
            color = event_colors.get(event.event_type, "tab:gray")
            axes[0].axvspan(event.start_sec, event.end_sec, color=color, alpha=0.18, linewidth=0)
            axes[2].axvspan(event.start_sec, event.end_sec, color=color, alpha=0.50, linewidth=0)
            axes[2].text(
                0.5 * (event.start_sec + event.end_sec),
                0.5,
                str(event.event_type).replace("_candidate", ""),
                ha="center",
                va="center",
                fontsize=8,
                rotation=0,
            )

    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    axes[2].set_ylabel("events")
    axes[2].set_xlabel("Time (s)")
    fig.suptitle("Quiet segments and respiratory event candidates", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_summary(
    csv_path: Path,
    data: dict[str, np.ndarray],
    gate: dict[str, np.ndarray | float],
    windows: pd.DataFrame,
    segments: pd.DataFrame,
    frames: pd.DataFrame,
    events: pd.DataFrame,
    baseline: dict[str, object],
    cfg: GateConfig,
) -> dict[str, object]:
    passed = windows[windows["pass_gate"]]
    analyzable_segments = segments[segments["analyzable"]] if not segments.empty else segments
    return {
        "input_csv": str(csv_path),
        "fs_raw": cfg.fs_raw,
        "fs_processed": cfg.fs,
        "raw_samples": int(data["raw_samples"][0]),
        "processed_samples": int(len(data["t"])),
        "duration_sec": float(data["t"][-1] - data["t"][0]) if len(data["t"]) else 0.0,
        "window_sec": cfg.window_sec,
        "step_sec": cfg.step_sec,
        "motion_threshold": float(gate["motion_threshold"]),
        "motion_seconds": float(np.sum(gate["motion_mask"]) / cfg.fs),
        "motion_ratio": float(np.mean(gate["motion_mask"])),
        "windows_total": int(len(windows)),
        "windows_passed": int(windows["pass_gate"].sum()),
        "window_pass_ratio": float(windows["pass_gate"].mean()) if len(windows) else 0.0,
        "median_rr_bpm_passed": float(passed["pvdf_rr_bpm"].median()) if len(passed) else None,
        "median_quality_passed": float(passed["quality_score"].median()) if len(passed) else None,
        "label_counts": windows["label"].value_counts().to_dict(),
        "quiet_segments_total": int(len(segments)),
        "quiet_segments_analyzable": int(len(analyzable_segments)),
        "breath_frames_total": int(len(frames)),
        "event_candidates_total": int(len(events)),
        "event_type_counts": events["event_type"].value_counts().to_dict() if len(events) else {},
        "baseline": baseline,
        "config": asdict(cfg),
    }


def write_text_summary(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "Dual-channel PVDF + piezoresistive quality gating summary",
        f"Input: {summary['input_csv']}",
        f"Duration: {summary['duration_sec']:.1f} s",
        f"Processed sampling rate: {summary['fs_processed']:.1f} Hz",
        f"Motion ratio: {summary['motion_ratio']:.3f}",
        f"Window pass ratio: {summary['window_pass_ratio']:.3f}",
        f"Passed windows: {summary['windows_passed']} / {summary['windows_total']}",
        f"Median RR in passed windows: {summary['median_rr_bpm_passed']}",
        f"Median SQI in passed windows: {summary['median_quality_passed']}",
        f"Analyzable quiet segments: {summary['quiet_segments_analyzable']} / {summary['quiet_segments_total']}",
        f"Breath frames: {summary['breath_frames_total']}",
        f"Event candidates: {summary['event_candidates_total']}",
        f"Baseline: {summary['baseline']}",
        f"Label counts: {summary['label_counts']}",
        f"Event type counts: {summary['event_type_counts']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_csv(csv_path: Path, out_dir: Path, cfg: GateConfig) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_dual_channel_csv(csv_path, cfg)
    resp = extract_pvdf_respiration(data["pvdf"], cfg)
    gate = compute_motion_gate(data["pvdf"], data["pr"], cfg)
    segments = build_quiet_segments(gate, cfg)
    windows = attach_window_segments(build_window_table(data, resp, gate, cfg), segments)
    frames = build_breath_frames(resp, segments, windows, cfg)
    baseline = estimate_baseline(frames, windows, cfg)
    baseline_amp = baseline.get("baseline_amplitude")
    if len(frames) and baseline_amp is not None and np.isfinite(float(baseline_amp)) and float(baseline_amp) > 0:
        frames = frames.copy()
        frames["amplitude_ratio_to_baseline"] = frames["amplitude"] / float(baseline_amp)
    events = detect_event_candidates(frames, segments, baseline, cfg)

    windows_path = out_dir / "quality_windows.csv"
    segments_path = out_dir / "quiet_segments.csv"
    frames_path = out_dir / "breath_frames.csv"
    events_path = out_dir / "event_candidates.csv"
    baseline_path = out_dir / "baseline.json"
    summary_json_path = out_dir / "summary.json"
    summary_txt_path = out_dir / "summary.txt"
    figure_path = out_dir / "quality_gating_overview.png"
    detail_figure_path = out_dir / "quality_gating_detail.png"
    event_figure_path = out_dir / "event_candidates_overview.png"

    windows.to_csv(windows_path, index=False, encoding="utf-8-sig")
    segments.to_csv(segments_path, index=False, encoding="utf-8-sig")
    frames.to_csv(frames_path, index=False, encoding="utf-8-sig")
    events.to_csv(events_path, index=False, encoding="utf-8-sig")
    baseline_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = make_summary(csv_path, data, gate, windows, segments, frames, events, baseline, cfg)
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_text_summary(summary_txt_path, summary)
    plot_overview(figure_path, data, resp, gate, windows, cfg)
    plot_detail(detail_figure_path, data, resp, gate, windows, cfg)
    plot_event_overview(event_figure_path, data, resp, gate, segments, events, cfg)

    summary["outputs"] = {
        "windows_csv": str(windows_path),
        "quiet_segments_csv": str(segments_path),
        "breath_frames_csv": str(frames_path),
        "event_candidates_csv": str(events_path),
        "baseline_json": str(baseline_path),
        "summary_json": str(summary_json_path),
        "summary_txt": str(summary_txt_path),
        "overview_png": str(figure_path),
        "detail_png": str(detail_figure_path),
        "event_overview_png": str(event_figure_path),
    }
    return summary


def resolve_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.csv"))
    return [path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PVDF + piezoresistive dual-channel quality gating.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV, help="CSV file or a directory of CSV files.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Directory for generated tables/figures.")
    parser.add_argument("--window-sec", type=float, default=30.0, help="SQI window length in seconds; try 20, 30, or 60.")
    parser.add_argument("--step-sec", type=float, default=5.0, help="Sliding-window step in seconds.")
    parser.add_argument("--motion-threshold-z", type=float, default=3.0, help="Robust z threshold for motion detection.")
    parser.add_argument("--pvdf-weight", type=float, default=0.5, help="PVDF weight in fused motion score; PR weight is 1-this.")
    parser.add_argument("--voltage-rate-weight", type=float, default=0.40, help="Weight of PVDF+PR smoothed voltage derivative in motion score.")
    parser.add_argument("--envelope-rate-weight", type=float, default=0.40, help="Weight of PVDF+PR envelope derivative in motion score.")
    parser.add_argument("--wavelet-motion-weight", type=float, default=0.20, help="Weight of PVDF-only wavelet detail energy; set 0 for smoothing-only motion gating.")
    parser.add_argument("--min-quiet-segment-sec", type=float, default=10.0, help="Quiet segments shorter than this are not analyzed for events.")
    parser.add_argument("--min-event-sec", type=float, default=10.0, help="Minimum duration for apnea/hypopnea event candidates.")
    parser.add_argument("--apnea-drop-fraction", type=float, default=0.90, help="Amplitude drop fraction for apnea-like candidates.")
    parser.add_argument("--hypopnea-drop-fraction", type=float, default=0.30, help="Amplitude drop fraction for hypopnea-like candidates.")
    parser.add_argument("--fs-raw", type=float, default=500.0, help="Raw sampling rate in Hz.")
    parser.add_argument("--downsample-q", type=int, default=10, help="Downsampling factor; 500/10 gives 50 Hz.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GateConfig(
        fs_raw=args.fs_raw,
        downsample_q=args.downsample_q,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        motion_threshold_z=args.motion_threshold_z,
        pvdf_weight=args.pvdf_weight,
        voltage_rate_weight=args.voltage_rate_weight,
        envelope_rate_weight=args.envelope_rate_weight,
        wavelet_motion_weight=args.wavelet_motion_weight,
        min_quiet_segment_sec=args.min_quiet_segment_sec,
        min_event_sec=args.min_event_sec,
        apnea_drop_fraction=args.apnea_drop_fraction,
        hypopnea_drop_fraction=args.hypopnea_drop_fraction,
    )

    csv_paths = resolve_inputs(args.input)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found: {args.input}")

    summaries = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        child_out = args.out_dir / csv_path.stem if len(csv_paths) > 1 else args.out_dir
        print(f"Analyzing {csv_path}")
        summaries.append(analyze_csv(csv_path, child_out, cfg))

    if len(summaries) > 1:
        aggregate_path = args.out_dir / "batch_summary.json"
        aggregate_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        brief_rows = []
        for item in summaries:
            brief_rows.append(
                {
                    "input_csv": Path(item["input_csv"]).name,
                    "duration_sec": item["duration_sec"],
                    "motion_ratio": item["motion_ratio"],
                    "window_pass_ratio": item["window_pass_ratio"],
                    "windows_passed": item["windows_passed"],
                    "windows_total": item["windows_total"],
                    "median_rr_bpm_passed": item["median_rr_bpm_passed"],
                    "median_quality_passed": item["median_quality_passed"],
                    "quiet_segments_analyzable": item["quiet_segments_analyzable"],
                    "breath_frames_total": item["breath_frames_total"],
                    "event_candidates_total": item["event_candidates_total"],
                }
            )
        brief_path = args.out_dir / "batch_summary.csv"
        pd.DataFrame(brief_rows).to_csv(brief_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {aggregate_path}")
        print(f"Wrote {brief_path}")
    else:
        print(json.dumps(summaries[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
