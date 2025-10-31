# pv/postprocess/dynamic_correction.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, List
import numpy as np
import pandas as pd


# ---------- 数据结构 ----------
@dataclass
class WarningEvent:
    """云层过境预警事件"""
    start: pd.Timestamp           # 预计开始时刻 s_j
    end: pd.Timestamp             # 预计结束时刻 e_j
    strength: float               # 影响强度 a_j (0~1，正值表示遮蔽增强)
    confidence: float             # 置信度 c_j (0~1)


@dataclass
class CorrectionParams:
    """修正模块超参数（来自文档公式）"""
    # 时间轴
    dt_seconds: Optional[float] = None           # 采样周期秒；None 时自动从 index 推断
    # 映射：影响信号 -> 时间移位 Δt(k)
    k_shift: float = 0.0                         # 映射系数 β
    delta_min: float = -3600.0                   # Δt 下限（秒）
    delta_max: float = 3600.0                    # Δt 上限（秒）
    # 映射：影响信号 -> 幅值缩放 α(k)
    k_scale: float = 0.0                         # 灵敏系数 η
    a_min: float = 0.0                           # 缩放下限 A_min
    a_max: float = 1.2                           # 缩放上限 A_max
    # 饱和边界
    p_max: Optional[float] = None                # 功率上限 P_max（装机/逆变限制）
    # 斜率约束（单位：功率/采样周期）
    s_up: float = np.inf                         # 上升斜率上限
    s_down: float = np.inf                       # 下降斜率上限
    # 置信度 → 融合权重 w(k)
    w0: float = 0.5                              # 基准权重
    k_w: float = 0.5                             # 线性增益
    w_min: float = 0.0
    w_max: float = 1.0


# ---------- 核心实现 ----------
def build_influence_and_conf(index: pd.DatetimeIndex,
                             events: Iterable[WarningEvent]) -> tuple[np.ndarray, np.ndarray]:
    """
    依据预警事件流构造：
      r(k) = Σ_j a_j * c_j * 1_{[s_j, e_j]}(t_k)
      c(k) = Σ_j c_j          * 1_{[s_j, e_j]}(t_k)   （置信度可用于融合权重）
    事件可交叠；同一时刻累加并不超过 1（做截断以避免过度修正）
    """
    r = np.zeros(len(index), dtype=float)
    c = np.zeros(len(index), dtype=float)
    if not events:
        return r, c

    # 将时间轴转秒，便于切片
    ts = index.view('int64') // 10**9
    for ev in events:
        s = int(pd.Timestamp(ev.start).value // 10**9)
        e = int(pd.Timestamp(ev.end).value // 10**9)
        if e < ts[0] or s > ts[-1]:
            continue
        m = (ts >= s) & (ts <= e)
        r[m] += ev.strength * ev.confidence
        c[m] += ev.confidence

    # 截断到 [0,1]
    r = np.clip(r, 0.0, 1.0)
    c = np.clip(c, 0.0, 1.0)
    return r, c


def _interp_with_shift(base: np.ndarray,
                       index: pd.DatetimeIndex,
                       delta_t_sec: np.ndarray) -> np.ndarray:
    """
    时间移位重采样：
      给出每个采样点的移位量 Δt(k)（秒），构造移动后的时间 grid: t_k + Δt(k)，
      再把 base(t) 线性插值回原始 t_k 上。
    """
    t = index.view('int64') / 1e9                      # 秒
    t_shifted = t + delta_t_sec.astype(float)

    # 单调性保障：避免 t_shifted 非单调导致的插值异常
    # 对局部小逆序做最小增量修复
    eps = np.finfo(float).eps
    for i in range(1, len(t_shifted)):
        if t_shifted[i] <= t_shifted[i-1]:
            t_shifted[i] = t_shifted[i-1] + max(eps, (t[1]-t[0])*1e-6)

    # 构造 base 的原始插值函数，并在 t_shifted 上取值
    y_base = base.astype(float)
    # 原始定义域（略加外延，避免边界 NaN）
    t0, t1 = t[0], t[-1]
    # 用 numpy.interp：超界使用边界值
    y_on_shifted = np.interp(t_shifted, t, y_base, left=y_base[0], right=y_base[-1])

    # 再把 y_on_shifted 视作“在 t_k+Δt(k) 的值”，插回到 t_k 上
    # 由于我们希望得到 ŷ(k) 与原始索引对齐，这里直接返回 y_on_shifted 即可
    return y_on_shifted


def _apply_slope_limits(y: np.ndarray, s_up: float, s_down: float) -> np.ndarray:
    """双向扫描，上升/下降斜率约束（每步不超过 s_up / s_down）"""
    if not np.isfinite(s_up) and not np.isfinite(s_down):
        return np.clip(y, 0, None)

    out = y.copy()
    # 早->晚：限制上升
    if np.isfinite(s_up):
        for i in range(1, len(out)):
            dy = out[i] - out[i-1]
            if dy > s_up:
                out[i] = out[i-1] + s_up
    # 晚->早：限制下降
    if np.isfinite(s_down):
        for i in range(len(out)-2, -1, -1):
            dy = out[i] - out[i+1]
            if dy > s_down:
                out[i] = out[i+1] + s_down
    return np.clip(out, 0, None)


def dynamic_correction(base: pd.Series,
                       events: List[WarningEvent],
                       params: CorrectionParams) -> pd.DataFrame:
    """
    输入：
      base   : 基础预测（Series，DatetimeIndex 等间隔）
      events : 预警事件列表
      params : 超参数

    输出：包含各中间产物与最终结果的 DataFrame：
      ['base','delta_t','alpha','y_shifted','y_scaled','y_sat','y_slope','w','y_final']
    """
    assert isinstance(base.index, pd.DatetimeIndex), "base.index 必须为 DatetimeIndex"
    idx = base.index
    y0 = np.asarray(base.values, dtype=float)

    # 推断采样周期
    if params.dt_seconds is None:
        if len(idx) < 2:
            raise ValueError("无法推断采样周期，请显式设置 params.dt_seconds")
        dt = (idx[1] - idx[0]).total_seconds()
    else:
        dt = float(params.dt_seconds)

    # 1) 影响信号 r(k) 与置信度 c(k)
    r, c = build_influence_and_conf(idx, events)

    # 2) 时间移位 Δt(k) = clip(β * r(k), [δ_min, δ_max])
    delta_t = np.clip(params.k_shift * r, params.delta_min, params.delta_max)

    # 3) 时间轴重采样（应用 Δt）
    y_shifted = _interp_with_shift(y0, idx, delta_t)

    # 4) 幅值缩放 α(k) = clip(1 - η * r(k), [A_min, A_max])
    alpha = np.clip(1.0 - params.k_scale * r, params.a_min, params.a_max)
    y_scaled = np.clip(y_shifted * alpha, 0, None)

    # 5) 饱和裁剪
    if params.p_max is not None:
        y_sat = np.minimum(y_scaled, params.p_max)
    else:
        y_sat = y_scaled

    # 6) 斜率约束（单位是“每采样步的功率变化上限”）
    # 文档里 S_up / S_down 为上下行斜率上限；若你给的是“kW/分钟”，可以乘以 dt/60 做换算
    y_slope = _apply_slope_limits(y_sat, params.s_up, params.s_down)

    # 7) 置信度融合权重：w(k) = clip(w0 + k_w * c(k), [w_min, w_max])
    w = np.clip(params.w0 + params.k_w * c, params.w_min, params.w_max)

    # 8) 目标预测 y(k) = w(k)*y_slope(k) + (1-w(k))*base(k)
    y_final = w * y_slope + (1.0 - w) * y0

    out = pd.DataFrame({
        "base": y0,
        "delta_t": delta_t,
        "alpha": alpha,
        "y_shifted": y_shifted,
        "y_scaled": y_scaled,
        "y_sat": y_sat,
        "y_slope": y_slope,
        "w": w,
        "y_final": y_final,
    }, index=idx)
    return out
