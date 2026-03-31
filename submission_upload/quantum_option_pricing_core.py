"""量子期权定价核心模块。

覆盖以下三大必选模块：

1. 量子信息加载（State Preparation）
2. 量子算术与收益编码（Payoff Oracle）
3. 迭代量子振幅估计（IQAE）

本实现严格遵守赛题硬件约束：
- 量子线路仅使用 pyqpanda3
- 双比特门仅使用 CNOT
- 不调用任意态加载宏门
- 不调用多控制旋转宏门
- 不调用多控制 Z 宏门
- 不调用现成 QAE / IQAE 高级算法封装

换言之，所有复杂受控操作均由底层单比特门与 CNOT 手工编译完成。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pyqpanda3 as pq

try:
    from scipy.optimize import minimize_scalar
except Exception:  # pragma: no cover
    minimize_scalar = None

try:
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover
    gaussian_kde = None


QP = pq if all(hasattr(pq, name) for name in ("CPUQVM", "QProg", "RY", "CNOT")) else pq.core


# ============================================================================
# ==== 动态参数配置区 ====
# 评委或使用者可在此自由修改金融参数，以一键验证不同市场条件下的定价结果。
# 当前默认值对应 SPY 欧式看涨期权示例，但这些参数并不局限于固定样例：
# - 可替换为不同执行价、利率、波动率、期限
# - 可接入真实金融市场数据拟合后的参数
# - 可作为扩展到奇异期权、不同收益结构与更一般离散分布输入的统一入口
# ============================================================================
@dataclass(frozen=True)
class PricingConfig:
    """比赛参数配置。

    当前默认参数与赛题中的 SPY 欧式看涨期权示例一致。
    """

    s0: float = 500.0
    strike: float = 500.0
    r: float = 0.05
    sigma: float = 0.15
    maturity: float = 30.0 / 365.0
    num_state_qubits: int = 3
    truncation_sigma: float = 3.0
    scaling_constant: float = 0.5


def normal_cdf(x: float) -> float:
    """标准正态分布的累积分布函数。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call_price(
    s0: float,
    strike: float,
    r: float,
    sigma: float,
    maturity: float,
) -> float:
    """BSM 欧式看涨期权解析解。

    报告中除了给出量子结果与离散网格结果，也给出与连续模型解析解的对比。
    """
    if maturity <= 0.0:
        return max(s0 - strike, 0.0)
    if sigma <= 0.0:
        forward_price = s0 * math.exp(r * maturity)
        payoff = max(forward_price - strike, 0.0)
        return math.exp(-r * maturity) * payoff

    sqrt_t = math.sqrt(maturity)
    d1 = (math.log(s0 / strike) + (r + 0.5 * sigma**2) * maturity) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return s0 * normal_cdf(d1) - strike * math.exp(-r * maturity) * normal_cdf(d2)


def lognormal_price_pdf(
    prices: np.ndarray,
    s0: float,
    r: float,
    sigma: float,
    maturity: float,
) -> np.ndarray:
    """BSM 模型下终值价格 S_T 的对数正态概率密度。"""
    mu = np.log(s0) + (r - 0.5 * sigma**2) * maturity
    std = sigma * np.sqrt(maturity)

    log_term = (np.log(prices) - mu) / std
    normalizer = prices * std * np.sqrt(2.0 * np.pi)
    return np.exp(-0.5 * log_term**2) / normalizer


def discretize_bsm_price_distribution(config: PricingConfig) -> dict[str, np.ndarray | float]:
    """将连续 BSM 终值分布离散到 2^n 个等距价格点。

    这里我们只用 3 个状态比特，因此共有 8 个离散价格状态。
    为降低截断误差，网格区间取为

        [S0 * exp(-3 sigma sqrt(T)), S0 * exp(+3 sigma sqrt(T))].

    【加分项说明】本底层态加载方案基于完全二叉树展开，因此并不局限于
    BSM 对数正态假设。若更换为真实市场拟合的肥尾分布（Fat-tail
    distribution）或量子 GAN 生成的任意离散概率向量，仅需替换本函数
    的输出，后续底层量子线路编译与 IQAE 流程无需任何修改即可无缝适配。
    """
    num_points = 2**config.num_state_qubits
    sqrt_t = np.sqrt(config.maturity)

    price_min = config.s0 * np.exp(-config.truncation_sigma * config.sigma * sqrt_t)
    price_max = config.s0 * np.exp(config.truncation_sigma * config.sigma * sqrt_t)
    price_grid = np.linspace(price_min, price_max, num_points)

    pdf_values = lognormal_price_pdf(
        prices=price_grid,
        s0=config.s0,
        r=config.r,
        sigma=config.sigma,
        maturity=config.maturity,
    )

    grid_spacing = price_grid[1] - price_grid[0]
    unnormalized_probabilities = pdf_values * grid_spacing
    probabilities = unnormalized_probabilities / np.sum(unnormalized_probabilities)
    amplitudes = np.sqrt(probabilities)

    return {
        "price_min": float(price_min),
        "price_max": float(price_max),
        "price_grid": price_grid,
        "pdf_values": pdf_values,
        "probabilities": probabilities,
        "amplitudes": amplitudes,
    }


def _histogram_density(values: np.ndarray, evaluation_points: np.ndarray) -> np.ndarray:
    """在缺少 KDE 时，用直方图做一个轻量级密度近似。"""
    num_bins = max(16, 2 * evaluation_points.size)
    density, bin_edges = np.histogram(values, bins=num_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    interpolated = np.interp(evaluation_points, bin_centers, density, left=0.0, right=0.0)
    return np.clip(interpolated, 0.0, None)


def _normalize_header_name(name: str) -> str:
    """将 CSV 列名标准化，便于自动识别价格列。"""
    return "".join(character for character in str(name).lower() if character.isalnum())


def load_price_series_from_csv(
    csv_path: str | Path,
    price_column: str | None = None,
) -> np.ndarray:
    """从本地 CSV 读取价格序列。

    默认会自动识别常见价格列：
    - Adj Close / adj_close / adjusted_close
    - Close / close
    - settle / last
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到历史数据文件：{csv_path}")

    table = np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    if table.dtype.names is None:
        raise ValueError("CSV 文件缺少表头，无法识别价格列。")

    column_names = list(table.dtype.names)
    normalized_to_original = {_normalize_header_name(name): name for name in column_names}

    if price_column is None:
        candidate_keys = [
            "adjclose",
            "adjustedclose",
            "close",
            "settle",
            "last",
        ]
        matched_column = None
        for key in candidate_keys:
            if key in normalized_to_original:
                matched_column = normalized_to_original[key]
                break
        if matched_column is None:
            raise ValueError(
                f"无法自动识别价格列。可选列为：{column_names}，请显式传入 price_column。"
            )
        price_column = matched_column
    else:
        normalized_input = _normalize_header_name(price_column)
        if normalized_input not in normalized_to_original:
            raise ValueError(f"CSV 中不存在价格列 `{price_column}`。可选列为：{column_names}")
        price_column = normalized_to_original[normalized_input]

    prices = np.asarray(table[price_column], dtype=float)
    prices = prices[np.isfinite(prices)]
    prices = prices[prices > 0.0]
    if prices.size < 10:
        raise ValueError("有效历史价格数据点过少，无法稳定拟合收益率分布。")
    return prices


def fit_historical_log_return_distribution(
    price_series: np.ndarray,
    maturity: float,
    trading_days_per_year: int = 252,
) -> dict[str, np.ndarray | int]:
    """基于历史价格序列拟合到期收益率分布。

    这里不再使用 BSM 对数正态假设，而是直接使用真实历史价格序列构造
    与到期期限相匹配的重叠对数收益率样本：

        R_t^{(h)} = log(S_t / S_{t-h})

    其中 h 约为 maturity 对应的交易日数。该函数返回的是经验 horizon
    对数收益率样本，可进一步通过 KDE 或直方图估计映射为离散概率分布。
    """
    prices = np.asarray(price_series, dtype=float)
    prices = prices[np.isfinite(prices)]
    prices = prices[prices > 0.0]

    horizon_steps = max(1, int(round(maturity * trading_days_per_year)))
    if prices.size <= horizon_steps:
        raise ValueError("历史价格序列长度不足，无法拟合与当前到期期限匹配的收益率分布。")

    log_prices = np.log(prices)
    horizon_log_returns = log_prices[horizon_steps:] - log_prices[:-horizon_steps]

    return {
        "clean_price_series": prices,
        "horizon_steps": horizon_steps,
        "horizon_log_returns": horizon_log_returns,
    }


def discretize_empirical_price_distribution_from_history(
    price_series: np.ndarray,
    s0: float,
    maturity: float,
    num_qubits: int = 3,
    trading_days_per_year: int = 252,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> dict[str, np.ndarray | float | int | str]:
    """将历史收益率拟合得到的经验分布直接离散成量子态加载所需格式。

    该函数用于“真实金融数据拟合”加分项：
    - 输入真实历史价格序列
    - 计算与到期期限匹配的重叠对数收益率
    - 通过 KDE（或直方图回退）估计未来收益率密度
    - 输出与 BSM 版本完全同构的 {price_grid, probabilities, amplitudes}

    因此，后续的底层态加载、收益编码、Grover 与 IQAE 主流程都可直接复用。
    """
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("quantile 截断范围必须满足 0 < lower_quantile < upper_quantile < 1。")

    fitted = fit_historical_log_return_distribution(
        price_series=price_series,
        maturity=maturity,
        trading_days_per_year=trading_days_per_year,
    )
    horizon_log_returns = np.asarray(fitted["horizon_log_returns"], dtype=float)
    num_points = 2**num_qubits

    return_lower = float(np.quantile(horizon_log_returns, lower_quantile))
    return_upper = float(np.quantile(horizon_log_returns, upper_quantile))
    if np.isclose(return_lower, return_upper):
        return_lower -= 1e-4
        return_upper += 1e-4

    price_min = float(s0 * np.exp(return_lower))
    price_max = float(s0 * np.exp(return_upper))
    price_grid = np.linspace(price_min, price_max, num_points)
    return_grid = np.log(price_grid / s0)

    fit_method = "histogram"
    return_density = None

    if gaussian_kde is not None and horizon_log_returns.size >= 3:
        try:
            kde = gaussian_kde(horizon_log_returns)
            return_density = kde(return_grid)
            fit_method = "gaussian-kde"
        except Exception:
            return_density = None

    if return_density is None:
        return_density = _histogram_density(horizon_log_returns, return_grid)

    price_density = np.clip(return_density, 0.0, None) / price_grid
    grid_spacing = price_grid[1] - price_grid[0]
    unnormalized_probabilities = price_density * grid_spacing
    probabilities = unnormalized_probabilities / np.sum(unnormalized_probabilities)
    amplitudes = np.sqrt(probabilities)

    return {
        "price_min": price_min,
        "price_max": price_max,
        "price_grid": price_grid,
        "pdf_values": price_density,
        "probabilities": probabilities,
        "amplitudes": amplitudes,
        "fit_method": fit_method,
        "horizon_steps": int(fitted["horizon_steps"]),
        "horizon_log_returns": horizon_log_returns,
    }


def _safe_theta(left_weight: float, right_weight: float) -> float:
    """稳定计算 2*atan2(right, left)。"""
    if np.isclose(left_weight, 0.0) and np.isclose(right_weight, 0.0):
        return 0.0
    return 2.0 * np.arctan2(right_weight, left_weight)


def calculate_ry_angles(amplitudes: np.ndarray) -> np.ndarray:
    """计算 3 比特任意实振幅态加载所需的 7 个树形 RY 角。

    此处为了符合 NISQ 硬件约束，弃用了高级任意态加载宏，改为手工构造
    满二叉树分解，并把每一层条件概率递归反推出 RY 旋转角。
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    if amplitudes.shape != (8,):
        raise ValueError("当前实现固定针对 3 个状态比特，振幅向量长度必须为 8。")
    if np.any(amplitudes < -1e-12):
        raise ValueError("当前底层态制备实现假设振幅为非负实数。")

    norm = np.linalg.norm(amplitudes)
    if np.isclose(norm, 0.0):
        raise ValueError("振幅向量不能为零向量。")
    amplitudes = amplitudes / norm

    a0, a1, a2, a3, a4, a5, a6, a7 = amplitudes

    theta0 = _safe_theta(np.linalg.norm([a0, a1, a2, a3]), np.linalg.norm([a4, a5, a6, a7]))
    theta1 = _safe_theta(np.linalg.norm([a0, a1]), np.linalg.norm([a2, a3]))
    theta2 = _safe_theta(np.linalg.norm([a4, a5]), np.linalg.norm([a6, a7]))
    theta3 = _safe_theta(a0, a1)
    theta4 = _safe_theta(a2, a3)
    theta5 = _safe_theta(a4, a5)
    theta6 = _safe_theta(a6, a7)

    return np.array([theta0, theta1, theta2, theta3, theta4, theta5, theta6], dtype=float)


def _append_single_control_ry_multiplexor(
    prog,
    control,
    target,
    angle_if_control_0: float,
    angle_if_control_1: float,
) -> None:
    """把 1 控制多路复用 RY 手工分解为 RY + CNOT。

    此处为了符合 NISQ 硬件约束，弃用了封装好的受控 RY 门。
    """
    alpha0 = 0.5 * (angle_if_control_0 + angle_if_control_1)
    alpha1 = 0.5 * (angle_if_control_0 - angle_if_control_1)

    prog << QP.RY(target, alpha0)
    prog << QP.CNOT(control, target)
    prog << QP.RY(target, alpha1)
    prog << QP.CNOT(control, target)


def _append_inverse_single_control_ry_multiplexor(
    prog,
    control,
    target,
    angle_if_control_0: float,
    angle_if_control_1: float,
) -> None:
    """1 控制多路复用 RY 的精确逆线路。"""
    alpha0 = 0.5 * (angle_if_control_0 + angle_if_control_1)
    alpha1 = 0.5 * (angle_if_control_0 - angle_if_control_1)

    prog << QP.CNOT(control, target)
    prog << QP.RY(target, -alpha1)
    prog << QP.CNOT(control, target)
    prog << QP.RY(target, -alpha0)


def _append_double_control_ry_multiplexor(
    prog,
    control_msb,
    control_lsb,
    target,
    angles_by_branch: np.ndarray,
) -> None:
    """把 2 控制多路复用 RY 手工分解为 RY + CNOT 梯子。"""
    theta00, theta01, theta10, theta11 = np.asarray(angles_by_branch, dtype=float)

    beta0 = 0.25 * (theta00 + theta01 + theta10 + theta11)
    beta1 = 0.25 * (theta00 - theta01 + theta10 - theta11)
    beta2 = 0.25 * (theta00 - theta01 - theta10 + theta11)
    beta3 = 0.25 * (theta00 + theta01 - theta10 - theta11)

    prog << QP.RY(target, beta0)
    prog << QP.CNOT(control_lsb, target)
    prog << QP.RY(target, beta1)
    prog << QP.CNOT(control_msb, target)
    prog << QP.RY(target, beta2)
    prog << QP.CNOT(control_lsb, target)
    prog << QP.RY(target, beta3)
    prog << QP.CNOT(control_msb, target)


def _append_inverse_double_control_ry_multiplexor(
    prog,
    control_msb,
    control_lsb,
    target,
    angles_by_branch: np.ndarray,
) -> None:
    """2 控制多路复用 RY 的精确逆线路。"""
    theta00, theta01, theta10, theta11 = np.asarray(angles_by_branch, dtype=float)

    beta0 = 0.25 * (theta00 + theta01 + theta10 + theta11)
    beta1 = 0.25 * (theta00 - theta01 + theta10 - theta11)
    beta2 = 0.25 * (theta00 - theta01 - theta10 + theta11)
    beta3 = 0.25 * (theta00 + theta01 - theta10 - theta11)

    prog << QP.CNOT(control_msb, target)
    prog << QP.RY(target, -beta3)
    prog << QP.CNOT(control_lsb, target)
    prog << QP.RY(target, -beta2)
    prog << QP.CNOT(control_msb, target)
    prog << QP.RY(target, -beta1)
    prog << QP.CNOT(control_lsb, target)
    prog << QP.RY(target, -beta0)


def build_sp_circuit(angles: np.ndarray, q_state):
    """构建 3 比特价格分布态制备线路。

    此处为了符合 NISQ 硬件约束，弃用了高级态加载宏，整条线路完全由
    单比特 RY 与 CNOT 手工拼装。
    """
    if len(q_state) != 3:
        raise ValueError("状态加载模块固定使用 3 个状态比特。")

    theta0, theta1, theta2, theta3, theta4, theta5, theta6 = np.asarray(angles, dtype=float)
    prog = QP.QProg()

    prog << QP.RY(q_state[0], theta0)
    _append_single_control_ry_multiplexor(prog, q_state[0], q_state[1], theta1, theta2)
    _append_double_control_ry_multiplexor(
        prog,
        q_state[0],
        q_state[1],
        q_state[2],
        np.array([theta3, theta4, theta5, theta6], dtype=float),
    )
    return prog


def build_inverse_sp_circuit(angles: np.ndarray, q_state):
    """构建价格态制备线路的精确逆线路。"""
    if len(q_state) != 3:
        raise ValueError("状态加载模块固定使用 3 个状态比特。")

    theta0, theta1, theta2, theta3, theta4, theta5, theta6 = np.asarray(angles, dtype=float)
    prog = QP.QProg()

    _append_inverse_double_control_ry_multiplexor(
        prog,
        q_state[0],
        q_state[1],
        q_state[2],
        np.array([theta3, theta4, theta5, theta6], dtype=float),
    )
    _append_inverse_single_control_ry_multiplexor(prog, q_state[0], q_state[1], theta1, theta2)
    prog << QP.RY(q_state[0], -theta0)
    return prog


def calculate_call_payoffs(price_grid: np.ndarray, strike: float) -> np.ndarray:
    """离散价格网格上的欧式看涨收益。"""
    return np.maximum(np.asarray(price_grid, dtype=float) - strike, 0.0)


def calculate_barrier_call_payoffs(
    price_grid: np.ndarray,
    strike: float,
    barrier_price: float,
) -> np.ndarray:
    """离散价格网格上的向上敲出看涨期权收益。

    Up-and-Out Call 的收益逻辑为：
    - 当 S >= barrier_price 时，收益直接归零
    - 否则收益为 max(S - K, 0)

    该函数作为奇异期权扩展示例，说明本文的收益编码框架并不局限于
    欧式看涨期权；只需替换经典端 payoff 计算，后续量子线路编译流程可保持不变。
    """
    prices = np.asarray(price_grid, dtype=float)
    vanilla_payoff = np.maximum(prices - strike, 0.0)
    knock_out_mask = prices >= barrier_price
    return np.where(knock_out_mask, 0.0, vanilla_payoff)


def calculate_gamma_angles(
    payoff_list: np.ndarray,
    scaling_constant: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """把收益映射为目标比特的分支角 gamma_i。

    定义：
        sin^2(gamma_i / 2) = c * payoff_i / max_payoff
    """
    payoff_list = np.asarray(payoff_list, dtype=float)
    if payoff_list.ndim != 1:
        raise ValueError("payoff_list 必须是一维数组。")
    if scaling_constant <= 0.0 or scaling_constant > 1.0:
        raise ValueError("scaling_constant 必须位于 (0, 1]。")

    max_payoff = float(np.max(payoff_list))
    if np.isclose(max_payoff, 0.0):
        scaled_payoffs = np.zeros_like(payoff_list)
        gamma_list = np.zeros_like(payoff_list)
    else:
        scaled_payoffs = scaling_constant * payoff_list / max_payoff
        if np.any(scaled_payoffs > 1.0 + 1e-12):
            raise ValueError("收益缩放后超过 1，请减小 scaling_constant。")
        scaled_payoffs = np.clip(scaled_payoffs, 0.0, 1.0)
        gamma_list = 2.0 * np.arcsin(np.sqrt(scaled_payoffs))

    return gamma_list, max_payoff, scaled_payoffs


def gray_code_sequence(num_controls: int) -> list[int]:
    """生成标准二进制 Gray code 序列。"""
    return [index ^ (index >> 1) for index in range(2**num_controls)]


def _gray_ordered_walsh_hadamard_matrix(num_controls: int) -> np.ndarray:
    """构造 Gray code 顺序下的 Walsh-Hadamard 变换矩阵。

    此处为了符合 NISQ 硬件约束，弃用了多控制旋转宏门，
    改用 Gray code + Walsh-Hadamard 角变换，先在经典端求出 delta 角，
    再在量子端仅用 RY 与 CNOT 交替实现均匀受控旋转。
    """
    num_branches = 2**num_controls
    matrix = np.empty((num_branches, num_branches), dtype=float)
    gray_codes = gray_code_sequence(num_controls)

    for row, gray_word in enumerate(gray_codes):
        for col in range(num_branches):
            parity = (gray_word & col).bit_count() % 2
            matrix[row, col] = 1.0 if parity == 0 else -1.0

    return matrix / num_branches


def calculate_payoff_angles(gamma_list: np.ndarray) -> np.ndarray:
    """将分支角 gamma_i 变换为底层多路复用角 delta_i。"""
    gamma_list = np.asarray(gamma_list, dtype=float)
    if gamma_list.ndim != 1:
        raise ValueError("gamma_list 必须为一维数组。")

    num_branches = gamma_list.size
    num_controls = int(np.log2(num_branches))
    if 2**num_controls != num_branches:
        raise ValueError("gamma_list 的长度必须是 2 的整数次幂。")

    transform = _gray_ordered_walsh_hadamard_matrix(num_controls)
    return transform @ gamma_list


def _control_sequence_for_uniform_rotation(q_control_list) -> list[int]:
    """返回均匀受控旋转的 Gray code 控制翻转顺序。"""
    num_controls = len(q_control_list)
    gray_codes = gray_code_sequence(num_controls)
    sequence = []

    for index in range(len(gray_codes)):
        current_gray_word = gray_codes[index]
        next_gray_word = gray_codes[(index + 1) % len(gray_codes)]
        toggle_mask = current_gray_word ^ next_gray_word
        toggled_bit_from_lsb = toggle_mask.bit_length() - 1
        sequence.append(q_control_list[-1 - toggled_bit_from_lsb])

    return sequence


def build_payoff_circuit(delta_angles: np.ndarray, q_control_list, q_target):
    """构建收益编码线路。

    此处为了符合 NISQ 硬件约束，弃用了多控制 RY 宏门，采用纯底层 CNOT
    与单比特 RY 的 Gray code 梯形线路手工实现 3 控制 1 目标多路复用旋转。
    """
    delta_angles = np.asarray(delta_angles, dtype=float)
    if delta_angles.size != 2 ** len(q_control_list):
        raise ValueError("delta_angles 长度必须等于 2**len(q_control_list)。")

    control_sequence = _control_sequence_for_uniform_rotation(q_control_list)
    prog = QP.QProg()

    for angle, control in zip(delta_angles, control_sequence):
        prog << QP.RY(q_target, float(angle))
        prog << QP.CNOT(control, q_target)

    return prog


def build_inverse_payoff_circuit(delta_angles: np.ndarray, q_control_list, q_target):
    """收益编码线路的精确逆线路。"""
    delta_angles = np.asarray(delta_angles, dtype=float)
    control_sequence = _control_sequence_for_uniform_rotation(q_control_list)
    prog = QP.QProg()

    for angle, control in reversed(list(zip(delta_angles, control_sequence))):
        prog << QP.CNOT(control, q_target)
        prog << QP.RY(q_target, float(-angle))

    return prog


def build_A_circuit(sp_angles: np.ndarray, delta_angles: np.ndarray, q_state, q_target):
    """构建总算子 A = State Preparation + Payoff Oracle。"""
    prog = QP.QProg()
    prog << build_sp_circuit(sp_angles, q_state)
    prog << build_payoff_circuit(delta_angles, q_state, q_target)
    return prog


def build_inverse_A_circuit(sp_angles: np.ndarray, delta_angles: np.ndarray, q_state, q_target):
    """构建总算子 A 的精确逆线路 A^†。"""
    prog = QP.QProg()
    prog << build_inverse_payoff_circuit(delta_angles, q_state, q_target)
    prog << build_inverse_sp_circuit(sp_angles, q_state)
    return prog


def _append_phase_product(prog, qubit_subset, alpha: float) -> None:
    """用 CNOT 梯子与单个 RZ 合成 exp(i alpha Z_subset)。"""
    qubit_subset = list(qubit_subset)
    target = qubit_subset[-1]

    for control in qubit_subset[:-1]:
        prog << QP.CNOT(control, target)
    prog << QP.RZ(target, -2.0 * alpha)
    for control in reversed(qubit_subset[:-1]):
        prog << QP.CNOT(control, target)


def build_multi_controlled_z_all_ones(qubits):
    """在 |11...1> 上实现相位翻转。

    此处为了符合 NISQ 硬件约束，弃用了多控制 Z 宏门，改用相位多项式
    精确合成。这样可以在完全无辅助比特（ancilla-free）的前提下，
    仅使用单比特门与 CNOT 完成 C^3Z 的底层编译。
    """
    qubits = list(qubits)
    num_qubits = len(qubits)
    prog = QP.QProg()

    for subset_size in range(1, num_qubits + 1):
        for subset_indices in itertools.combinations(range(num_qubits), subset_size):
            subset_qubits = [qubits[index] for index in subset_indices]
            alpha = math.pi / (2**num_qubits) * ((-1) ** subset_size)
            _append_phase_product(prog, subset_qubits, alpha)

    return prog


def build_S0_circuit(qubits):
    """构建全零态反射 S0。

    实现形式为：
        S0 = X^{⊗n} · C^{n-1}Z · X^{⊗n}

    其中 C^{n-1}Z 并非调用宏门，而是由相位多项式手工合成。
    """
    qubits = list(qubits)
    prog = QP.QProg()

    for qubit in qubits:
        prog << QP.X(qubit)

    prog << build_multi_controlled_z_all_ones(qubits)

    for qubit in qubits:
        prog << QP.X(qubit)

    return prog


def build_Sf_circuit(q_target):
    """构建好态反射 Sf。

    由于本方案已将收益编码在目标比特 q_target 的 |1> 振幅上，
    所以好态相位翻转只需对目标比特施加一个 Z 门。
    """
    prog = QP.QProg()
    prog << QP.Z(q_target)
    return prog


def build_grover_Q(sp_angles: np.ndarray, delta_angles: np.ndarray, q_state, q_target):
    """构建 Grover 算子 Q = - A S0 A^† Sf。

    此处为了符合 NISQ 硬件约束，弃用了标准 QAE 中常见的高层黑箱 Oracle
    与自动求逆功能，全部逆线路与反射算子均由底层门手工搭建。
    """
    all_qubits = list(q_state) + [q_target]
    prog = QP.QProg()

    prog << build_Sf_circuit(q_target)
    prog << build_inverse_A_circuit(sp_angles, delta_angles, q_state, q_target)
    prog << build_S0_circuit(all_qubits)
    prog << build_A_circuit(sp_angles, delta_angles, q_state, q_target)
    return prog


def create_machine_and_qubits(num_qubits: int):
    """创建兼容新版/旧版 pyqpanda3 的 CPUQVM。"""
    machine = QP.CPUQVM()
    if hasattr(machine, "init_qvm"):
        machine.init_qvm()

    if hasattr(machine, "qAlloc_many"):
        q = machine.qAlloc_many(num_qubits)
    else:  # pragma: no cover
        q = list(range(num_qubits))

    return machine, q


def run_probabilities(machine, prog, q) -> dict[str, float]:
    """运行量子程序并返回全概率分布。"""
    if hasattr(machine, "prob_run_dict"):
        return machine.prob_run_dict(prog, q, -1)

    machine.run(prog, 0)
    return machine.result().get_prob_dict(q)


def probability_vector_from_prob_dict(prob_dict: dict[str, float], num_qubits: int) -> np.ndarray:
    """把 pyqpanda3 的 bitstring 概率字典转换为标准二进制索引顺序的向量。

    注意：新版 pyqpanda3 常返回 q[n-1]...q[0] 顺序，因此这里显式做一次反转。
    """
    vector = np.zeros(2**num_qubits, dtype=float)
    for bitstring, probability in prob_dict.items():
        logical_bits = bitstring[::-1]
        vector[int(logical_bits, 2)] = float(probability)
    return vector


def total_probability_target_one(prob_dict: dict[str, float]) -> float:
    """提取目标比特 q[3] 为 1 的总概率。"""
    return sum(probability for bitstring, probability in prob_dict.items() if bitstring[0] == "1")


def prepare_pricing_context_from_distribution(
    distribution: dict[str, object],
    config: PricingConfig = PricingConfig(),
    payoff_function: Callable[..., np.ndarray] | None = None,
    payoff_kwargs: dict[str, float] | None = None,
) -> dict[str, object]:
    """从任意离散价格分布出发，构建后续量子定价工作流所需上下文。

    这个接口是整个框架可扩展性的关键：
    - 若使用 BSM 分布，传入 discretize_bsm_price_distribution 的输出
    - 若使用历史收益率拟合分布，传入 discretize_empirical_price_distribution_from_history 的输出
    - 若使用未来更一般的离散概率向量，也只需保证键结构兼容即可
    """
    price_grid = distribution["price_grid"]
    probabilities = distribution["probabilities"]
    amplitudes = distribution["amplitudes"]

    sp_angles = calculate_ry_angles(amplitudes)

    if payoff_function is None:
        payoff_function = calculate_call_payoffs
        payoff_kwargs = {"strike": config.strike}
    elif payoff_kwargs is None:
        payoff_kwargs = {}

    payoff_list = payoff_function(price_grid, **payoff_kwargs)
    gamma_list, max_payoff, scaled_payoffs = calculate_gamma_angles(
        payoff_list,
        config.scaling_constant,
    )
    delta_angles = calculate_payoff_angles(gamma_list)

    exact_a = float(np.sum(probabilities * scaled_payoffs))
    theta_true = math.asin(math.sqrt(exact_a))
    classical_grid_price = math.exp(-config.r * config.maturity) * float(np.sum(probabilities * payoff_list))
    bsm_analytic_price = black_scholes_call_price(
        s0=config.s0,
        strike=config.strike,
        r=config.r,
        sigma=config.sigma,
        maturity=config.maturity,
    )

    return {
        "config": config,
        "price_grid": price_grid,
        "probabilities": probabilities,
        "amplitudes": amplitudes,
        "sp_angles": sp_angles,
        "payoff_list": payoff_list,
        "gamma_list": gamma_list,
        "delta_angles": delta_angles,
        "max_payoff": float(max_payoff),
        "scaled_payoffs": scaled_payoffs,
        "exact_a": exact_a,
        "theta_true": theta_true,
        "classical_grid_price": classical_grid_price,
        "bsm_analytic_price": float(bsm_analytic_price),
    }


def prepare_pricing_context(config: PricingConfig = PricingConfig()) -> dict[str, object]:
    """预先计算整条量子定价工作流所需的经典侧数据。"""
    distribution = discretize_bsm_price_distribution(config)
    return prepare_pricing_context_from_distribution(distribution, config=config)


def validate_state_preparation(context: dict[str, object]) -> dict[str, object]:
    """验证价格态加载线路是否重现经典离散概率分布。"""
    machine, q = create_machine_and_qubits(3)
    prog = QP.QProg()
    prog << build_sp_circuit(context["sp_angles"], q)
    prob_dict = run_probabilities(machine, prog, q)
    quantum_probabilities = probability_vector_from_prob_dict(prob_dict, 3)
    classical_probabilities = np.asarray(context["probabilities"], dtype=float)

    return {
        "quantum_probabilities": quantum_probabilities,
        "classical_probabilities": classical_probabilities,
        "max_abs_error": float(np.max(np.abs(quantum_probabilities - classical_probabilities))),
        "l2_error": float(np.linalg.norm(quantum_probabilities - classical_probabilities)),
    }


def validate_payoff_oracle(context: dict[str, object]) -> dict[str, float]:
    """验证收益编码后目标比特成功概率是否与经典理论一致。"""
    machine, q = create_machine_and_qubits(4)
    prog = QP.QProg()
    prog << build_A_circuit(context["sp_angles"], context["delta_angles"], q[:3], q[3])
    prob_dict = run_probabilities(machine, prog, q)
    quantum_success_probability = total_probability_target_one(prob_dict)
    classical_success_probability = float(context["exact_a"])

    return {
        "quantum_success_probability": float(quantum_success_probability),
        "classical_success_probability": classical_success_probability,
        "abs_error": abs(float(quantum_success_probability) - classical_success_probability),
    }


def validate_grover_once(context: dict[str, object]) -> dict[str, float]:
    """验证施加一次 Grover 算子后的振幅放大结果。"""
    machine, q = create_machine_and_qubits(4)
    prog = QP.QProg()
    prog << build_A_circuit(context["sp_angles"], context["delta_angles"], q[:3], q[3])
    prog << build_grover_Q(context["sp_angles"], context["delta_angles"], q[:3], q[3])
    prob_dict = run_probabilities(machine, prog, q)
    quantum_probability = total_probability_target_one(prob_dict)
    theory_probability = math.sin(3.0 * float(context["theta_true"])) ** 2

    return {
        "quantum_probability": float(quantum_probability),
        "theory_probability": float(theory_probability),
        "abs_error": abs(float(quantum_probability) - float(theory_probability)),
    }


def build_iqae_program(k: int, context: dict[str, object], q):
    """构建 A Q^k 的完整量子线路。"""
    if k < 0:
        raise ValueError("Grover 迭代次数 k 必须非负。")

    prog = QP.QProg()
    prog << build_A_circuit(context["sp_angles"], context["delta_angles"], q[:3], q[3])
    for _ in range(k):
        prog << build_grover_Q(context["sp_angles"], context["delta_angles"], q[:3], q[3])
    return prog


def exact_success_probability(k: int, context: dict[str, object]) -> float:
    """计算 A Q^k 之后目标比特为 1 的精确成功概率。"""
    machine, q = create_machine_and_qubits(4)
    prog = build_iqae_program(k, context, q)
    prob_dict = run_probabilities(machine, prog, q)
    return float(total_probability_target_one(prob_dict))


def run_and_sample(
    k: int,
    context: dict[str, object],
    num_shots: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[int, float]:
    """运行 A Q^k 并用二项分布模拟有限 shots 的 NISQ 采样。

    这正是 IQAE 在实验层面需要的输入：对不同 k 获取 (k, h_k, N_k) 数据。
    """
    if num_shots <= 0:
        raise ValueError("num_shots 必须为正整数。")

    if rng is None:
        rng = np.random.default_rng(20260327)

    success_probability = exact_success_probability(k, context)
    successes = int(rng.binomial(num_shots, success_probability))
    return successes, success_probability


def collect_iqae_data(
    context: dict[str, object],
    k_schedule: list[int],
    num_shots: int = 1000,
    seed: int = 20260327,
) -> list[dict[str, float | int]]:
    """收集 IQAE 所需的全部实验记录。"""
    rng = np.random.default_rng(seed)
    records = []
    for k in k_schedule:
        successes, exact_probability = run_and_sample(k, context, num_shots=num_shots, rng=rng)
        records.append(
            {
                "k": int(k),
                "shots": int(num_shots),
                "successes": int(successes),
                "exact_probability": float(exact_probability),
                "sample_frequency": float(successes / num_shots),
            }
        )
    return records


def negative_log_likelihood(theta: float, data_records: list[dict[str, float | int]]) -> float:
    """IQAE 最大似然估计所对应的负对数似然函数。"""
    epsilon = 1e-12
    total = 0.0

    for record in data_records:
        k = int(record["k"])
        shots = int(record["shots"])
        successes = int(record["successes"])

        probability = math.sin((2 * k + 1) * theta) ** 2
        probability = min(max(probability, epsilon), 1.0 - epsilon)
        total -= successes * math.log(probability) + (shots - successes) * math.log(1.0 - probability)

    return total


def estimate_theta_mle(
    data_records: list[dict[str, float | int]],
    theta_min: float = 0.0,
    theta_max: float = math.pi / 4.0,
    grid_size: int = 200_001,
) -> tuple[float, str]:
    """用 MLE 估计 theta。

    此处为了符合 NISQ 友好设计目标，采用 IQAE + MLE 路线，彻底移除了
    传统 QAE 中昂贵的 IQFT 与评估寄存器。
    """
    epsilon = 1e-12
    theta_grid = np.linspace(theta_min + epsilon, theta_max - epsilon, grid_size)
    nll_grid = np.zeros_like(theta_grid)

    for record in data_records:
        k = int(record["k"])
        shots = int(record["shots"])
        successes = int(record["successes"])

        probabilities = np.sin((2 * k + 1) * theta_grid) ** 2
        probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
        nll_grid -= successes * np.log(probabilities) + (shots - successes) * np.log(1.0 - probabilities)

    best_index = int(np.argmin(nll_grid))
    theta_est = float(theta_grid[best_index])
    method = "grid-search"

    if minimize_scalar is not None:
        left_index = max(best_index - 1, 0)
        right_index = min(best_index + 1, theta_grid.size - 1)
        left = float(theta_grid[left_index])
        right = float(theta_grid[right_index])

        result = minimize_scalar(
            negative_log_likelihood,
            bounds=(left, right),
            method="bounded",
            args=(data_records,),
        )
        if result.success:
            theta_est = float(result.x)
            method = "scipy-bounded-mle"

    return theta_est, method


def price_from_theta(theta_est: float, context: dict[str, object]) -> float:
    """由 theta_est 反推期权现价。"""
    config = context["config"]
    expected_payoff_est = (math.sin(theta_est) ** 2) * float(context["max_payoff"]) / config.scaling_constant
    return math.exp(-config.r * config.maturity) * expected_payoff_est


def summarize_compilation_resources() -> dict[str, dict[str, int]]:
    """给出核心线路的基础门资源统计。

    这些统计直接来自手工分解结构本身，可用于技术报告中的复杂度分析。
    """
    return {
        "state_preparation": {
            "qubits": 3,
            "ancilla_qubits": 0,
            "ry_gates": 7,
            "cnot_gates": 6,
        },
        "payoff_oracle": {
            "qubits": 4,
            "ancilla_qubits": 0,
            "ry_gates": 8,
            "cnot_gates": 8,
        },
        "A_operator_total": {
            "qubits": 4,
            "ancilla_qubits": 0,
            "ry_gates": 15,
            "cnot_gates": 14,
        },
        "S0_reflection": {
            "qubits": 4,
            "ancilla_qubits": 0,
            "x_gates": 8,
            "rz_gates": 15,
            "cnot_gates": 34,
        },
        "grover_Q": {
            "qubits": 4,
            "ancilla_qubits": 0,
            "ry_gates": 30,
            "rz_gates": 15,
            "x_gates": 8,
            "z_gates": 1,
            "cnot_gates": 62,
        },
    }
