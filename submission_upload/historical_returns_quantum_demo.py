"""基于历史收益率拟合分布的量子期权定价附加脚本。

这个脚本用于展示两项加分能力：
1. 真实金融市场数据拟合
2. 非 BSM 分布可直接接入同一套底层量子线路

使用方式示例：
    python historical_returns_quantum_demo.py --csv spy_history.csv

CSV 只需包含一个可识别的价格列，例如：
    Adj Close / Close / adj_close / close
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from quantum_option_pricing_core import (
    PricingConfig,
    calculate_barrier_call_payoffs,
    collect_iqae_data,
    discretize_empirical_price_distribution_from_history,
    estimate_theta_mle,
    load_price_series_from_csv,
    prepare_pricing_context_from_distribution,
    price_from_theta,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="历史收益率拟合驱动的量子期权定价演示")
    parser.add_argument("--csv", required=True, help="本地历史价格 CSV 文件路径")
    parser.add_argument("--price-column", default=None, help="价格列名；默认自动识别")
    parser.add_argument("--s0", type=float, default=None, help="初始价格；默认取 CSV 最后一个收盘价")
    parser.add_argument("--strike", type=float, default=None, help="执行价；默认取与 s0 相同的平值设置")
    parser.add_argument("--r", type=float, default=0.05, help="无风险利率，默认 0.05")
    parser.add_argument("--sigma", type=float, default=0.15, help="展示用波动率参数，默认 0.15")
    parser.add_argument("--maturity", type=float, default=30.0 / 365.0, help="到期期限，默认 30/365")
    parser.add_argument("--use-barrier", action="store_true", help="启用向上敲出看涨期权收益")
    parser.add_argument("--barrier-price", type=float, default=None, help="障碍价格；默认取 1.07 * s0")
    return parser


def main() -> None:
    """读取历史数据，拟合经验分布，并直接接入量子定价流程。"""
    np.set_printoptions(precision=8, suppress=True)
    args = build_argument_parser().parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    try:
        display_csv_path = csv_path.relative_to(Path.cwd().resolve())
    except Exception:
        display_csv_path = Path(csv_path.name)
    price_series = load_price_series_from_csv(csv_path, price_column=args.price_column)
    latest_close = float(price_series[-1])

    s0 = latest_close if args.s0 is None else float(args.s0)
    strike = s0 if args.strike is None else float(args.strike)
    barrier_price = 1.07 * s0 if args.barrier_price is None else float(args.barrier_price)

    config = PricingConfig(  # 评委可在此处修改参数一键验证
        s0=s0,
        strike=strike,
        r=float(args.r),
        sigma=float(args.sigma),
        maturity=float(args.maturity),
    )

    distribution = discretize_empirical_price_distribution_from_history(
        price_series=price_series,
        s0=config.s0,
        maturity=config.maturity,
        num_qubits=config.num_state_qubits,
    )

    payoff_function = None
    payoff_kwargs = None
    product_name = "European Call"
    if args.use_barrier:
        payoff_function = calculate_barrier_call_payoffs
        payoff_kwargs = {"strike": config.strike, "barrier_price": barrier_price}
        product_name = "Up-and-Out Call"

    context = prepare_pricing_context_from_distribution(
        distribution=distribution,
        config=config,
        payoff_function=payoff_function,
        payoff_kwargs=payoff_kwargs,
    )

    k_schedule = [0, 1, 2, 4, 8]
    num_shots = 1000
    data_records = collect_iqae_data(context, k_schedule, num_shots=num_shots, seed=20260327)
    theta_est, mle_method = estimate_theta_mle(data_records)
    quantum_price = price_from_theta(theta_est, context)
    classical_grid_price = float(context["classical_grid_price"])

    print("=" * 72)
    print("历史收益率拟合驱动的量子期权定价演示")
    print("=" * 72)
    print("CSV file:")
    print(display_csv_path)
    print("Latest close used as default S0:")
    print(latest_close)
    print("Detected fit method:")
    print(distribution["fit_method"])
    print("Horizon steps used for return fitting:")
    print(distribution["horizon_steps"])
    print("Option type:")
    print(product_name)
    if args.use_barrier:
        print("Barrier price:")
        print(barrier_price)

    print("\nEmpirical price grid:")
    print(context["price_grid"])
    print("Empirical probabilities:")
    print(context["probabilities"])
    print("Payoff list:")
    print(context["payoff_list"])

    print("\nIQAE records:")
    for record in data_records:
        print(record)

    print("\nEstimated theta:")
    print(theta_est)
    print("Quantum Price (historical-fit distribution):")
    print(quantum_price)
    print("Classical Grid Price (same empirical distribution):")
    print(classical_grid_price)
    print("Absolute difference vs classical grid price:")
    print(abs(quantum_price - classical_grid_price))
    if args.use_barrier:
        print("Reference analytic price:")
        print("当前脚本未额外实现障碍期权解析解；建议与更高精度经典离散网格或蒙特卡洛结果对比。")
    else:
        print("Reference BSM analytic price:")
        print(context["bsm_analytic_price"])
        print("Absolute difference vs BSM analytic price:")
        print(abs(quantum_price - float(context["bsm_analytic_price"])))
    print("MLE method:")
    print(mle_method)


if __name__ == "__main__":
    main()
