"""SPY 欧式看涨期权端到端演示脚本（比赛提交入口）。

推荐评委或复现实验时直接运行本文件。
它会依次输出：

1. 经典端价格离散化结果
2. 状态加载精度验证
3. 收益编码精度验证
4. Grover 单次放大验证
5. IQAE + MLE 的最终期权定价结果
6. 与经典离散网格价格、BSM 解析解的对比

整套线路始终保持：
- 仅 4 个量子比特（3 个状态比特 + 1 个目标比特）
- 不使用任何辅助比特（ancilla-free）
- 不调用任何高级宏门
"""

from __future__ import annotations

import numpy as np

from quantum_option_pricing_core import (
    PricingConfig,
    collect_iqae_data,
    estimate_theta_mle,
    prepare_pricing_context,
    price_from_theta,
    summarize_compilation_resources,
    validate_grover_once,
    validate_payoff_oracle,
    validate_state_preparation,
)


def main() -> None:
    """运行完整的比赛版端到端流程。"""
    np.set_printoptions(precision=8, suppress=True)

    config = PricingConfig(  # 评委可在此处修改参数一键验证
        s0=500.0,
        strike=500.0,
        r=0.05,
        sigma=0.15,
        maturity=30.0 / 365.0,
    )
    context = prepare_pricing_context(config)

    state_validation = validate_state_preparation(context)
    payoff_validation = validate_payoff_oracle(context)
    grover_validation = validate_grover_once(context)

    k_schedule = [0, 1, 2, 4, 8]
    num_shots = 1000
    data_records = collect_iqae_data(context, k_schedule, num_shots=num_shots, seed=20260327)
    theta_est, mle_method = estimate_theta_mle(data_records)

    quantum_price = price_from_theta(theta_est, context)
    classical_grid_price = float(context["classical_grid_price"])
    bsm_analytic_price = float(context["bsm_analytic_price"])
    resource_summary = summarize_compilation_resources()

    print("=" * 72)
    print("SPY 欧式看涨期权量子定价（NISQ 友好版）")
    print("=" * 72)
    print("方案亮点：")
    print("1. 仅使用 4 个量子比特，且完全无辅助比特。")
    print("2. 所有复杂受控操作均由底层单比特门与 CNOT 手工编译。")
    print("3. 采用 IQAE + MLE，彻底移除传统 QAE 的 IQFT。")

    print("\n[经典离散化结果]")
    print("Price grid:")
    print(context["price_grid"])
    print("Normalized probabilities:")
    print(context["probabilities"])
    print("Amplitudes:")
    print(context["amplitudes"])
    print("Payoff list:")
    print(context["payoff_list"])

    print("\n[状态加载精度验证]")
    print("Quantum probabilities:")
    print(state_validation["quantum_probabilities"])
    print("Classical probabilities:")
    print(state_validation["classical_probabilities"])
    print("Max abs error:")
    print(state_validation["max_abs_error"])
    print("L2 error:")
    print(state_validation["l2_error"])

    print("\n[收益编码精度验证]")
    print("Quantum success probability P(q[3]=1 | A):")
    print(payoff_validation["quantum_success_probability"])
    print("Classical theory:")
    print(payoff_validation["classical_success_probability"])
    print("Absolute error:")
    print(payoff_validation["abs_error"])

    print("\n[Grover 单次放大验证]")
    print("Quantum success probability P(q[3]=1 | A Q):")
    print(grover_validation["quantum_probability"])
    print("Amplitude amplification theory sin^2(3 theta):")
    print(grover_validation["theory_probability"])
    print("Absolute error:")
    print(grover_validation["abs_error"])

    print("\n[IQAE 采样记录]")
    print("k schedule:")
    print(k_schedule)
    print("Shots per k:")
    print(num_shots)
    for record in data_records:
        print(record)

    print("\n[最终定价结果]")
    print("MLE method:")
    print(mle_method)
    print("True theta:")
    print(context["theta_true"])
    print("Estimated theta:")
    print(theta_est)
    print("Absolute theta error:")
    print(abs(theta_est - float(context["theta_true"])))
    print("Quantum Price:")
    print(quantum_price)
    print("Classical Grid Price:")
    print(classical_grid_price)
    print("BSM Analytic Price:")
    print(bsm_analytic_price)
    print("Absolute error vs grid price:")
    print(abs(quantum_price - classical_grid_price))
    print("Absolute error vs BSM analytic price:")
    print(abs(quantum_price - bsm_analytic_price))

    print("\n[底层资源统计]")
    for module_name, resources in resource_summary.items():
        print(module_name, resources)


if __name__ == "__main__":
    main()
