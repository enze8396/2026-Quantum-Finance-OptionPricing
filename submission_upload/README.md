# 2026-Quantum-Finance-OptionPricing

2026 年 CIC“悟空杯”量子金融赛项。

## 核心交付物

- `technical_report.pdf`
  最终技术报告 PDF。

- `technical_report.tex`
  技术报告 LaTeX 源文件。

- `quantum_option_pricing_core.py`
  量子期权定价核心模块。包含状态加载、收益编码、Grover 算子、IQAE、历史数据经验分布拟合接口。

- `run_spy_option_pricing_demo.py`
  主演示脚本。基于 BSM 离散分布完成欧式看涨期权定价。

- `historical_returns_quantum_demo.py`
  附加分脚本。基于真实历史收益率拟合经验分布，并直接接入同一套量子线路。

- `data/spy_history_stooq.csv`
  真实 SPY 历史日线 CSV 示例，来源于公开下载接口。

## 方案亮点

1. 仅使用 4 个量子比特。
2. 完全无辅助比特（ancilla-free）。
3. 不调用任意态加载宏门、多控制旋转宏门、多控制 Z 宏门。
4. Gray code 与相位多项式全部手工分解到底层门。
5. 使用 IQAE + MLE 取代标准 QAE + IQFT。
6. 支持奇异期权扩展与非 BSM 分布输入。
7. 支持真实历史价格 CSV 直接驱动量子定价流程。

## 运行方式

在 `qfinance` 环境中：

```bash
python run_spy_option_pricing_demo.py
```

如需演示真实历史收益率拟合：

```bash
python historical_returns_quantum_demo.py --csv data/spy_history_stooq.csv
```

如需演示向上敲出看涨期权：

```bash
python historical_returns_quantum_demo.py --csv data/spy_history_stooq.csv --use-barrier
```

