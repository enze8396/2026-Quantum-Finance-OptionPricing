r"""# End-to-End SPY Option Pricing via IQAE

This script implements an iterative quantum amplitude estimation style workflow
on top of the manually compiled operators from the previous stages.

Let the success probability after the state-loading operator $\mathcal{A}$ be

\[
a=\sin^2(\theta).
\]

After applying the Grover iterate $\mathcal{Q}$ exactly $k$ times, the ideal
success probability becomes

\[
p_k(\theta)=\sin^2((2k+1)\theta).
\]

For each chosen Grover depth $k$, we perform a finite-shot Bernoulli sampling
experiment and observe $h_k$ successes out of $N_k$ shots. Under the binomial
model, the likelihood of $\theta$ is

\[
L(\theta)\propto \prod_k p_k(\theta)^{h_k}\left(1-p_k(\theta)\right)^{N_k-h_k}.
\]

Equivalently, we maximize the log-likelihood

\[
\log L(\theta)=\sum_k
\left[
h_k\log p_k(\theta)+(N_k-h_k)\log(1-p_k(\theta))
\right],
\qquad \theta\in[0,\pi/4].
\]

Once the maximum-likelihood estimator $\theta_{\mathrm{est}}$ is obtained, the
expected payoff is reconstructed by

\[
\mathbb{E}[\mathrm{payoff}]
=
\sin^2(\theta_{\mathrm{est}})\cdot \frac{\max(\mathrm{payoff})}{c},
\]

and the option price is discounted as

\[
V_0=e^{-rT}\,\mathbb{E}[\mathrm{payoff}].
\]

The script below uses the exact simulator probability to emulate NISQ sampling
through `numpy.random.binomial`, and then performs one-dimensional MLE either
with SciPy (if available) or with a dense grid-search fallback.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import types

import numpy as np

try:
    from scipy.optimize import minimize_scalar
except Exception:  # pragma: no cover - grid-search fallback remains available
    minimize_scalar = None


def _load_grover_module():
    """Load the manually compiled Grover-operator module from the local file.

    This helper supports both:
    - normal `.py` execution, where `__file__` exists
    - Jupyter notebooks, where `__file__` is usually undefined
    """
    required_names = [
        "QP",
        "discretize_bsm_price_distribution",
        "calculate_ry_angles",
        "calculate_call_payoffs",
        "calculate_gamma_angles",
        "calculate_payoff_angles",
        "create_machine_and_qubits",
        "build_A_circuit",
        "build_grover_Q",
        "run_probabilities",
        "total_probability_target_one",
    ]

    if all(name in globals() for name in required_names):
        return types.SimpleNamespace(**{name: globals()[name] for name in required_names})

    candidate_paths = []
    if "__file__" in globals():
        candidate_paths.append(Path(__file__).resolve().with_name("04_SPY_Grover_Operator.py"))
    candidate_paths.append((Path.cwd() / "04_SPY_Grover_Operator.py").resolve())

    checked_paths = []
    seen_paths = set()

    for module_path in candidate_paths:
        module_path_str = str(module_path)
        if module_path_str in seen_paths:
            continue
        seen_paths.add(module_path_str)
        checked_paths.append(module_path_str)

        if not module_path.exists():
            continue

        spec = importlib.util.spec_from_file_location("spy_grover_operator", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module from {module_path}.")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    raise FileNotFoundError(
        "Unable to locate `04_SPY_Grover_Operator.py`. "
        "Please keep it in the current working directory or the same directory as this script. "
        f"Searched paths: {checked_paths}"
    )


core = _load_grover_module()

IQAE_CONTEXT: dict[str, object] = {}
SUCCESS_PROBABILITY_CACHE: dict[int, float] = {}
RNG = np.random.default_rng(20260327)


def initialize_iqae_context(
    strike: float = 500.0,
    scaling_constant: float = 0.5,
    s0: float = 500.0,
    r: float = 0.05,
    sigma: float = 0.15,
    maturity: float = 30.0 / 365.0,
    seed: int = 20260327,
) -> dict[str, object]:
    """Prepare all classical inputs and cached circuit angles for IQAE."""
    global IQAE_CONTEXT, SUCCESS_PROBABILITY_CACHE, RNG

    distribution = core.discretize_bsm_price_distribution(
        s0=s0,
        r=r,
        sigma=sigma,
        maturity=maturity,
        num_qubits=3,
        truncation_sigma=3.0,
    )

    price_grid = distribution["price_grid"]
    price_probabilities = distribution["probabilities"]
    price_amplitudes = distribution["amplitudes"]

    sp_angles = core.calculate_ry_angles(price_amplitudes)

    payoff_list = core.calculate_call_payoffs(price_grid, strike=strike)
    gamma_list, max_payoff, scaled_payoffs = core.calculate_gamma_angles(
        payoff_list,
        scaling_constant=scaling_constant,
    )
    delta_angles = core.calculate_payoff_angles(gamma_list)

    exact_a = float(np.sum(price_probabilities * scaled_payoffs))
    classical_grid_price = math.exp(-r * maturity) * float(np.sum(price_probabilities * payoff_list))

    IQAE_CONTEXT = {
        "strike": strike,
        "scaling_constant": scaling_constant,
        "s0": s0,
        "r": r,
        "sigma": sigma,
        "maturity": maturity,
        "price_grid": price_grid,
        "price_probabilities": price_probabilities,
        "price_amplitudes": price_amplitudes,
        "payoff_list": payoff_list,
        "max_payoff": max_payoff,
        "scaled_payoffs": scaled_payoffs,
        "gamma_list": gamma_list,
        "delta_angles": delta_angles,
        "sp_angles": sp_angles,
        "exact_a": exact_a,
        "theta_true": math.asin(math.sqrt(exact_a)),
        "classical_grid_price": classical_grid_price,
    }

    SUCCESS_PROBABILITY_CACHE = {}
    RNG = np.random.default_rng(seed)
    return IQAE_CONTEXT


def exact_success_probability(k: int) -> float:
    """Compute the exact success probability after A Q^k using the simulator."""
    if k < 0:
        raise ValueError("k must be nonnegative.")

    if k in SUCCESS_PROBABILITY_CACHE:
        return SUCCESS_PROBABILITY_CACHE[k]

    if not IQAE_CONTEXT:
        raise RuntimeError("IQAE context has not been initialized.")

    machine, q = core.create_machine_and_qubits(4)

    prog = core.QP.QProg()
    prog << core.build_A_circuit(
        IQAE_CONTEXT["sp_angles"],
        IQAE_CONTEXT["delta_angles"],
        q[:3],
        q[3],
    )

    for _ in range(k):
        prog << core.build_grover_Q(
            IQAE_CONTEXT["sp_angles"],
            IQAE_CONTEXT["delta_angles"],
            q[:3],
            q[3],
        )

    prob_dict = core.run_probabilities(machine, prog, q)
    success_probability = core.total_probability_target_one(prob_dict)

    SUCCESS_PROBABILITY_CACHE[k] = success_probability
    return success_probability


def run_and_sample(k: int, num_shots: int = 1000) -> tuple[int, float]:
    """Run A Q^k exactly, then emulate finite-shot NISQ sampling via binomial draws."""
    if num_shots <= 0:
        raise ValueError("num_shots must be positive.")

    success_probability = exact_success_probability(k)
    h_k = int(RNG.binomial(num_shots, success_probability))
    return h_k, success_probability


def collect_iqae_data(k_schedule: list[int], num_shots: int = 1000) -> list[dict[str, float | int]]:
    """Collect all (k, h_k, N_k) data records for the chosen IQAE schedule."""
    records = []
    for k in k_schedule:
        h_k, success_probability = run_and_sample(k, num_shots=num_shots)
        records.append(
            {
                "k": int(k),
                "shots": int(num_shots),
                "successes": int(h_k),
                "exact_probability": float(success_probability),
                "sample_frequency": float(h_k / num_shots),
            }
        )
    return records


def negative_log_likelihood(theta: float, data_records: list[dict[str, float | int]]) -> float:
    """Return the negative log-likelihood for the IQAE binomial model."""
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
    """Estimate theta by maximizing the total likelihood over all k records."""
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


def price_from_theta(theta_est: float) -> float:
    """Reconstruct the discounted option price from the estimated amplitude angle."""
    if not IQAE_CONTEXT:
        raise RuntimeError("IQAE context has not been initialized.")

    scaling_constant = float(IQAE_CONTEXT["scaling_constant"])
    max_payoff = float(IQAE_CONTEXT["max_payoff"])
    r = float(IQAE_CONTEXT["r"])
    maturity = float(IQAE_CONTEXT["maturity"])

    expected_payoff_est = (math.sin(theta_est) ** 2) * max_payoff / scaling_constant
    return math.exp(-r * maturity) * expected_payoff_est


if __name__ == "__main__":
    np.set_printoptions(precision=8, suppress=True)

    k_schedule = [0, 1, 2, 4, 8]
    num_shots = 1000

    context = initialize_iqae_context()
    data_records = collect_iqae_data(k_schedule, num_shots=num_shots)
    theta_est, mle_method = estimate_theta_mle(data_records)

    theta_true = float(context["theta_true"])
    exact_a = float(context["exact_a"])
    classical_grid_price = float(context["classical_grid_price"])
    quantum_price = price_from_theta(theta_est)
    price_abs_error = abs(quantum_price - classical_grid_price)

    print("Price grid:")
    print(context["price_grid"])
    print("\nPayoff list:")
    print(context["payoff_list"])
    print("\nTrue amplitude a = sin^2(theta):")
    print(exact_a)
    print("True theta:")
    print(theta_true)
    print("\nIQAE schedule:")
    print(k_schedule)
    print("Shots per k:")
    print(num_shots)
    print("\nData records:")
    for record in data_records:
        print(record)
    print("\nMLE method:")
    print(mle_method)
    print("Estimated theta:")
    print(theta_est)
    print("Absolute theta error:")
    print(abs(theta_est - theta_true))
    print("\nQuantum Price:")
    print(quantum_price)
    print("Classical Grid Price:")
    print(classical_grid_price)
    print("Absolute pricing error:")
    print(price_abs_error)
