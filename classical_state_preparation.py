"""Classical preprocessing for SPY price loading under the BSM model."""

from __future__ import annotations

import numpy as np


def lognormal_price_pdf(
    prices: np.ndarray,
    s0: float,
    r: float,
    sigma: float,
    maturity: float,
) -> np.ndarray:
    """Return the BSM log-normal density evaluated at the given prices."""
    if s0 <= 0:
        raise ValueError("s0 must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if maturity <= 0:
        raise ValueError("maturity must be positive.")
    if np.any(prices <= 0):
        raise ValueError("All price grid points must be positive.")

    mu = np.log(s0) + (r - 0.5 * sigma**2) * maturity
    std = sigma * np.sqrt(maturity)

    log_term = (np.log(prices) - mu) / std
    normalizer = prices * std * np.sqrt(2.0 * np.pi)
    return np.exp(-0.5 * log_term**2) / normalizer


def discretize_bsm_price_distribution(
    s0: float = 500.0,
    r: float = 0.05,
    sigma: float = 0.15,
    maturity: float = 30.0 / 365.0,
    num_qubits: int = 3,
    truncation_sigma: float = 3.0,
) -> dict[str, np.ndarray | float | int]:
    """Discretize the BSM price distribution on a uniform price grid."""
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive.")
    if truncation_sigma <= 0:
        raise ValueError("truncation_sigma must be positive.")

    num_points = 2**num_qubits
    sqrt_t = np.sqrt(maturity)

    price_min = s0 * np.exp(-truncation_sigma * sigma * sqrt_t)
    price_max = s0 * np.exp(truncation_sigma * sigma * sqrt_t)
    price_grid = np.linspace(price_min, price_max, num_points)

    pdf_values = lognormal_price_pdf(
        prices=price_grid,
        s0=s0,
        r=r,
        sigma=sigma,
        maturity=maturity,
    )

    grid_spacing = price_grid[1] - price_grid[0]
    unnormalized_probabilities = pdf_values * grid_spacing
    probabilities = unnormalized_probabilities / np.sum(unnormalized_probabilities)
    amplitudes = np.sqrt(probabilities)

    return {
        "num_qubits": num_qubits,
        "num_points": num_points,
        "price_min": price_min,
        "price_max": price_max,
        "price_grid": price_grid,
        "pdf_values": pdf_values,
        "probabilities": probabilities,
        "amplitudes": amplitudes,
    }


if __name__ == "__main__":
    result = discretize_bsm_price_distribution()

    np.set_printoptions(precision=8, suppress=True)

    print("Price grid:")
    print(result["price_grid"])
    print("\nNormalized probabilities:")
    print(result["probabilities"])
    print("\nAmplitudes:")
    print(result["amplitudes"])
    print("\nProbability sum:", np.sum(result["probabilities"]))
    print("Amplitude norm:", np.sum(result["amplitudes"] ** 2))
