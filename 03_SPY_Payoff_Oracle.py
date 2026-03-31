"""State preparation plus payoff encoding for a 3-qubit SPY call option model.

This script is fully manual:
- only pyqpanda3 is used for quantum circuits
- only RY and CNOT are used to build the circuits
- no packaged controlled rotations or state-loader macros are used
"""

from __future__ import annotations

import numpy as np
import pyqpanda3 as pq

QP = pq if all(hasattr(pq, name) for name in ("CPUQVM", "QProg", "RY", "CNOT")) else pq.core


def lognormal_price_pdf(
    prices: np.ndarray,
    s0: float,
    r: float,
    sigma: float,
    maturity: float,
) -> np.ndarray:
    """BSM log-normal density for terminal price S_T."""
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
) -> dict[str, np.ndarray | float]:
    """Discretize the BSM price density on a uniform price grid."""
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
        "price_grid": price_grid,
        "probabilities": probabilities,
        "amplitudes": amplitudes,
    }


def _safe_theta(left_weight: float, right_weight: float) -> float:
    """Return 2*atan2(right, left), with 0 for the degenerate zero-zero case."""
    if np.isclose(left_weight, 0.0) and np.isclose(right_weight, 0.0):
        return 0.0
    return 2.0 * np.arctan2(right_weight, left_weight)


def calculate_ry_angles(amplitudes: np.ndarray) -> np.ndarray:
    """Compute the 7 binary-tree RY angles for 3-qubit real state preparation."""
    amplitudes = np.asarray(amplitudes, dtype=float)
    if amplitudes.shape != (8,):
        raise ValueError("For 3 qubits, amplitudes must be a length-8 vector.")
    if np.any(amplitudes < -1e-12):
        raise ValueError("This loader assumes nonnegative real amplitudes.")

    norm = np.linalg.norm(amplitudes)
    if np.isclose(norm, 0.0):
        raise ValueError("Amplitude vector must not be the zero vector.")
    amplitudes = amplitudes / norm

    a0, a1, a2, a3, a4, a5, a6, a7 = amplitudes

    theta0 = _safe_theta(
        np.linalg.norm([a0, a1, a2, a3]),
        np.linalg.norm([a4, a5, a6, a7]),
    )
    theta1 = _safe_theta(np.linalg.norm([a0, a1]), np.linalg.norm([a2, a3]))
    theta2 = _safe_theta(np.linalg.norm([a4, a5]), np.linalg.norm([a6, a7]))
    theta3 = _safe_theta(a0, a1)
    theta4 = _safe_theta(a2, a3)
    theta5 = _safe_theta(a4, a5)
    theta6 = _safe_theta(a6, a7)

    return np.array([theta0, theta1, theta2, theta3, theta4, theta5, theta6])


def _append_single_control_ry_multiplexor(
    prog,
    control,
    target,
    angle_if_control_0: float,
    angle_if_control_1: float,
) -> None:
    """Compile a 1-control RY multiplexor into RY + CNOT only."""
    alpha0 = 0.5 * (angle_if_control_0 + angle_if_control_1)
    alpha1 = 0.5 * (angle_if_control_0 - angle_if_control_1)

    prog << QP.RY(target, alpha0)
    prog << QP.CNOT(control, target)
    prog << QP.RY(target, alpha1)
    prog << QP.CNOT(control, target)


def _append_double_control_ry_multiplexor(
    prog,
    control_msb,
    control_lsb,
    target,
    angles_by_branch: np.ndarray,
) -> None:
    """Compile a 2-control RY multiplexor into RY + CNOT only."""
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


def build_sp_circuit(angles: np.ndarray, q):
    """Build the 3-qubit price-loading circuit using only RY and CNOT."""
    if len(q) != 3:
        raise ValueError("State preparation expects exactly 3 qubits.")

    theta0, theta1, theta2, theta3, theta4, theta5, theta6 = np.asarray(angles, dtype=float)

    prog = QP.QProg()
    prog << QP.RY(q[0], theta0)

    _append_single_control_ry_multiplexor(
        prog=prog,
        control=q[0],
        target=q[1],
        angle_if_control_0=theta1,
        angle_if_control_1=theta2,
    )

    _append_double_control_ry_multiplexor(
        prog=prog,
        control_msb=q[0],
        control_lsb=q[1],
        target=q[2],
        angles_by_branch=np.array([theta3, theta4, theta5, theta6]),
    )

    return prog


def calculate_call_payoffs(price_grid: np.ndarray, strike: float = 500.0) -> np.ndarray:
    """European call payoff max(S-K, 0) on the discrete price grid."""
    return np.maximum(np.asarray(price_grid, dtype=float) - strike, 0.0)


def calculate_gamma_angles(
    payoff_list: np.ndarray,
    scaling_constant: float = 0.5,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Convert payoffs into branch rotation angles gamma_i.

    gamma_i is defined by
    sin^2(gamma_i / 2) = c * payoff_i / max_payoff.
    """
    payoff_list = np.asarray(payoff_list, dtype=float)
    if payoff_list.ndim != 1:
        raise ValueError("payoff_list must be a one-dimensional array.")
    if scaling_constant <= 0.0 or scaling_constant > 1.0:
        raise ValueError("scaling_constant must lie in (0, 1].")

    max_payoff = float(np.max(payoff_list))
    if np.isclose(max_payoff, 0.0):
        scaled_payoffs = np.zeros_like(payoff_list)
        gamma_list = np.zeros_like(payoff_list)
    else:
        scaled_payoffs = scaling_constant * payoff_list / max_payoff
        if np.any(scaled_payoffs > 1.0 + 1e-12):
            raise ValueError("Scaled payoff exceeds 1, decrease the scaling constant.")
        scaled_payoffs = np.clip(scaled_payoffs, 0.0, 1.0)
        gamma_list = 2.0 * np.arcsin(np.sqrt(scaled_payoffs))

    return gamma_list, max_payoff, scaled_payoffs


def gray_code_sequence(num_controls: int) -> list[int]:
    """Return the standard binary-reflected Gray code sequence."""
    return [index ^ (index >> 1) for index in range(2**num_controls)]


def _gray_ordered_walsh_hadamard_matrix(num_controls: int) -> np.ndarray:
    """Return the Gray-ordered Walsh-Hadamard transform matrix.

    For a k-control uniformly controlled RY rotation, the base angles satisfy

        delta = M * gamma,

    where

        M[r, c] = 2^{-k} (-1)^{popcount(gray_r & c)}.

    Row r is indexed by the r-th Gray code word, while column c is indexed by
    the standard binary branch label.
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
    """Convert branch angles gamma_i into Gray-code multiplexor angles delta_i."""
    gamma_list = np.asarray(gamma_list, dtype=float)
    if gamma_list.ndim != 1:
        raise ValueError("gamma_list must be one-dimensional.")

    num_branches = gamma_list.size
    num_controls = int(np.log2(num_branches))
    if 2**num_controls != num_branches:
        raise ValueError("gamma_list length must be a power of two.")

    transform = _gray_ordered_walsh_hadamard_matrix(num_controls)
    return transform @ gamma_list


def _control_for_gray_transition(
    q_control_list,
    current_gray_word: int,
    next_gray_word: int,
):
    """Map a Gray-code bit flip to the corresponding control qubit.

    q_control_list is ordered as [MSB, ..., LSB].
    """
    toggle_mask = current_gray_word ^ next_gray_word
    if toggle_mask == 0 or toggle_mask & (toggle_mask - 1):
        raise ValueError("Gray-code transitions must flip exactly one bit.")

    toggled_bit_from_lsb = toggle_mask.bit_length() - 1
    return q_control_list[-1 - toggled_bit_from_lsb]


def build_payoff_circuit(delta_angles: np.ndarray, q_control_list, q_target):
    """Build a 3-control uniformly controlled RY payoff circuit.

    The circuit alternates RY and CNOT gates in Gray-code order:
    RY(delta_0) -> CNOT -> RY(delta_1) -> CNOT -> ... -> RY(delta_7) -> CNOT.
    """
    delta_angles = np.asarray(delta_angles, dtype=float)
    num_controls = len(q_control_list)
    if delta_angles.size != 2**num_controls:
        raise ValueError("delta_angles length must be 2**len(q_control_list).")

    gray_codes = gray_code_sequence(num_controls)
    prog = QP.QProg()

    for index, angle in enumerate(delta_angles):
        prog << QP.RY(q_target, float(angle))

        current_gray_word = gray_codes[index]
        next_gray_word = gray_codes[(index + 1) % len(gray_codes)]
        control_qubit = _control_for_gray_transition(
            q_control_list=q_control_list,
            current_gray_word=current_gray_word,
            next_gray_word=next_gray_word,
        )
        prog << QP.CNOT(control_qubit, q_target)

    return prog


def create_machine_and_qubits(num_qubits: int):
    """Create a CPUQVM and a qubit container compatible with old/new pyqpanda3 APIs."""
    machine = QP.CPUQVM()
    if hasattr(machine, "init_qvm"):
        machine.init_qvm()

    if hasattr(machine, "qAlloc_many"):
        q = machine.qAlloc_many(num_qubits)
    else:
        q = list(range(num_qubits))

    return machine, q


def run_probabilities(machine, prog, q) -> dict[str, float]:
    """Run a quantum program and return the full probability dictionary."""
    if hasattr(machine, "prob_run_dict"):
        return machine.prob_run_dict(prog, q, -1)

    machine.run(prog, 0)
    result = machine.result()
    return result.get_prob_dict(q)


def build_expected_full_distribution(
    price_probabilities: np.ndarray,
    scaled_payoffs: np.ndarray,
) -> dict[str, float]:
    """Return the expected 4-qubit probability distribution.

    The qubit list is assumed to be [q0, q1, q2, q3], while pyqpanda3 prints
    bitstrings as q3 q2 q1 q0 when get_prob_dict([0,1,2,3]) is used.
    """
    expected = {}
    for branch_index, (branch_prob, scaled_payoff) in enumerate(zip(price_probabilities, scaled_payoffs)):
        branch_bits = format(branch_index, "03b")  # q0 q1 q2 in the math convention
        simulator_control_bits = branch_bits[::-1]  # q2 q1 q0 in pyqpanda3 output strings

        expected["0" + simulator_control_bits] = branch_prob * (1.0 - scaled_payoff)
        expected["1" + simulator_control_bits] = branch_prob * scaled_payoff

    return expected


def total_probability_target_one(prob_dict: dict[str, float]) -> float:
    """Extract the total probability that q[3] = 1.

    With get_prob_dict([0,1,2,3]), pyqpanda3 prints bitstrings as q3 q2 q1 q0,
    so q[3] is the leading bit.
    """
    return sum(probability for bitstring, probability in prob_dict.items() if bitstring[0] == "1")


if __name__ == "__main__":
    np.set_printoptions(precision=8, suppress=True)

    strike = 500.0
    scaling_constant = 0.5

    distribution = discretize_bsm_price_distribution()
    price_grid = distribution["price_grid"]
    price_probabilities = distribution["probabilities"]
    price_amplitudes = distribution["amplitudes"]

    sp_angles = calculate_ry_angles(price_amplitudes)

    payoff_list = calculate_call_payoffs(price_grid, strike=strike)
    gamma_list, max_payoff, scaled_payoffs = calculate_gamma_angles(
        payoff_list,
        scaling_constant=scaling_constant,
    )
    delta_angles = calculate_payoff_angles(gamma_list)

    theoretical_target_one_probability = float(np.sum(price_probabilities * scaled_payoffs))

    machine, q = create_machine_and_qubits(4)

    prog = QP.QProg()
    prog << build_sp_circuit(sp_angles, q[:3])
    prog << build_payoff_circuit(delta_angles, q[:3], q[3])

    prob_dict = run_probabilities(machine, prog, q)
    expected_prob_dict = build_expected_full_distribution(price_probabilities, scaled_payoffs)

    raw_labels = [format(index, "04b") for index in range(16)]
    quantum_probabilities = np.array([prob_dict[label] for label in raw_labels], dtype=float)
    expected_probabilities = np.array([expected_prob_dict[label] for label in raw_labels], dtype=float)
    full_abs_error = np.abs(quantum_probabilities - expected_probabilities)

    quantum_target_one_probability = total_probability_target_one(prob_dict)
    target_one_abs_error = abs(quantum_target_one_probability - theoretical_target_one_probability)

    print("Price grid:")
    print(price_grid)
    print("\nPrice probabilities:")
    print(price_probabilities)
    print("\nCall payoff list:")
    print(payoff_list)
    print("\nGamma angles:")
    print(gamma_list)
    print("\nDelta angles:")
    print(delta_angles)
    print("\nRaw 4-qubit probability distribution:")
    print(prob_dict)
    print("\nExpected 4-qubit probability distribution:")
    print(expected_prob_dict)
    print("\nFull-distribution absolute error:")
    print(full_abs_error)
    print("\nMax full-distribution absolute error:", np.max(full_abs_error))
    print("\nTheoretical P(q[3]=1):", theoretical_target_one_probability)
    print("Quantum P(q[3]=1):", quantum_target_one_probability)
    print("Absolute error on P(q[3]=1):", target_one_abs_error)
