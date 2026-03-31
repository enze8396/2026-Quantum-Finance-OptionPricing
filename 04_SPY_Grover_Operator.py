r"""Core Grover operator for IQAE-based SPY option pricing.

This script constructs the oracle-ready Grover iterate

    Q = - A S_0 A^\dagger S_f

using only low-level single-qubit gates together with CNOT.
No state-loader macro, controlled-rotation macro, or multi-controlled gate macro
is used anywhere in the implementation.
"""

from __future__ import annotations

import itertools
import math
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
    """Compute the 7 binary-tree RY angles for the 3-qubit price state."""
    amplitudes = np.asarray(amplitudes, dtype=float)
    amplitudes = amplitudes / np.linalg.norm(amplitudes)

    a0, a1, a2, a3, a4, a5, a6, a7 = amplitudes

    theta0 = _safe_theta(np.linalg.norm([a0, a1, a2, a3]), np.linalg.norm([a4, a5, a6, a7]))
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


def _append_inverse_single_control_ry_multiplexor(
    prog,
    control,
    target,
    angle_if_control_0: float,
    angle_if_control_1: float,
) -> None:
    """Append the inverse of the 1-control RY multiplexor."""
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


def _append_inverse_double_control_ry_multiplexor(
    prog,
    control_msb,
    control_lsb,
    target,
    angles_by_branch: np.ndarray,
) -> None:
    """Append the inverse of the 2-control RY multiplexor."""
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


def build_sp_circuit(angles: np.ndarray, q):
    """Build the 3-qubit state-preparation circuit."""
    theta0, theta1, theta2, theta3, theta4, theta5, theta6 = np.asarray(angles, dtype=float)

    prog = QP.QProg()
    prog << QP.RY(q[0], theta0)
    _append_single_control_ry_multiplexor(prog, q[0], q[1], theta1, theta2)
    _append_double_control_ry_multiplexor(prog, q[0], q[1], q[2], np.array([theta3, theta4, theta5, theta6]))
    return prog


def build_inverse_sp_circuit(angles: np.ndarray, q):
    """Build the inverse of the 3-qubit state-preparation circuit."""
    theta0, theta1, theta2, theta3, theta4, theta5, theta6 = np.asarray(angles, dtype=float)

    prog = QP.QProg()
    _append_inverse_double_control_ry_multiplexor(
        prog,
        q[0],
        q[1],
        q[2],
        np.array([theta3, theta4, theta5, theta6]),
    )
    _append_inverse_single_control_ry_multiplexor(prog, q[0], q[1], theta1, theta2)
    prog << QP.RY(q[0], -theta0)
    return prog


def calculate_call_payoffs(price_grid: np.ndarray, strike: float = 500.0) -> np.ndarray:
    """European call payoff max(S-K, 0)."""
    return np.maximum(np.asarray(price_grid, dtype=float) - strike, 0.0)


def calculate_gamma_angles(
    payoff_list: np.ndarray,
    scaling_constant: float = 0.5,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Map call payoffs to target-qubit branch angles gamma_i."""
    payoff_list = np.asarray(payoff_list, dtype=float)
    max_payoff = float(np.max(payoff_list))

    if np.isclose(max_payoff, 0.0):
        scaled_payoffs = np.zeros_like(payoff_list)
        gamma_list = np.zeros_like(payoff_list)
    else:
        scaled_payoffs = scaling_constant * payoff_list / max_payoff
        scaled_payoffs = np.clip(scaled_payoffs, 0.0, 1.0)
        gamma_list = 2.0 * np.arcsin(np.sqrt(scaled_payoffs))

    return gamma_list, max_payoff, scaled_payoffs


def gray_code_sequence(num_controls: int) -> list[int]:
    """Return the standard binary-reflected Gray code sequence."""
    return [index ^ (index >> 1) for index in range(2**num_controls)]


def _gray_ordered_walsh_hadamard_matrix(num_controls: int) -> np.ndarray:
    """Gray-ordered Walsh-Hadamard transform for uniformly controlled rotations."""
    num_branches = 2**num_controls
    matrix = np.empty((num_branches, num_branches), dtype=float)
    gray_codes = gray_code_sequence(num_controls)

    for row, gray_word in enumerate(gray_codes):
        for col in range(num_branches):
            parity = (gray_word & col).bit_count() % 2
            matrix[row, col] = 1.0 if parity == 0 else -1.0

    return matrix / num_branches


def calculate_payoff_angles(gamma_list: np.ndarray) -> np.ndarray:
    """Compute the delta angles for the 3-control uniformly controlled RY."""
    gamma_list = np.asarray(gamma_list, dtype=float)
    num_controls = int(np.log2(gamma_list.size))
    transform = _gray_ordered_walsh_hadamard_matrix(num_controls)
    return transform @ gamma_list


def _control_sequence_for_uniform_rotation(q_control_list) -> list[int]:
    """Return the Gray-code control-toggle sequence for uniformly controlled RY."""
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
    """Build the 3-control payoff oracle using only RY and CNOT."""
    delta_angles = np.asarray(delta_angles, dtype=float)
    control_sequence = _control_sequence_for_uniform_rotation(q_control_list)

    prog = QP.QProg()
    for angle, control in zip(delta_angles, control_sequence):
        prog << QP.RY(q_target, float(angle))
        prog << QP.CNOT(control, q_target)

    return prog


def build_inverse_payoff_circuit(delta_angles: np.ndarray, q_control_list, q_target):
    """Build the inverse of the payoff multiplexor."""
    delta_angles = np.asarray(delta_angles, dtype=float)
    control_sequence = _control_sequence_for_uniform_rotation(q_control_list)

    prog = QP.QProg()
    for angle, control in reversed(list(zip(delta_angles, control_sequence))):
        prog << QP.CNOT(control, q_target)
        prog << QP.RY(q_target, float(-angle))

    return prog


def build_A_circuit(sp_angles: np.ndarray, delta_angles: np.ndarray, q_control_list, q_target):
    """Build the full state-loading operator A = SP + Payoff."""
    prog = QP.QProg()
    prog << build_sp_circuit(sp_angles, q_control_list)
    prog << build_payoff_circuit(delta_angles, q_control_list, q_target)
    return prog


def build_inverse_A_circuit(
    sp_angles: np.ndarray,
    delta_angles: np.ndarray,
    q_control_list,
    q_target,
):
    """Build the exact inverse A^\u2020 by reversing all low-level gates manually."""
    prog = QP.QProg()
    prog << build_inverse_payoff_circuit(delta_angles, q_control_list, q_target)
    prog << build_inverse_sp_circuit(sp_angles, q_control_list)
    return prog


def _append_phase_product(prog, qubit_subset, alpha: float) -> None:
    """Append exp(i alpha Z_{subset}) using only CNOT and RZ."""
    qubit_subset = list(qubit_subset)
    target = qubit_subset[-1]

    for control in qubit_subset[:-1]:
        prog << QP.CNOT(control, target)
    prog << QP.RZ(target, -2.0 * alpha)
    for control in reversed(qubit_subset[:-1]):
        prog << QP.CNOT(control, target)


def build_multi_controlled_z_all_ones(qubits):
    """Build a phase flip on |11...1> using phase-polynomial synthesis.

    Up to a physically irrelevant global phase, the operator is the exact
    multi-controlled Z reflection on the all-ones basis state.
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
    """Build the all-zero reflection S_0 using X wrappers and an exact C^3Z."""
    qubits = list(qubits)
    prog = QP.QProg()

    for qubit in qubits:
        prog << QP.X(qubit)

    prog << build_multi_controlled_z_all_ones(qubits)

    for qubit in qubits:
        prog << QP.X(qubit)

    return prog


def build_Sf_circuit(q_target):
    """Build S_f, the phase flip on the good subspace q_target = 1."""
    prog = QP.QProg()
    prog << QP.Z(q_target)
    return prog


def build_grover_Q(
    sp_angles: np.ndarray,
    delta_angles: np.ndarray,
    q_control_list,
    q_target,
):
    """Build the Grover iterate Q = -A S_0 A^\u2020 S_f.

    The leading global phase -1 is omitted, because it has no physical effect.
    """
    all_qubits = list(q_control_list) + [q_target]

    prog = QP.QProg()
    prog << build_Sf_circuit(q_target)
    prog << build_inverse_A_circuit(sp_angles, delta_angles, q_control_list, q_target)
    prog << build_S0_circuit(all_qubits)
    prog << build_A_circuit(sp_angles, delta_angles, q_control_list, q_target)
    return prog


def create_machine_and_qubits(num_qubits: int):
    """Create a CPUQVM with old/new pyqpanda3 compatibility."""
    machine = QP.CPUQVM()
    if hasattr(machine, "init_qvm"):
        machine.init_qvm()

    if hasattr(machine, "qAlloc_many"):
        q = machine.qAlloc_many(num_qubits)
    else:
        q = list(range(num_qubits))

    return machine, q


def run_probabilities(machine, prog, q) -> dict[str, float]:
    """Run a quantum program and return its full probability dictionary."""
    if hasattr(machine, "prob_run_dict"):
        return machine.prob_run_dict(prog, q, -1)

    machine.run(prog, 0)
    return machine.result().get_prob_dict(q)


def total_probability_target_one(prob_dict: dict[str, float]) -> float:
    """Extract the total probability that q[3] = 1.

    In pyqpanda3's get_prob_dict([0,1,2,3]), bitstrings are printed as q3 q2 q1 q0.
    Hence q[3] corresponds to the leading bit.
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

    initial_success_probability = float(np.sum(price_probabilities * scaled_payoffs))
    theta = math.asin(math.sqrt(initial_success_probability))
    one_grover_theory = math.sin(3.0 * theta) ** 2

    machine, q = create_machine_and_qubits(4)

    prog_A = QP.QProg()
    prog_A << build_A_circuit(sp_angles, delta_angles, q[:3], q[3])
    prob_dict_A = run_probabilities(machine, prog_A, q)
    success_probability_A = total_probability_target_one(prob_dict_A)

    machine_after_Q, q_after_Q = create_machine_and_qubits(4)
    prog_AQ = QP.QProg()
    prog_AQ << build_A_circuit(sp_angles, delta_angles, q_after_Q[:3], q_after_Q[3])
    prog_AQ << build_grover_Q(sp_angles, delta_angles, q_after_Q[:3], q_after_Q[3])
    prob_dict_AQ = run_probabilities(machine_after_Q, prog_AQ, q_after_Q)
    success_probability_AQ = total_probability_target_one(prob_dict_AQ)

    print("Price grid:")
    print(price_grid)
    print("\nPayoff list:")
    print(payoff_list)
    print("\nGamma angles:")
    print(gamma_list)
    print("\nDelta angles:")
    print(delta_angles)
    print("\nInitial success probability P(q[3]=1) after A:")
    print(success_probability_A)
    print("Classical theory for A:")
    print(initial_success_probability)
    print("\nSuccess probability P(q[3]=1) after A followed by one Q:")
    print(success_probability_AQ)
    print("Amplitude-amplification theory sin^2(3 theta):")
    print(one_grover_theory)
    print("Absolute error after one Q:")
    print(abs(success_probability_AQ - one_grover_theory))
    print("\nRaw probability dictionary after A then Q:")
    print(prob_dict_AQ)
