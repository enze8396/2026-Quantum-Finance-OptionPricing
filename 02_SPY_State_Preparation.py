"""Manual state preparation for 3-qubit SPY price loading.

This script follows the project constraints in agent.md:
- quantum framework: pyqpanda3 only
- gate set for this part: RY and CNOT only
- no macro state-loader or packaged controlled-RY gates
"""

from __future__ import annotations

import numpy as np
import pyqpanda3 as pq

# pyqpanda3 has version differences:
# - old style: symbols such as CPUQVM/QProg/RY/CNOT are exported at top level
# - new style: these symbols live under pyqpanda3.core
QP = pq if all(hasattr(pq, name) for name in ("CPUQVM", "QProg", "RY", "CNOT")) else pq.core


def lognormal_price_pdf(
    prices: np.ndarray,
    s0: float,
    r: float,
    sigma: float,
    maturity: float,
) -> np.ndarray:
    """BSM log-normal density for the terminal asset price S_T."""
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
    """Discretize the continuous BSM price density on a uniform grid."""
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
    """Compute the 7 binary-tree RY angles for a 3-qubit real-amplitude state.

    The amplitude ordering is assumed to be:
    [a000, a001, a010, a011, a100, a101, a110, a111],
    where q[0] is the most significant qubit in the binary label.
    """
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

    theta1 = _safe_theta(
        np.linalg.norm([a0, a1]),
        np.linalg.norm([a2, a3]),
    )
    theta2 = _safe_theta(
        np.linalg.norm([a4, a5]),
        np.linalg.norm([a6, a7]),
    )

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
    """Compile a 2-control RY multiplexor into RY + CNOT only.

    angles_by_branch uses the control order:
    [angle_00, angle_01, angle_10, angle_11]
    with control_msb control_lsb as the binary branch label.
    """
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
    """Build the 3-qubit state-preparation circuit using only RY and CNOT."""
    if len(q) != 3:
        raise ValueError("This circuit builder expects exactly 3 qubits.")

    angles = np.asarray(angles, dtype=float)
    if angles.shape != (7,):
        raise ValueError("Expected 7 tree angles for the 3-qubit loader.")

    theta0, theta1, theta2, theta3, theta4, theta5, theta6 = angles

    prog = QP.QProg() if hasattr(QP, "QProg") else pq.core.QProg()

    # Root split: q[0] decides between the left half and right half of the tree.
    prog << QP.RY(q[0], theta0)

    # Second level: q[1] is rotated conditioned on q[0].
    _append_single_control_ry_multiplexor(
        prog=prog,
        control=q[0],
        target=q[1],
        angle_if_control_0=theta1,
        angle_if_control_1=theta2,
    )

    # Third level: q[2] is rotated conditioned on the branch q[0]q[1].
    _append_double_control_ry_multiplexor(
        prog=prog,
        control_msb=q[0],
        control_lsb=q[1],
        target=q[2],
        angles_by_branch=np.array([theta3, theta4, theta5, theta6]),
    )

    return prog


def create_machine_and_qubits(num_qubits: int = 3):
    """Create a CPU simulator and qubit labels with old/new pyqpanda3 compatibility."""
    machine = QP.CPUQVM()

    if hasattr(machine, "init_qvm"):
        machine.init_qvm()

    if hasattr(machine, "qAlloc_many"):
        q = machine.qAlloc_many(num_qubits)
    else:
        q = list(range(num_qubits))

    return machine, q


def run_probabilities(machine, prog, q) -> dict[str, float]:
    """Run the circuit and return a probability dictionary across 3 qubits."""
    if hasattr(machine, "prob_run_dict"):
        return machine.prob_run_dict(prog, q, -1)

    machine.run(prog, 0)
    result = machine.result()

    if hasattr(result, "get_prob_dict"):
        return result.get_prob_dict(q)

    raise AttributeError("Unable to find a supported probability-readout API in pyqpanda3.")


def _ordered_probabilities_from_dict(
    prob_dict: dict[str, float],
    num_qubits: int,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Convert prob_run_dict output into a dense probability vector.

    If reference is provided, we also test the reversed-bit ordering and keep
    the one with the smaller max error. This makes the comparison robust to
    simulator bitstring conventions.
    """
    labels = [format(index, f"0{num_qubits}b") for index in range(2**num_qubits)]
    direct = np.array([prob_dict[label] for label in labels], dtype=float)

    if reference is None:
        return direct, "direct"

    reversed_bits = np.array([prob_dict[label[::-1]] for label in labels], dtype=float)

    direct_error = np.max(np.abs(direct - reference))
    reversed_error = np.max(np.abs(reversed_bits - reference))

    if direct_error <= reversed_error:
        return direct, "direct"
    return reversed_bits, "bit-reversed"


if __name__ == "__main__":
    np.set_printoptions(precision=8, suppress=True)

    distribution = discretize_bsm_price_distribution()
    target_probabilities = distribution["probabilities"]
    target_amplitudes = distribution["amplitudes"]

    angles = calculate_ry_angles(target_amplitudes)

    print("Target amplitudes:")
    print(target_amplitudes)
    print("\nTree angles theta_0 ... theta_6:")
    print(angles)

    machine, q = create_machine_and_qubits(3)
    prog = build_sp_circuit(angles, q)

    prob_dict = run_probabilities(machine, prog, q)
    quantum_probabilities, ordering = _ordered_probabilities_from_dict(
        prob_dict=prob_dict,
        num_qubits=3,
        reference=target_probabilities,
    )

    abs_error = np.abs(quantum_probabilities - target_probabilities)

    print("\nRaw prob_run_dict output:")
    print(prob_dict)
    print("\nSelected ordering for comparison:", ordering)
    print("\nTarget probabilities:")
    print(target_probabilities)
    print("\nQuantum probabilities:")
    print(quantum_probabilities)
    print("\nAbsolute error:")
    print(abs_error)
    print("\nMax absolute error:", np.max(abs_error))
    print("L2 error:", np.linalg.norm(abs_error))
