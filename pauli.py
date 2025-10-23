import numpy as np
from math import comb
from itertools import combinations
import os


class Pauli:
    I = np.array([[1, 0], [0, 1]], dtype=complex)

    X = np.array([[0, 1], [1, 0]], dtype=complex)

    Y = np.array([[0, -1j], [1j, 0]])

    Z = np.array([[1, 0], [0, -1]], dtype=complex)


def get_ps_basis(s: int | tuple[int], N: int) -> np.ndarray:
    """
    Get the particle-number (and possibly spin) sector basis for a system of N sites.
    If the basis file does not exist, it will be calculated and saved.
    Args:
        s (int | tuple[int]): Number of particles (or tuple of spin-up and
            spin-down particles).
        N (int): Total number of sites.
    Returns:
        np.ndarray: Array of basis states in integer representation.
    """
    spin_protected = type(s) == tuple
    basis_dir = "basis_ixs"
    os.makedirs(basis_dir, exist_ok=True)
    if spin_protected:
        basis_file = os.path.join(basis_dir, f"basis_{N}_{s[0]}_{s[1]}.txt")
    else:
        basis_file = os.path.join(basis_dir, f"basis_{N}_{s}.txt")
    try:
        basis = np.loadtxt(basis_file).astype(np.int64)
        return basis

    except FileNotFoundError:
        print("No basis file, calculating now...")
        if spin_protected:
            if N % 2 != 0:
                raise ValueError("N must be even for spin-protected basis.")
            if (s[0] > N // 2) or (s[1] > N // 2):
                raise ValueError(
                    "Number of spin-up or spin-down particles exceeds half the system size."
                )

            half = N // 2
            # Generate all bit patterns for each half
            first_half = np.zeros(comb(half, s[0]), dtype=np.int64)
            for i, ones_pos in enumerate(combinations(range(half), s[0])):
                n = 0
                for pos in ones_pos:
                    n |= 1 << (half - 1 - pos)
                first_half[i] = n

            second_half = np.zeros(comb(half, s[1]), dtype=np.int64)
            for i, ones_pos in enumerate(combinations(range(half), s[1])):
                n = 0
                for pos in ones_pos:
                    n |= 1 << (half - 1 - pos)
                second_half[i] = n

            # Combine both halves
            basis = np.zeros(len(first_half) * len(second_half), dtype=np.int64)
            idx = 0
            for fh in first_half:
                for sh in second_half:
                    combined = (fh << half) | sh
                    basis[idx] = combined
                    idx += 1

            basis = np.sort(basis)
            np.savetxt(basis_file, basis, fmt="%d")
            return basis

        else:

            basis = np.zeros(comb(N, s), dtype=np.int64)
            for i, ones_pos in enumerate(combinations(range(N), s)):  # type: ignore
                n = 0
                for pos in ones_pos:
                    n |= 1 << (N - 1 - pos)
                basis[i] = n
            basis = np.sort(basis)
            np.savetxt(basis_file, basis, fmt="%d")
            return basis


def gen_from_pauli_string_old(
    N: int, pauli_string: str, particle_selection: tuple[int] | int = None
) -> np.ndarray:
    next_location = 0
    mat = np.eye(1)
    for i in np.arange(0, len(pauli_string), 2):
        for j in range(int(pauli_string[i + 1]) - next_location):
            mat = np.kron(mat, Pauli.I)
        next_location = int(pauli_string[i + 1]) + 1
        if pauli_string[i] == "I":
            mat = np.kron(mat, Pauli.I)
        elif pauli_string[i] == "X":
            mat = np.kron(mat, Pauli.X)
        elif pauli_string[i] == "Y":
            mat = np.kron(mat, Pauli.Y)
        elif pauli_string[i] == "Z":
            mat = np.kron(mat, Pauli.Z)

    for i in np.arange(N - next_location):
        mat = np.kron(mat, Pauli.I)

    if type(particle_selection) != type(None):
        basis = get_ps_basis(particle_selection, N)
        mat = mat[:, basis][basis]

    return mat


def gen_from_pauli_string(
    N: int, pauli_string: str, particle_selection: tuple[int] | int = None
) -> np.ndarray:
    pauli_map = {
        "I": Pauli.I,
        "X": Pauli.X,
        "Y": Pauli.Y,
        "Z": Pauli.Z,
    }
    pauli_list = pauli_string.split(" ")
    if pauli_list == [""]:
        mat = np.eye(2**N)
        if particle_selection is not None:
            basis = get_ps_basis(particle_selection, N)
            mat = mat[:, basis][basis]
        return mat

    else:
        next_location = 0
        mat = np.eye(1)
        for i in range(0, len(pauli_list)):
            index = int(pauli_list[i][1:])
            # Add identity matrices for skipped qubits
            for _ in range(index - next_location):
                mat = np.kron(mat, Pauli.I)
            next_location = index + 1
            # Add the specified Pauli matrix
            mat = np.kron(mat, pauli_map[pauli_list[i][0]])

        # Add identity matrices for remaining qubits
        for _ in range(N - next_location):
            mat = np.kron(mat, Pauli.I)

        if particle_selection is not None:
            basis = get_ps_basis(particle_selection, N)
            mat = mat[:, basis][basis]

        return mat


def of_operator_to_pauli_and_coeff(N, of_operator) -> list[tuple[str, complex]]:
    pauli_list = []
    for term in of_operator.terms:
        pauli_string = ""
        for qubit_index in range(N):
            if qubit_index in [idx for idx, _ in term]:
                pauli_type = [ptype for idx, ptype in term if idx == qubit_index][0]
                pauli_string += f"{pauli_type}{qubit_index} "
        pauli_string = pauli_string.strip()  # Remove trailing space
        coeff = of_operator.terms[term]
        pauli_list.append((pauli_string, coeff))
    return pauli_list
