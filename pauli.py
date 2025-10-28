import numpy as np
from math import comb
from itertools import combinations
import os
from openfermion import (
    FermionOperator,
    jordan_wigner,
    fermi_hubbard,
    get_fermion_operator,
    generate_hamiltonian,
    QubitOperator,
)
from openfermion.linalg import get_sparse_operator


class Pauli:
    I = np.array([[1, 0], [0, 1]], dtype=complex)

    X = np.array([[0, 1], [1, 0]], dtype=complex)

    Y = np.array([[0, -1j], [1j, 0]])

    Z = np.array([[1, 0], [0, -1]], dtype=complex)


def get_ps_basis(s: int | tuple[int], N: int, ordering="uudd") -> np.ndarray:
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

    spin_protected = isinstance(s, tuple)
    basis_dir = "basis_ixs_udud"
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
            if ordering == "udud":
                if N % 2 != 0:
                    raise ValueError("N must be even for spin-protected basis.")
                if (s[0] > N // 2) or (s[1] > N // 2):
                    raise ValueError(
                        "Number of spin-up or spin-down particles exceeds half the system size."
                    )

                half = N // 2
                # Generate all bit patterns for each half
                # Build patterns on even and odd site indices (0-based)
                even_positions = list(range(0, N, 2))
                odd_positions = list(range(1, N, 2))

                first_half = np.zeros(comb(len(even_positions), s[0]), dtype=np.int64)
                for i, ones_idx in enumerate(
                    combinations(range(len(even_positions)), s[0])
                ):
                    n = 0
                    for idx in ones_idx:
                        pos = even_positions[idx]
                        n |= 1 << (N - 1 - pos)
                    first_half[i] = n

                second_half = np.zeros(comb(len(odd_positions), s[1]), dtype=np.int64)
                for i, ones_idx in enumerate(
                    combinations(range(len(odd_positions)), s[1])
                ):
                    n = 0
                    for idx in ones_idx:
                        pos = odd_positions[idx]
                        n |= 1 << (N - 1 - pos)
                    second_half[i] = n

                # Combine even- and odd-site patterns
                basis = np.zeros(len(first_half) * len(second_half), dtype=np.int64)
                idx = 0
                for fh in first_half:
                    for sh in second_half:
                        basis[idx] = fh | sh
                        idx += 1

                basis = np.sort(basis)
                np.savetxt(basis_file, basis, fmt="%d")
                return basis

            if ordering == "uudd":

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
                raise NotImplementedError(
                    f"Ordering '{ordering}' not implemented for spin-protected basis."
                )

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


def gen_from_pauli_string_old2(
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


def gen_from_pauli_string(
    N: int,
    pauli_string: str,
    particle_selection: tuple[int, int] | int = None,
    ordering="uudd",
) -> np.ndarray:
    if pauli_string == "":
        mat = np.eye(2**N)
        if particle_selection is not None:
            basis = get_ps_basis(particle_selection, N, ordering=ordering)
            mat = mat[:, basis][basis]
        return mat
    else:
        op = QubitOperator(pauli_string)
        # print(op)
        mat = get_sparse_operator(op, N).toarray()

        # import matplotlib.pyplot as plt
        # plt.imshow(mat.real, cmap="bwr")
        # plt.colorbar()
        # plt.title(f"{pauli_string} full matrix")
        # plt.show()
        if particle_selection is not None:
            basis = get_ps_basis(particle_selection, N, ordering=ordering)
            mat = mat[:, basis][basis]
            # plt.imshow(mat.real, cmap="bwr")
            # plt.colorbar()
            # plt.title(f"{pauli_string} reduced matrix")
            # plt.show()
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


def model_to_paulis(
    N: int, model: str, model_parameters: dict
) -> list[tuple[str, complex]]:

    if model == "fermi_hubbard":
        t = model_parameters.get("t", 1.0)
        U = model_parameters.get("U", 0.0)
        mu = model_parameters.get("mu", 0.0)
        of_hamiltonian = fermi_hubbard(
            N,
            1,
            t,
            U,
            chemical_potential=mu,
            periodic=False,
        )
        jw_hamiltonian = jordan_wigner(of_hamiltonian)
        pauli_list = of_operator_to_pauli_and_coeff(N * 2, jw_hamiltonian)
        return pauli_list

    elif model == "heisenberg":
        Jx = model_parameters.get("Jx", 1.0)
        Jy = model_parameters.get("Jy", 1.0)
        Jz = model_parameters.get("Jz", 1.0)
        h = model_parameters.get("h", 0.0)
        periodic = model_parameters.get("periodic", False)

        operator = QubitOperator()
        for i in range(N - 1):
            operator += QubitOperator(f"X{i} X{i+1}", Jx)
            operator += QubitOperator(f"Y{i} Y{i+1}", Jy)
            operator += QubitOperator(f"Z{i} Z{i+1}", Jz)
        if periodic:
            operator += QubitOperator(f"X{N-1} X0", Jx)
            operator += QubitOperator(f"Y{N-1} Y0", Jy)
            operator += QubitOperator(f"Z{N-1} Z0", Jz)

        for i in range(N):
            operator += QubitOperator(f"Z{i}", h)

        pauli_list = of_operator_to_pauli_and_coeff(N, operator)
        return pauli_list

    elif model == "TFIM":
        operator = QubitOperator()
        J = model_parameters.get("J", -1.0)
        h = model_parameters.get("h", 1.0)
        periodic = model_parameters.get("periodic", False)
        for i in range(N):
            operator += QubitOperator(f"X{i}", h)
        for i in range(N - 1):
            operator += QubitOperator(f"Z{i} Z{i+1}", J)
        if periodic:
            operator += QubitOperator(f"Z{N-1} Z0", J)

        pauli_list = of_operator_to_pauli_and_coeff(N, operator)
        return pauli_list

    elif model == "TFXY":
        operator = QubitOperator()
        Jx = model_parameters.get("Jx", 1.0)
        Jy = model_parameters.get("Jy", 1.0)
        h = model_parameters.get("h", 1.0)
        periodic = model_parameters.get("periodic", False)
        for i in range(N):
            operator += QubitOperator(f"X{i}", h)
        for i in range(N - 1):
            operator += QubitOperator(f"X{i} X{i+1}", Jx)
            operator += QubitOperator(f"Y{i} Y{i+1}", Jy)
        if periodic:
            operator += QubitOperator(f"X{N-1} X0", Jx)
            operator += QubitOperator(f"Y{N-1} Y0", Jy)

        pauli_list = of_operator_to_pauli_and_coeff(N, operator)
        return pauli_list

    elif model == "AIM":
        one_body = np.zeros((N, N))
        two_body = np.zeros((N, N, N, N))

        NI = model_parameters.get("NI", 1)
        NB = N - NI
        ei = model_parameters.get("ei", [0.0] * NI)
        ebs = model_parameters.get("eb", np.linspace(0.1, 2, NB))
        vbs = model_parameters.get("vb", np.linspace(0.1, 2, NB))
        U = model_parameters.get("U", 1.0)
        mu = model_parameters.get("mu", U / 2)

        for i in range(NI):
            one_body[i, i] = ei[i] - mu
            two_body[i, i, i, i] = U
        for i in range(NI - 1):
            two_body[i, i + 1, i + 1, i] = U
            two_body[i + 1, i, i, i + 1] = U

        for i in range(NI):
            for j in range(NB):
                one_body[NI + j, NI + j] = ebs[j]
                one_body[i, NI + j] = vbs[j]
                one_body[NI + j, i] = vbs[j]

        jw_hamiltonian = jordan_wigner(
            get_fermion_operator(generate_hamiltonian(one_body, two_body, 0))
        )

        pauli_list = of_operator_to_pauli_and_coeff(N * 2, jw_hamiltonian)
        return pauli_list

    else:
        raise ValueError(f"Model '{model}' not recognized.")
