# Bring up to Norman
# Particle Selection?
#  - get_ps_basis
#  - get_occupied_indices
#  - H[:,basis][basis[
# any good hamiltonians to test with?
##### allow "fixing" some parameters?
# multithreading (possibly with Python 3.14?)
# JIT (possibly with Python 3.14?)

import numpy as np
import scipy as sp
from pauli import *
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class SurrogateModel:
    """
    A class to do surrogate optimizations on a Hamiltonian and a training grid
    of parameters

    Attributes:
        N : `int`
            The number of particles in the system
        pauli_strings: `list[str]`
            A list of Pauli strings that comprise the Hamiltonian
    """

    N: int
    pauli_strings: list[str]
    H_terms: list[np.ndarray]
    H2_terms: list[np.ndarray]
    training_grid: list[list[complex]]
    training_grid2: list[list[complex]]
    opt_basis: np.ndarray
    overlap: np.ndarray
    reduced_terms: list[np.ndarray]
    particle_selection: tuple[int, int] | int = None

    def __init__(
        self,
        N: int,
        pauli_strings: list[str],
        training_grid: list[list[complex]],
        particle_selection: tuple[int, int] | int = None,
    ):
        self.N = N
        self.pauli_strings = pauli_strings
        self.H_terms = None
        self.H2_terms = None
        self.training_grid = training_grid
        self.training_grid2 = None
        self.opt_basis = None
        self.overlap = None
        self.reduced_terms = None
        self.particle_selection = particle_selection

    def build_terms(self):
        self.H_terms = []
        for pauli_string in self.pauli_strings:
            self.H_terms.append(
                gen_from_pauli_string(self.N, pauli_string, self.particle_selection),
            )

        self.H2_terms = []
        for h_i in self.H_terms:
            for h_j in self.H_terms:
                self.H2_terms.append(h_i @ h_j)

        self.training_grid2 = []
        for mu in self.training_grid:
            bulk = []
            for mu_i in mu:
                for mu_j in mu:
                    bulk.append(mu_i * mu_j)
            self.training_grid2.append(bulk)

    def _build_H_full(self, parameter_idx: int) -> np.ndarray:
        H_full = np.zeros_like(self.H_terms[0], dtype=complex)
        for p, h in zip(self.training_grid[parameter_idx], self.H_terms):
            H_full += p * h

        return H_full

    def _build_H2_full(self, parameter_idx: int) -> np.ndarray:
        H2_full = np.zeros_like(self.H2_terms[0], dtype=complex)
        for mu2_i, h2_i in zip(self.training_grid2[parameter_idx], self.H2_terms):
            H2_full += mu2_i * h2_i

        return H2_full

    def optimize(
        self,
        residue_threshold: float = 1e-3,
        init_vec: np.ndarray = None,
        solution_grid: np.ndarray = None,
    ):
        # build terms if they are not already built
        if (
            type(self.H_terms) == type(None)
            or type(self.H2_terms) == type(None)
            or type(self.training_grid2) == type(None)
        ):
            self.build_terms()

        # list of indices into the training grid
        chosen = []

        # list of remaining indices into the training grid
        not_chosen = list(range(len(self.training_grid)))

        # list of ill-conditioned choices
        dont_choose = []

        # initial vector is not provided, so we choose from the training grid
        if init_vec == None:
            H_full = self._build_H_full(0)
            evals, evecs = sp.linalg.eigh(H_full)
            init_vec = evecs[:, 0]
            chosen.append(0)
            not_chosen.remove(0)

        basis_list = [init_vec]
        basis = np.array(basis_list).T

        if type(solution_grid) != type(None):
            answer_grid = np.zeros(solution_grid.shape, dtype=complex)
            for y in range(solution_grid.shape[0]):
                for x in range(solution_grid.shape[1]):
                    H_full = self._build_H_full(y * 10 + x)
                    Hr = basis.conj().T @ H_full @ basis
                    overlap = basis.conj().T @ basis
                    evals, evecs = sp.linalg.eigh(Hr, overlap)
                    answer_grid[y, x] = evals[0]

            plt.imshow(
                np.abs((answer_grid - solution_grid).real) + 1e-20,
                norm=mcolors.LogNorm(vmin=1e-20, vmax=1),
            )
            plt.colorbar(
                norm=mcolors.LogNorm(vmin=1e-20, vmax=1e0),
                label="Absolute Error",  # , ticks=[1e-20, 1e-10, 1e0]
            )
            plt.scatter(
                chosen[0] % solution_grid.shape[1],  # type: ignore
                chosen[0] // solution_grid.shape[1],  # type: ignore
                marker="x",
                color="red",
                s=100,
                label="First choice",
            )
            plt.xlabel("p2")
            plt.ylabel("p1")
            plt.title("It 1")
            plt.show()

        # iteration
        num_iterations = len(not_chosen)
        for i in range(num_iterations):
            overlap = basis.conj().T @ basis
            max_res2 = -np.inf
            next_choice = None
            chosen_H_full = None
            residues = []
            for j in not_chosen:
                # construct Hr and H2r
                # technically H_full and H2_full only need to be constructed
                # once, however, due to possible memory limitations based on
                # the trianing grid size and full Hilbert space size, these are
                # constructed on demand
                H_full = self._build_H_full(j)
                H2_full = self._build_H2_full(j)

                Hr = basis.conj().T @ H_full @ basis
                H2r = basis.conj().T @ H2_full @ basis
                evals, evecs = sp.linalg.eigh(Hr, overlap)

                # find degeneracy of the ground state
                eps = 1e-10  # for comparing floating points of GSE
                degeneracy = 0
                for e in evals:
                    if abs(e - evals[0]) < eps:
                        degeneracy += 1
                    else:
                        break

                # calculate residue
                res2 = 0
                for k in range(degeneracy):
                    res2 += (
                        evecs[:, k].conj().T
                        @ (H2r - evals[k] * evals[k] * overlap)
                        @ evecs[:, k]
                    )
                residues.append(res2)
                if res2 > max_res2:
                    max_res2 = res2
                    next_choice = j
                    chosen_H_full = H_full

            plt.plot(not_chosen, residues, "o-")
            plt.xlabel("Training Grid Index")
            plt.ylabel("Residue")
            plt.title(f"It {i + 1} Residues")
            plt.show()
            print("Max Residue", max_res2)
            evals, evecs = np.linalg.eigh(chosen_H_full)
            # find degeneracy of the ground state
            eps = 1e-10  # for comparing floating points of GSE
            degeneracy = 0
            for e in evals:
                if abs(e - evals[0]) < eps:
                    degeneracy += 1
                else:
                    break

            basis_addition = evecs[:, 0:degeneracy]

            # compress the basis
            projection = basis_addition - basis @ sp.linalg.solve(
                overlap, basis.conj().T @ basis_addition
            )

            U, sigmas, Vdagger = np.linalg.svd(projection)
            compress_add = 0
            for s in sigmas:
                if s > eps:
                    compress_add += 1
                else:
                    break

            for j in range(compress_add):
                basis_list += [U[:, j]]
            basis = np.array(basis_list).T

            not_chosen.remove(next_choice)
            chosen.append(next_choice)

            if max_res2 < residue_threshold or len(chosen) >= 2**self.N - 1:
                chosen = chosen[:-1]
                break

            if type(solution_grid) != type(None):
                answer_grid = np.zeros(solution_grid.shape, dtype=complex)  # type: ignore
                for y in range(solution_grid.shape[0]):  # type: ignore
                    for x in range(solution_grid.shape[1]):  # type: ignore
                        H_full = self._build_H_full(y * 10 + x)
                        Hr = basis.conj().T @ H_full @ basis
                        overlap = basis.conj().T @ basis
                        evals, evecs = sp.linalg.eigh(Hr, overlap)
                        answer_grid[y, x] = evals[0]

                plt.imshow(
                    np.abs((answer_grid - solution_grid).real) + 1e-20,
                    norm=mcolors.LogNorm(vmin=1e-20, vmax=1e0),
                )
                plt.colorbar(
                    norm=mcolors.LogNorm(vmin=1e-20, vmax=1),
                    label="Absolute Error",  # , ticks=[1e-20, 1e-10, 1e0]
                )
                plt.scatter(
                    next_choice % solution_grid.shape[1],  # type: ignore
                    next_choice // solution_grid.shape[1],  # type: ignore
                    marker="x",
                    color="red",
                    s=100,
                    label="Next Choice",
                )
                plt.xlabel("p2")
                plt.ylabel("p1")
                plt.title(f"It {i + 2}")
                plt.show()

        self.opt_basis = basis
        self.overlap = basis.conj().T @ basis
        self.reduced_terms = None

        return chosen, basis

    def solve(self, parameters: list[complex]) -> complex:
        if type(self.opt_basis) == type(None) or type(self.overlap) == type(None):
            self.optimize()

        if self.reduced_terms == None:
            self.reduced_terms = []
            for h in self.H_terms:
                self.reduced_terms.append(
                    self.opt_basis.conj().T @ h @ self.opt_basis  # type: ignore
                )

        Hr = np.zeros(
            (self.opt_basis.shape[1], self.opt_basis.shape[1]),  # type: ignore
            dtype=complex,
        )

        for p, h in zip(parameters, self.reduced_terms):
            Hr += p * h

        evals, evecs = sp.linalg.eigh(Hr, self.overlap)

        return evals[0]


if __name__ == "__main__":
    mu = np.linspace(-3, 3, 10)
    N = 6
    H_paulis = (
        [f"X{i}X{(i+1)%N}" for i in range(N - 1)]
        + [f"X0X{N-1}"]
        + [f"Z{i}" for i in range(N)]
    )

    twobody_terms = [len(term) > 2 for term in H_paulis]
    one_body_terms = [len(term) <= 2 for term in H_paulis]
    n2b_terms = len(np.where(twobody_terms)[0])
    n1b_terms = len(np.where(one_body_terms)[0])

    training_grid = np.array(
        [[i] * n2b_terms + [j] * n1b_terms for i in mu for j in mu]
    )
    model = SurrogateModel(N, H_paulis, training_grid, particle_selection=None)

    model.build_terms()

    # calc real solutions
    solution_grid = np.zeros((10, 10), dtype=complex)
    for i in range(10):
        for j in range(10):
            H_full = np.zeros_like(model.H_terms[0], dtype=complex)
            parameters = training_grid[i * 10 + j]
            for k, h in enumerate(model.H_terms):
                H_full += parameters[k] * h
            evals, evecs = np.linalg.eigh(H_full)
            solution_grid[i, j] = evals[0]

    chosen, basis = model.optimize(solution_grid=solution_grid)
    print("Chosen Indices", chosen)
    print("Basis Size", basis.shape[1])
    # print(model.optimize(solution_grid=None))

    for i in range(30):
        H_full = np.zeros_like(model.H_terms[0], dtype=complex)

        (J, Bz) = 3 * np.random.randn(2)
        parameters = [J] * n2b_terms + [Bz] * n1b_terms
        for i, h in enumerate(model.H_terms):
            H_full += parameters[i] * h

        evals, evecs = np.linalg.eigh(H_full)

        # print(parameters)
        print("Real", evals[0])
        print("Approx", model.solve(parameters))
        print(
            "Relative Error",
            np.abs(evals[0] - model.solve(parameters)) / np.abs(evals[0]),
        )
        print()
