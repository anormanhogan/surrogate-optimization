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
import openfermion as of


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
        basis_ordering: str = "uudd",
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
        self.basis_ordering = basis_ordering

    def build_terms(self):
        self.H_terms = []
        for pauli_string in self.pauli_strings:
            self.H_terms.append(
                gen_from_pauli_string(
                    self.N,
                    pauli_string,
                    self.particle_selection,
                    ordering=self.basis_ordering,
                ),
            )
            print("Built term:", pauli_string, self.H_terms[-1].shape)

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
        residue_threshold: float = 1e-6,
        init_vec: np.ndarray = None,
        solution_grid: np.ndarray = None,
        svd_tolerance: float = 1e-8,
        degeneracy_truncation: int = 5,
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

        # initial vector is not provided, so we choose from the training grid
        if init_vec == None:
            H_full = self._build_H_full(0)
            evals, evecs = sp.linalg.eigh(H_full)
            init_vec = evecs[:, 0]
            chosen.append(0)
            not_chosen.remove(0)

        basis_list = [init_vec]
        basis = np.array(basis_list).T

        if type(solution_grid[0]) != type(None):
            answer_grid = np.zeros(solution_grid[0].shape, dtype=complex)
            for y in range(solution_grid[0].shape[0]):
                for x in range(solution_grid[0].shape[1]):
                    H_full = self._build_H_full(y * solution_grid[0].shape[1] + x)
                    Hr = basis.conj().T @ H_full @ basis
                    overlap = basis.conj().T @ basis
                    evals, evecs = sp.linalg.eigh(Hr, overlap)
                    answer_grid[y, x] = evals[0]

            plt.imshow(
                np.abs((answer_grid - solution_grid[0]).real) + 1e-14,
                norm=mcolors.LogNorm(vmin=1e-14, vmax=1),
            )
            plt.colorbar(
                norm=mcolors.LogNorm(vmin=1e-14, vmax=1),
                label="Absolute Error",  # , ticks=[1e-14, 1e-len(mu), 1e0]
            )
            plt.scatter(
                chosen[0] % solution_grid[0].shape[1],  # type: ignore
                chosen[0] // solution_grid[0].shape[1],  # type: ignore
                marker="x",
                color="red",
                s=20,
                label="First choice",
            )
            plt.xlabel(r"$\mu_2$")
            plt.ylabel(r"$\mu_1$")
            plt.title("It 1, Basis Size 1")
            plt.xticks(
                range(0, solution_grid[0].shape[1], 2),
                labels=np.round(solution_grid[1], 2)[::2],
            )
            plt.yticks(
                range(0, solution_grid[0].shape[0], 2),
                labels=np.round(solution_grid[1], 2)[::2],
                rotation=45,
            )
            plt.show()

        # iteration
        num_iterations = len(not_chosen)
        for i in range(num_iterations):
            print("Iteration", i + 1)
            overlap = (basis.conj().T @ basis).real
            # plt.imshow(overlap.real, label="Current Basis Vectors")
            # plt.colorbar()
            # plt.title(f"It {i + 1} Overlap")
            # plt.show()
            max_res2 = -np.inf
            next_choice = None
            chosen_H_full = None
            residues = []
            print("First loop over not chosen indices")
            for j in not_chosen:
                # print("Parameters:", self.training_grid[j])
                # construct Hr and H2r
                # technically H_full and H2_full only need to be constructed
                # once, however, due to possible memory limitations based on
                # the training grid size and full Hilbert space size, these are
                # constructed on demand

                H_full = self._build_H_full(j)
                H2_full = H_full @ H_full  # self._build_H2_full(j)
                full_evals, full_evecs = np.linalg.eigh(H_full)
                print()
                print("Full gs energy:", full_evals[0])
                print("Full gs energy squared:", full_evals[0] ** 2)
                print(
                    "Full gs on H2_full:",
                    (full_evecs[:, 0].conj().T @ H2_full @ full_evecs[:, 0]).real,
                )
                print("Full eigenvalues (first 5):", full_evals[:5])

                Hr = basis.conj().T @ H_full @ basis
                H2r = basis.conj().T @ H2_full @ basis
                evals, evecs = sp.linalg.eigh(Hr, overlap)
                print("Reduced gs energy:", evals[0])
                print("Reduced gs energy squared:", evals[0] ** 2)
                print(
                    "Reduced gs on H2r:",
                    (evecs[:, 0].conj().T @ H2r @ evecs[:, 0]).real,
                )

                # plt.imshow(evecs.real, label="Reduced Basis", aspect="auto")
                # plt.colorbar()
                # plt.title(f"It {i + 1}, Parameter Index {j} Eigenvectors")
                # plt.show()

                # plt.imshow(
                #     full_evecs[:, :5].real,
                #     label="Full Basis (first 5 vecs)",
                #     aspect="auto",
                # )
                # plt.colorbar()
                # plt.title(f"It {i + 1}, Parameter Index {j} Full Eigenvectors")
                # plt.show()

                # find degeneracy of the ground state
                degeneracy = 0
                eps = 1e-10
                for e in evals:
                    if abs(e - evals[0]) < eps:
                        degeneracy += 1
                    else:
                        break
                    if degeneracy >= degeneracy_truncation:
                        break

                # calculate residue
                res2 = 0
                for k in range(degeneracy):
                    res2 += (
                        evecs[:, k].conj().T
                        @ (H2r - ((evals[k] * evals[k]) * overlap))
                        @ evecs[:, k]
                    )
                print("Total Residue for parameter index", j, ":", res2.real)
                residues.append(res2.real)
                if res2 > max_res2:
                    max_res2 = res2
                    next_choice = j
                    chosen_H_full = H_full

            chosen_H_full = self._build_H_full(next_choice)
            max_res2 = np.max(residues)
            print("First loop complete.")
            print("Max Residue", max_res2)
            print("Number of residues calculated:", len(residues))

            if max_res2 < residue_threshold or len(chosen) >= 2**self.N - 1:
                print("Optimization complete.")
                plt.plot(not_chosen, np.array(residues).real, "o-")
                plt.plot(
                    [next_choice],
                    [max_res2.real],
                    "rx",
                    label="Next Choice",
                )
                plt.xlabel("Training Grid Index")
                plt.ylabel("Residue")
                plt.title(f"Termination Residues")
                plt.show()

                if type(solution_grid[0]) != type(None):
                    answer_grid = np.zeros(solution_grid[0].shape, dtype=complex)  # type: ignore
                    count = 0
                    for y in range(0, solution_grid[0].shape[0]):  # type: ignore
                        for x in range(0, solution_grid[0].shape[1]):  # type: ignore
                            H_full = self._build_H_full(
                                y * solution_grid[0].shape[1] + x
                            )
                            Hr = basis.conj().T @ H_full @ basis
                            overlap = basis.conj().T @ basis
                            evals, evecs = sp.linalg.eigh(Hr, overlap)
                            answer_grid[y, x] = evals[0]

                            if y * solution_grid[0].shape[1] + x in not_chosen:
                                # print(
                                #     "Estimation:",
                                #     answer_grid[y, x],
                                #     "Actual:",
                                #     solution_grid[0][y, x],
                                #     "Residue:",
                                #     residues[count],
                                # )
                                count += 1

                    plt.imshow(
                        np.abs((answer_grid - solution_grid[0]).real) + 1e-14,
                        norm=mcolors.LogNorm(vmin=1e-14, vmax=1),
                    )
                    plt.colorbar(
                        norm=mcolors.LogNorm(
                            vmin=1e-14, vmax=1
                        )  # , ticks=[1e-14, 1e-10, 1e0]
                    )
                    plt.scatter(
                        np.array(chosen) % solution_grid[0].shape[1],  # type: ignore
                        np.array(chosen) // solution_grid[0].shape[1],  # type: ignore
                        marker="o",
                        color="orange",
                        s=20,
                        label="Chosen Points",
                    )
                    plt.xlabel(r"$\mu_2$")
                    plt.ylabel(r"$\mu_1$")
                    plt.title(f"Termination errors, Basis Size {basis.shape[1]}")
                    plt.xticks(
                        range(0, solution_grid[0].shape[1], 2),
                        labels=np.round(solution_grid[1], 2)[::2],
                    )
                    plt.yticks(
                        range(0, solution_grid[0].shape[0], 2),
                        labels=np.round(solution_grid[1], 2)[::2],
                        rotation=45,
                    )
                    plt.show()

                break

            plt.plot(not_chosen, np.array(residues).real, "o-")
            plt.plot([next_choice], [max_res2.real], "rx", label="Next Choice")
            plt.xlabel("Training Grid Index")
            plt.ylabel("Residue")
            plt.title(f"It {i + 1} Residues")
            plt.show()

            # plt.plot(basis.real, label="Current Basis Vectors")
            # plt.title(f"It {i + 1} Basis")
            # plt.legend()
            # plt.show()

            evals, evecs = np.linalg.eigh(chosen_H_full)
            print("Full system size:", evals.shape[0])
            # print("Full Eigenvalues (degen of Hr + 10):", evals[: degeneracy + 10])
            # find degeneracy of the ground state
            # for comparing floating points of GSE
            degeneracy = 0
            for e in evals:
                if abs(e - evals[0]) < eps:
                    degeneracy += 1
                else:
                    break
                if degeneracy >= degeneracy_truncation:
                    break

            print("Degeneracy of chosen H_full ground state:", degeneracy)
            basis_addition = evecs[:, 0:degeneracy]
            # plt.plot(basis_addition, label="Basis Addition Vectors")
            # plt.plot(basis, label="Current Basis Vectors")
            # plt.title(f"It {i + 1} Basis Addition")
            # plt.legend()
            # plt.show()

            # compress the basis
            projection = basis_addition - basis @ sp.linalg.solve(
                overlap, basis.conj().T @ basis_addition
            )

            # plt.plot(projection.real, label="Projection Vectors")
            # plt.title(f"It {i + 1} Projection")
            # plt.legend()
            # plt.show()

            U, sigmas, Vdagger = np.linalg.svd(projection)
            compress_add = 0
            for s in sigmas:
                print("Sigma:", s)
                if s > svd_tolerance:
                    compress_add += 1
                else:
                    break

            for j in range(compress_add):
                basis_list += [U[:, j]]

            basis_reduced = np.array(basis_list).T
            print("Basis size before compression:", basis.shape[1])
            print("Basis size after compression:", basis_reduced.shape[1])

            if basis_reduced.shape[1] <= basis.shape[1]:
                print("Warning: Basis did not increase in size after compression.")
                basis = copy.copy(basis_reduced)
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
                    svd_tolerance = 1e-10  # for comparing floating points of GSE
                    degeneracy = 1

                    # calculate residue
                    res2 = 0
                    for k in range(degeneracy):
                        res2 += (
                            evecs[:, k].conj().T
                            @ (H2r - evals[k] * evals[k] * overlap)
                            @ evecs[:, k]
                        )
                    residues.append(res2.real)
                    if res2 > max_res2:
                        max_res2 = res2
                        next_choice = j
                        chosen_H_full = H_full

                chosen_H_full = self._build_H_full(next_choice)
                max_res2 = np.max(residues)
                print("Max Residue after Compression", max_res2)

                if max_res2 < residue_threshold or len(chosen) >= 2**self.N - 1:
                    print("Optimization complete.")
                    plt.plot(not_chosen, np.array(residues).real, "o-")
                    plt.plot(
                        [next_choice],
                        [max_res2.real],
                        "rx",
                        label="Next Choice",
                    )
                    plt.xlabel("Training Grid Index")
                    plt.ylabel("Residue")
                    plt.title(f"Termination Residues")
                    plt.show()

                    if type(solution_grid[0]) != type(None):
                        answer_grid = np.zeros(solution_grid[0].shape, dtype=complex)  # type: ignore
                        count = 0
                        for y in range(0, solution_grid[0].shape[0]):  # type: ignore
                            for x in range(0, solution_grid[0].shape[1]):  # type: ignore
                                H_full = self._build_H_full(
                                    y * solution_grid[0].shape[1] + x
                                )
                                Hr = basis.conj().T @ H_full @ basis
                                overlap = basis.conj().T @ basis
                                evals, evecs = sp.linalg.eigh(Hr, overlap)
                                answer_grid[y, x] = evals[0]
                                if y * solution_grid[0].shape[1] + x in not_chosen:
                                    # print(
                                    #     "Estimation:",
                                    #     answer_grid[y, x],
                                    #     "Actual:",
                                    #     solution_grid[0][y, x],
                                    #     "Residue:",
                                    #     residues[count],
                                    # )
                                    count += 1

                        plt.imshow(
                            np.abs((answer_grid - solution_grid[0]).real) + 1e-14,
                            norm=mcolors.LogNorm(vmin=1e-14, vmax=1),
                        )
                        plt.colorbar(
                            norm=mcolors.LogNorm(
                                vmin=1e-14, vmax=1
                            )  # , ticks=[1e-14, 1e-10, 1e0]
                        )
                        plt.scatter(
                            np.array(chosen) % solution_grid[0].shape[1],  # type: ignore
                            np.array(chosen) // solution_grid[0].shape[1],  # type: ignore
                            marker="o",
                            color="orange",
                            s=20,
                            label="Chosen Points",
                        )
                        plt.xlabel(r"$\mu_2$")
                        plt.ylabel(r"$\mu_1$")
                        plt.title(f"Termination errors, Basis Size {basis.shape[1]}")
                        plt.xticks(
                            range(0, solution_grid[0].shape[1], 2),
                            labels=np.round(solution_grid[1], 2)[::2],
                        )
                        plt.yticks(
                            range(0, solution_grid[0].shape[0], 2),
                            labels=np.round(solution_grid[1], 2)[::2],
                            rotation=45,
                        )
                        plt.show()
                    break

                plt.plot(not_chosen, np.array(residues).real, "o-")
                plt.plot(
                    [next_choice],
                    [max_res2.real],
                    "rx",
                    label="Next Choice",
                )
                plt.xlabel("Training Grid Index")
                plt.ylabel("Residue")
                plt.title(f"It {i + 1} Residues after Compression")
                plt.show()
            else:
                basis = copy.copy(basis_reduced)

            print("Looking at solutions with current basis of size", basis.shape[1])
            if type(solution_grid[0]) != type(None):
                answer_grid = np.zeros(solution_grid[0].shape, dtype=complex)  # type: ignore
                count = 0
                for y in range(solution_grid[0].shape[0]):  # type: ignore
                    for x in range(solution_grid[0].shape[1]):  # type: ignore
                        H_full = self._build_H_full(y * solution_grid[0].shape[1] + x)
                        Hr = basis.conj().T @ H_full @ basis
                        overlap = basis.conj().T @ basis
                        evals, evecs = sp.linalg.eigh(Hr, overlap)
                        answer_grid[y, x] = evals[0]
                        if y * solution_grid[0].shape[1] + x in not_chosen:
                            # print(
                            #     "Estimation:",
                            #     answer_grid[y, x],
                            #     "Actual:",
                            #     solution_grid[0][y, x],
                            #     "Residue:",
                            #     residues[count],
                            # )
                            count += 1

                plt.imshow(
                    np.abs((answer_grid - solution_grid[0]).real) + 1e-14,
                    norm=mcolors.LogNorm(vmin=1e-14, vmax=1),
                )
                plt.colorbar(
                    norm=mcolors.LogNorm(
                        vmin=1e-14, vmax=1
                    )  # , ticks=[1e-14, 1e-10, 1e0]
                )
                plt.scatter(
                    next_choice % solution_grid[0].shape[1],  # type: ignore
                    next_choice // solution_grid[0].shape[1],  # type: ignore
                    marker="x",
                    color="red",
                    s=20,
                    label="Next Choice",
                )
                plt.scatter(
                    np.array(chosen) % solution_grid[0].shape[1],  # type: ignore
                    np.array(chosen) // solution_grid[0].shape[1],  # type: ignore
                    marker="o",
                    color="orange",
                    s=20,
                    label="Chosen Points",
                )
                plt.xlabel(r"$\mu_2$")
                plt.ylabel(r"$\mu_1$")
                plt.title(f"It {i + 2}, Basis Size {basis.shape[1]}")
                plt.xticks(
                    range(0, solution_grid[0].shape[1], 2),
                    labels=np.round(solution_grid[1], 2)[::2],
                )
                plt.yticks(
                    range(0, solution_grid[0].shape[0], 2),
                    labels=np.round(solution_grid[1], 2)[::2],
                    rotation=45,
                )
                plt.show()

            not_chosen.remove(next_choice)
            chosen.append(next_choice)

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

    ###############################################################
    res_thresh = 1e-6
    svd_tol = 1e-5
    mu = np.linspace(-3.0, 3.0, 10)
    N = 4
    ps = None  # Between 0 and (2*N for AIM, fermi_hubbard), N for TFIM/TFXY/Heisenberg
    ### NOTE: Particle selection *required* for heisenberg

    model_type = "AIM"  # AIM = Single Impurity Anderson Model, fermi_hubbard, TFIM, TFXY, heisenberg

    if model_type == "TFIM":
        model_parameters = {
            "J": 1,
            "h": 1,
            "periodic": False,
        }
    elif model_type == "TFXY":
        model_parameters = {
            "Jx": 1,
            "Jy": 1,
            "h": 1,
            "periodic": False,
        }
    elif model_type == "heisenberg":
        model_parameters = {
            "Jx": 1,
            "Jy": 1,
            "Jz": 1,
            "h": 1,
            "periodic": False,
        }
    elif model_type == "fermi_hubbard":
        U = 3.0
        model_parameters = {
            "t": -1,
            "mu": U / 2,
            "U": U,
            "periodic": False,
        }
    elif model_type == "AIM":
        U = 3.0
        NI = 1
        NB = N - NI
        model_parameters = {
            "NI": NI,
            "NB": NB,
            "U": U,
            "ei": [0.0] * NI,
            "vb": np.array([0.01] * ((NB) % 2) + [1.0] * (NB - (NB) % 2)),
            "eb": np.array(
                [0.0] * ((NB) % 2)
                + [1.0] * ((NB - (NB) % 2) // 2)
                + [-1.0] * ((NB - (NB) % 2) // 2)
            ),
            "mu": U / 2,
            "periodic": False,
        }

    model_paulis = model_to_paulis(N, model_type, model_parameters)

    H_paulis = [t[0] for t in model_paulis]

    training_grid = []
    for m1 in mu:
        for m2 in mu:
            if model_type == "TFIM":
                model_parameters["J"] = m1
                model_parameters["h"] = m2
            elif model_type == "TFXY":
                model_parameters["Jx"] = m1
                model_parameters["Jy"] = m1
                model_parameters["h"] = m2
            elif model_type == "heisenberg":
                model_parameters["Jx"] = m1
                model_parameters["Jy"] = m1
                model_parameters["Jz"] = m2
                model_parameters["h"] = 0.1
            elif model_type == "fermi_hubbard":
                model_parameters["t"] = m1
                model_parameters["mu"] = m2
                model_parameters["U"] = U
            elif model_type == "AIM":
                model_parameters["vb"] = np.array(
                    [0.01] * ((NB) % 2) + [m1] * (NB - (NB) % 2)
                )

                model_parameters["eb"] = np.array(
                    [0.0] * ((NB) % 2)
                    + [m2] * ((NB - (NB) % 2) // 2)
                    + [-m2] * ((NB - (NB) % 2) // 2)
                )
            model_paulis = model_to_paulis(N, model_type, model_parameters)

            if model_type == "AIM" or model_type == "fermi_hubbard":
                id_loc = np.where(np.array([t[0] for t in model_paulis]) == "")[0][0]
                grid_point1 = [model_paulis[id_loc][1]]
                grid_point = grid_point1 + [
                    model_paulis[k][1] for k in range(len(model_paulis)) if k != id_loc
                ]
                training_grid.append(grid_point)
            else:
                training_grid.append([t[1] for t in model_paulis])

    if model_type == "AIM" or model_type == "fermi_hubbard":
        surrogate_N = 2 * N
        surrogate_ord = "udud"
    else:
        surrogate_N = N
        surrogate_ord = "uudd"
    model = SurrogateModel(
        surrogate_N,
        H_paulis,
        training_grid,
        particle_selection=ps,
        basis_ordering=surrogate_ord,
    )

    model.build_terms()
    print("Number of H terms:", len(model.H_terms))

    # calc real solutions
    solution_grid = np.zeros((len(mu), len(mu)), dtype=complex)
    for i in range(len(mu)):
        for j in range(len(mu)):
            H_full = np.zeros_like(model.H_terms[0], dtype=complex)
            parameters = training_grid[i * len(mu) + j]
            for k, h in enumerate(model.H_terms):
                H_full += parameters[k] * h
            evals, evecs = np.linalg.eigh(H_full)
            solution_grid[i, j] = evals[0]

    chosen, basis = model.optimize(
        solution_grid=(solution_grid, mu),
        svd_tolerance=svd_tol,
        residue_threshold=res_thresh,
    )
    print("Chosen Indices", chosen)
    print("Basis Size", basis.shape[1])

    errors = []
    all_ps = []
    for i in range(100):
        H_full = np.zeros_like(model.H_terms[0], dtype=complex)

        if model_type == "TFIM":
            J = 2 * np.random.randn()
            h = 2 * np.random.randn()
            model_paulis = model_to_paulis(
                N,
                model_type,
                {
                    "J": J,
                    "h": h,
                    "periodic": False,
                },
            )
        elif model_type == "TFXY":
            Jx = 2 * np.random.randn()
            Jy = Jx
            h = 2 * np.random.randn()
            model_paulis = model_to_paulis(
                N,
                model_type,
                {
                    "Jx": Jx,
                    "Jy": Jy,
                    "h": h,
                    "periodic": False,
                },
            )
        elif model_type == "heisenberg":
            Jx = 2 * np.random.randn()
            Jy = Jx
            Jz = 2 * np.random.randn()
            h = 2 * np.random.randn()
            model_paulis = model_to_paulis(
                N,
                model_type,
                {
                    "Jx": Jx,
                    "Jy": Jy,
                    "Jz": Jz,
                    "h": h,
                    "periodic": False,
                },
            )
        elif model_type == "fermi_hubbard":
            model_paulis = model_to_paulis(
                N,
                model_type,
                {
                    "t": 2 * np.random.randn(),
                    "mu": 2 * np.random.randn(),
                    "U": U,
                },
            )
        elif model_type == "AIM":
            vb_test = np.array(
                [0.01] * ((NB) % 2) + [2 * np.random.randn()] * (NB - (NB) % 2)
            )
            eb_r = 2.0 * np.random.randn()
            eb_test = np.array(
                [0.0] * ((NB) % 2)
                + [eb_r] * ((NB - (NB) % 2) // 2)
                + [-eb_r] * ((NB - (NB) % 2) // 2)
            )
            print("vb_test", vb_test)
            print("eb_test", eb_test)
            model_paulis = model_to_paulis(
                N,
                model_type,
                {
                    "NI": NI,
                    "NB": NB,
                    "U": U,
                    "ei": [0.0] * NI,
                    "vb": vb_test,
                    "eb": eb_test,
                    "mu": U / 2,
                    "periodic": False,
                },
            )

        if model_type == "AIM" or model_type == "fermi_hubbard":
            id_loc = np.where(np.array([t[0] for t in model_paulis]) == "")[0][0]
            grid_point1 = [model_paulis[id_loc][1]]
            parameters = grid_point1 + [
                model_paulis[k][1] for k in range(len(model_paulis)) if k != id_loc
            ]
        else:
            parameters = [t[1] for t in model_paulis]

        for i, h in enumerate(model.H_terms):
            H_full += parameters[i] * h

        evals, evecs = np.linalg.eigh(H_full)

        # print(parameters)
        print("Real", evals[0])
        print("Approx", model.solve(parameters))
        if abs(evals[0]) < 1e-12:
            errors.append(np.abs(evals[0] - model.solve(parameters)))
        else:
            errors.append(np.abs(evals[0] - model.solve(parameters)) / np.abs(evals[0]))
        print(
            "Relative Error",
            errors[-1],
        )
        print()
        all_ps.append(parameters)
    print("Basis Size:", basis.shape[1])
    print("Full Hilbert Size:", model.H_terms[0].shape[0])
    plt.plot(errors, "o-")
    plt.xlabel("Test Case")
    plt.ylabel("Relative Error")
    plt.title("Surrogate Model Relative Errors")
    plt.yscale("log")
    plt.ylim(1e-20, 1)
    # plt.xticks(
    #     range(len(errors)),
    #     [f"({p[0]:.2f}, {p[n2b_terms + 1]:.2f})" for p in all_ps],
    #     rotation=90,
    # )
    plt.show()
