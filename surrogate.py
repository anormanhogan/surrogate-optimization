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
        residue_threshold: float = 1e-6,
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
                    H_full = self._build_H_full(y * solution_grid.shape[1] + x)
                    Hr = basis.conj().T @ H_full @ basis
                    overlap = basis.conj().T @ basis
                    evals, evecs = sp.linalg.eigh(Hr, overlap)
                    answer_grid[y, x] = evals[0]

            plt.imshow(
                np.abs((answer_grid - solution_grid).real) + 1e-20,
                norm=mcolors.LogNorm(vmin=1e-20, vmax=1),
            )
            plt.colorbar(
                norm=mcolors.LogNorm(vmin=1e-20, vmax=1),
                label="Absolute Error",  # , ticks=[1e-20, 1e-len(mu), 1e0]
            )
            plt.scatter(
                chosen[0] % solution_grid.shape[1],  # type: ignore
                chosen[0] // solution_grid.shape[1],  # type: ignore
                marker="x",
                color="red",
                s=solution_grid.shape[0] ** 2,
                label="First choice",
            )
            plt.xlabel("p2")
            plt.ylabel("p1")
            plt.title("It 1, Basis Size 1")
            plt.show()

        # iteration
        num_iterations = len(not_chosen)
        for i in range(num_iterations):
            print("Iteration", i + 1)
            overlap = (basis.conj().T @ basis).real
            plt.imshow(overlap.real, label="Current Basis Vectors")
            plt.colorbar()
            plt.title(f"It {i + 1} Overlap")
            plt.show()
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
                # the trianing grid size and full Hilbert space size, these are
                # constructed on demand
                H_full = self._build_H_full(j)
                H2_full = H_full @ H_full  # self._build_H2_full(j)
                full_evals, full_evecs = np.linalg.eigh(H_full)

                Hr = basis.conj().T @ H_full @ basis
                H2r = basis.conj().T @ H2_full @ basis
                evals, evecs = sp.linalg.eigh(Hr, overlap)
                print("gs of Hr:", evals[0])
                print("gs of H_full", full_evals[0])
                print(
                    "gs of H2_full",
                    full_evecs[:, 0].conj().T @ H2_full @ full_evecs[:, 0],
                )
                # print("gs of H2_full squared:", np.linalg.eigh(H2_full)[0][0] ** 2)
                # print("gs of H_full squared:", np.linalg.eigh(H_full)[0][0] ** 2)
                # print("gs of Hr squared:", np.linalg.eigh(Hr)[0][0] ** 2)
                # print("gs of H2r squared:", np.linalg.eigh(H2r)[0][0] ** 2)

                # plt.imshow(
                #     np.abs(H2_full - H_full @ H_full) + 1e-20,
                #     norm=mcolors.LogNorm(vmin=1e-20, vmax=1),
                # )
                # plt.colorbar(
                #     norm=mcolors.LogNorm(
                #         vmin=1e-20, vmax=1
                #     )  # , ticks=[1e-20, 1e-10, 1e0]
                # )
                # plt.title("H2 - H^2 Full Matrix Difference")
                # plt.show()

                # find degeneracy of the ground state
                eps = 1e-10  # for comparing floating points of GSE
                degeneracy = 0
                for e in evals:
                    if abs(e - evals[0]) < eps:
                        degeneracy += 1
                    else:
                        break

                # plt.imshow(
                #     evecs[:, 0:degeneracy].real,
                #     label="Ground State Degeneracy Vectors for parameter index {}".format(
                #         j
                #     ),
                #     vmin=-1,
                #     vmax=1,
                #     aspect="auto",
                # )
                # plt.colorbar()
                # plt.title(
                #     "Ground State Degeneracy Vectors for parameter index {}".format(j)
                # )
                # for n in range(evecs[:, 0:degeneracy].shape[0]):
                #     for m in range(evecs[:, 0:degeneracy].shape[1]):
                #         # print(n, m)
                #         plt.text(
                #             m,
                #             n,
                #             f"{evecs[:, 0:degeneracy][n,m].real:.2e}",
                #             ha="center",
                #             va="center",
                #             color="r",
                #         )
                # # print(evecs[:, 0:degeneracy].shape)

                # plt.show()
                # inner_res = (H2r - evals[0] * evals[0] * overlap).real
                # plt.imshow(
                #     inner_res,
                #     label="H2r - E0^2 S",
                #     vmin=0,
                #     vmax=1,
                # )
                # plt.colorbar()
                # plt.title("H2r - E0^2 S for parameter index {}".format(j))
                # for n in range(inner_res.shape[0]):
                #     for m in range(inner_res.shape[1]):
                #         plt.text(
                #             n,
                #             m,
                #             f"{inner_res[n,m]:.2e}",
                #             ha="center",
                #             va="center",
                #             color="r",
                #         )
                # plt.show()
                # calculate residue
                res2 = 0
                for k in range(degeneracy):
                    res2 += (
                        evecs[:, k].conj().T
                        @ (H2r - ((evals[k] * evals[k]) * overlap))
                        @ evecs[:, k]
                    )
                    # print(evals[k] ** 2)
                    # print(np.linalg.eigh(H2r)[0][0] ** 2)
                    print("Residue contribution from state", k, ":", res2.real)
                    print(
                        "evals squared",
                        evecs[:, k].conj().T @ (evals[k] ** 2 * overlap) @ evecs[:, k],
                    )
                    print("H2r exp", evecs[:, k].conj().T @ H2r @ evecs[:, k])
                    print("exp overlap", evecs[:, k].conj().T @ overlap @ evecs[:, k])

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
                    markersize=12,
                    label="Next Choice",
                )
                plt.xlabel("Training Grid Index")
                plt.ylabel("Residue")
                plt.title(f"Termination Residues")
                plt.show()

                if type(solution_grid) != type(None):
                    answer_grid = np.zeros(solution_grid.shape, dtype=complex)  # type: ignore
                    count = 0
                    for y in range(solution_grid.shape[0]):  # type: ignore
                        for x in range(solution_grid.shape[1]):  # type: ignore
                            H_full = self._build_H_full(y * solution_grid.shape[1] + x)
                            Hr = basis.conj().T @ H_full @ basis
                            overlap = basis.conj().T @ basis
                            evals, evecs = sp.linalg.eigh(Hr, overlap)
                            answer_grid[y, x] = evals[0]

                            if y * solution_grid.shape[1] + x in not_chosen:
                                print(
                                    "Estimation:",
                                    answer_grid[y, x],
                                    "Actual:",
                                    solution_grid[y, x],
                                    "Residue:",
                                    residues[count],
                                )
                                count += 1

                    plt.imshow(
                        np.abs((answer_grid - solution_grid).real) + 1e-20,
                        norm=mcolors.LogNorm(vmin=1e-20, vmax=1),
                    )
                    plt.colorbar(
                        norm=mcolors.LogNorm(
                            vmin=1e-20, vmax=1
                        )  # , ticks=[1e-20, 1e-10, 1e0]
                    )
                    plt.scatter(
                        np.array(chosen) % solution_grid.shape[1],  # type: ignore
                        np.array(chosen) // solution_grid.shape[1],  # type: ignore
                        marker="o",
                        color="orange",
                        s=solution_grid.shape[0] ** 2,
                        label="Chosen Points",
                    )
                    plt.xlabel("p2")
                    plt.ylabel("p1")
                    plt.title(
                        f"Termination errors, Basis Size {basis.shape[1]}, U={4*self.training_grid[i][7].real:.2f}"
                    )
                    plt.show()

                break

            plt.plot(not_chosen, np.array(residues).real, "o-")
            plt.plot(
                [next_choice], [max_res2.real], "rx", markersize=12, label="Next Choice"
            )
            plt.xlabel("Training Grid Index")
            plt.ylabel("Residue")
            plt.title(f"It {i + 1} Residues")
            plt.show()

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
            plt.plot(basis_addition, label="Basis Addition Vectors")
            plt.plot(basis, label="Current Basis Vectors")
            plt.title(f"It {i + 1} Basis Addition")
            plt.legend()
            plt.show()

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
                        markersize=12,
                        label="Next Choice",
                    )
                    plt.xlabel("Training Grid Index")
                    plt.ylabel("Residue")
                    plt.title(f"Termination Residues")
                    plt.show()

                    if type(solution_grid) != type(None):
                        answer_grid = np.zeros(solution_grid.shape, dtype=complex)  # type: ignore
                        count = 0
                        for y in range(solution_grid.shape[0]):  # type: ignore
                            for x in range(solution_grid.shape[1]):  # type: ignore
                                H_full = self._build_H_full(
                                    y * solution_grid.shape[1] + x
                                )
                                Hr = basis.conj().T @ H_full @ basis
                                overlap = basis.conj().T @ basis
                                evals, evecs = sp.linalg.eigh(Hr, overlap)
                                answer_grid[y, x] = evals[0]
                                if y * solution_grid.shape[1] + x in not_chosen:
                                    print(
                                        "Estimation:",
                                        answer_grid[y, x],
                                        "Actual:",
                                        solution_grid[y, x],
                                        "Residue:",
                                        residues[count],
                                    )
                                    count += 1

                        plt.imshow(
                            np.abs((answer_grid - solution_grid).real) + 1e-20,
                            norm=mcolors.LogNorm(vmin=1e-20, vmax=1),
                        )
                        plt.colorbar(
                            norm=mcolors.LogNorm(
                                vmin=1e-20, vmax=1
                            )  # , ticks=[1e-20, 1e-10, 1e0]
                        )
                        plt.scatter(
                            np.array(chosen) % solution_grid.shape[1],  # type: ignore
                            np.array(chosen) // solution_grid.shape[1],  # type: ignore
                            marker="o",
                            color="orange",
                            s=solution_grid.shape[0] ** 2,
                            label="Chosen Points",
                        )
                        plt.xlabel("p2")
                        plt.ylabel("p1")
                        plt.title(
                            f"Termination errors, Basis Size {basis.shape[1]}, U={4*self.training_grid[i][7].real:.2f}"
                        )
                        plt.show()
                    break

                plt.plot(not_chosen, np.array(residues).real, "o-")
                plt.plot(
                    [next_choice],
                    [max_res2.real],
                    "rx",
                    markersize=12,
                    label="Next Choice",
                )
                plt.xlabel("Training Grid Index")
                plt.ylabel("Residue")
                plt.title(f"It {i + 1} Residues after Compression")
                plt.show()
            else:
                basis = copy.copy(basis_reduced)

            print("Looking at solutions with current basis of size", basis.shape[1])
            if type(solution_grid) != type(None):
                answer_grid = np.zeros(solution_grid.shape, dtype=complex)  # type: ignore
                count = 0
                for y in range(solution_grid.shape[0]):  # type: ignore
                    for x in range(solution_grid.shape[1]):  # type: ignore
                        H_full = self._build_H_full(y * solution_grid.shape[1] + x)
                        Hr = basis.conj().T @ H_full @ basis
                        overlap = basis.conj().T @ basis
                        evals, evecs = sp.linalg.eigh(Hr, overlap)
                        answer_grid[y, x] = evals[0]
                        if y * solution_grid.shape[1] + x in not_chosen:
                            print(
                                "Estimation:",
                                answer_grid[y, x],
                                "Actual:",
                                solution_grid[y, x],
                                "Residue:",
                                residues[count],
                            )
                            count += 1

                plt.imshow(
                    np.abs((answer_grid - solution_grid).real) + 1e-20,
                    norm=mcolors.LogNorm(vmin=1e-20, vmax=1),
                )
                plt.colorbar(
                    norm=mcolors.LogNorm(
                        vmin=1e-20, vmax=1
                    )  # , ticks=[1e-20, 1e-10, 1e0]
                )
                plt.scatter(
                    next_choice % solution_grid.shape[1],  # type: ignore
                    next_choice // solution_grid.shape[1],  # type: ignore
                    marker="x",
                    color="red",
                    s=solution_grid.shape[0] ** 2,
                    label="Next Choice",
                )
                plt.scatter(
                    np.array(chosen) % solution_grid.shape[1],  # type: ignore
                    np.array(chosen) // solution_grid.shape[1],  # type: ignore
                    marker="o",
                    color="orange",
                    s=solution_grid.shape[0] ** 2,
                    label="Chosen Points",
                )
                plt.xlabel("p2")
                plt.ylabel("p1")
                plt.title(
                    f"It {i + 2}, Basis Size {basis.shape[1]}, U={4*self.training_grid[i][7].real:.2f}"
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

    from openfermion import *
    import openfermion as of

    mu = np.linspace(-5, 5, 10)
    N = 6
    ps = 2  # Between 0 and N

    model_type = "FH"  # SIAM = Single Impurity Anderson Model, FH = Fermi Hubbard

    for U in [1.0, 5.0, 9.0]:

        hubbard = of.fermi_hubbard(
            1,
            N,
            tunneling=1.0,
            coulomb=1.0,
            chemical_potential=1.0,
            periodic=False,
        )

        print(of.jordan_wigner(hubbard))
        jw_hubbard = of.jordan_wigner(hubbard)
        ps_and_cs = of_operator_to_pauli_and_coeff(2 * N, jw_hubbard)

        H_paulis = [t[0] for t in ps_and_cs]

        if model_type == "SIAM":
            locs = []
            for i in range(0, 2 * N - 1, 2):
                loc = np.where([f"Z{i} Z{(i+1)}" == term for term in H_paulis])[0][0]
                locs.append(loc)
            print("locs", locs)
            locs = locs[1:]
            non_locs = [i for i in range(len(H_paulis)) if i not in locs]
            H_paulis = [H_paulis[i] for i in non_locs]

        training_grid = []
        for m1 in mu:
            for m2 in mu:
                train_h = of.jordan_wigner(
                    of.fermi_hubbard(
                        1,
                        N,
                        tunneling=m1,
                        coulomb=U,
                        chemical_potential=m2,
                        periodic=False,
                    )
                )
                ps_and_cs = of_operator_to_pauli_and_coeff(2 * N, train_h)
                if model_type == "FH":
                    training_grid.append([t[1] for t in ps_and_cs])
                elif model_type == "SIAM":
                    training_grid.append(
                        [t[1] for t in [ps_and_cs[k] for k in non_locs]]
                    )

        print(H_paulis)

        model = SurrogateModel(2 * N, H_paulis, training_grid, particle_selection=ps)

        model.build_terms()

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

        chosen, basis = model.optimize(solution_grid=solution_grid)
        print("Chosen Indices", chosen)
        print("Basis Size", basis.shape[1])
        # print(model.optimize(solution_grid=None))

        errors = []
        all_ps = []
        for i in range(30):
            H_full = np.zeros_like(model.H_terms[0], dtype=complex)

            # (J, Bz, u) = (3 * np.random.randn(2)).tolist() + [U]
            # parameters = [J] * n2b_terms + [Bz] * n1b_terms + [u] * n_int_terms

            test_h = of.jordan_wigner(
                of.fermi_hubbard(
                    1,
                    N,
                    tunneling=2 * np.random.randn(),
                    coulomb=U,
                    chemical_potential=2 * np.random.randn(),
                    periodic=False,
                )
            )
            ps_and_cs = of_operator_to_pauli_and_coeff(2 * N, test_h)
            if model_type == "FH":
                parameters = [t[1] for t in ps_and_cs]
            elif model_type == "SIAM":
                parameters = [t[1] for t in [ps_and_cs[k] for k in non_locs]]

            for i, h in enumerate(model.H_terms):
                H_full += parameters[i] * h

            evals, evecs = np.linalg.eigh(H_full)

            # print(parameters)
            print("Real", evals[0])
            print("Approx", model.solve(parameters))
            if abs(evals[0]) < 1e-12:
                errors.append(np.abs(evals[0] - model.solve(parameters)))
            else:
                errors.append(
                    np.abs(evals[0] - model.solve(parameters)) / np.abs(evals[0])
                )
            print(
                "Relative Error",
                errors[-1],
            )
            print()
            all_ps.append(parameters)

        plt.plot(errors, "o-")
        plt.xlabel("Test Case")
        plt.ylabel("Relative Error for U={:.2f}".format(U))
        plt.title("Surrogate Model Relative Errors")
        plt.yscale("log")
        plt.ylim(1e-20, 1)
        # plt.xticks(
        #     range(len(errors)),
        #     [f"({p[0]:.2f}, {p[n2b_terms + 1]:.2f})" for p in all_ps],
        #     rotation=90,
        # )
        plt.show()


###############################################################
