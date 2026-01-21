############ Laplace transformation ############

import bisect
import numpy as np


def _find_laplace(laplace_holder: dict, R, error):
    """
    find the laplace holder with the smallest error that is larger than the given error

    Args:
        laplace_holder: dict
            a dictionary that contains all the laplace holder
        R: float
            1/x is fitted as summation of exponential functions on interval [1,R]
        error: float
            the relative error threshold

    Return:

    """

    ### find key via binary search ###

    keys = list(laplace_holder.keys())
    keys.sort()

    index = bisect.bisect_left(keys, R)
    if index == len(keys):
        return None
    else:
        key = keys[index]
        items = laplace_holder[key]
        items.sort(key=lambda x: x["error"], reverse=True)
        for item in items:
            if item["error"] <= error:
                return item
        return None


def _build_laplace_holder(r_min, r_max, rel_error):
    """ """

    import os, pickle

    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, "laplace.pkl")
    with open(file_path, "rb") as f:
        laplace_holder = pickle.load(f)

    item_found = _find_laplace(laplace_holder, r_max / r_min, rel_error)

    if item_found is None:
        raise NotImplementedError("No laplace holder found")

    return {
        "a_values": np.array(item_found["a_values"]) / r_min,
        "b_values": np.array(item_found["b_values"]) / r_min,
        "degree": item_found["degree"],
        "error": item_found["error"],
    }


class laplace_holder:
    r"""laplace transformation of energy denominator
    For order 2, 1/(ea+eb-ei-ej) ~ \sum_T (tao_v)_a^T (tao_v)_b^T (tao_o)_i^T (tao_o)_j^T

    Ref:
        (1) Almlof1992   : J. Chem. Phys. 96, 489-494 (1992) https://doi.org/10.1063/1.462485
        (2) Hackbusch2008: J. Chem. Phys. 129, 044112 (2008) https://doi.org/10.1063/1.2958921
        (3) https://gitlab.mis.mpg.de/scicomp/EXP_SUM

    """

    _keys = {
        "mo_ene",
        "nocc",
        "order",
        "holder",
        "a_values",
        "b_values",
        "_degree",
        "_error",
        "laplace_occ",
        "laplace_vir",
    }

    def __init__(self, mo_ene, nocc, order=2, rel_error=1e-6, verbose=True):

        occ_ene = mo_ene[:nocc]
        vir_ene = mo_ene[nocc:]

        max_occ = np.max(occ_ene)
        min_occ = np.min(occ_ene)
        min_vir = np.min(vir_ene)
        max_vir = np.max(vir_ene)

        if max_occ > min_vir + 1e-8:
            print("Warning: max_occ > min_vir")
            raise NotImplementedError

        r_min = (min_vir - max_occ) * order
        r_max = (max_vir - min_occ) * order

        self.mo_ene = mo_ene
        self.nocc = nocc
        self.order = order
        self.holder = _build_laplace_holder(r_min, r_max, rel_error)
        self.a_values = self.holder["a_values"]
        self.b_values = self.holder["b_values"]
        self._degree = self.holder["degree"]
        self._error = self.holder["error"]

        self.laplace_occ = self._build_laplace_occ(occ_ene, order=order)
        self.laplace_vir = self._build_laplace_vir(vir_ene, order=order)

    @property
    def degree(self):
        return self._degree

    @property
    def error(self):
        return self._error

    @property
    def delta_full(self):
        if self.order != 2:
            raise NotImplementedError
        else:
            return np.einsum(
                "iP,jP,aP,bP->ijab",
                self.laplace_occ,
                self.laplace_occ,
                self.laplace_vir,
                self.laplace_vir,
            )

    def _build_laplace_occ(self, occ_ene, order=2):

        nocc = len(occ_ene)
        degree = self.degree
        res = np.zeros((nocc, degree))
        order2 = 2 * order

        for icol, (a, b) in enumerate(zip(self.a_values, self.b_values)):
            res[:, icol] = (a ** ((1.0 / (float(order2))))) * np.exp(b * occ_ene)

        return res

    def _build_laplace_vir(self, vir_ene, order=2):

        nvir = len(vir_ene)
        degree = self.degree
        res = np.zeros((nvir, degree))
        order2 = 2 * order

        print("vir_ene = ", vir_ene)

        for icol, (a, b) in enumerate(zip(self.a_values, self.b_values)):
            res[:, icol] = (a ** ((1.0 / (float(order2))))) * np.exp(-b * vir_ene)

        return res


if __name__ == "__main__":

    # generate a pyscf example #

    from pyscf import gto, scf

    mol = gto.Mole()
    mol.atom = """
    O  0.   0.       0.
    H  0.   -0.757   0.587
    H  0.   0.757    0.587
    """
    mol.basis = "cc-pvdz"
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()

    mo_ene = mf.mo_energy
    nocc = mol.nelectron // 2
    laplace = laplace_holder(mo_ene, nocc, order=2, rel_error=1e-6, verbose=True)

    # benchmark #

    delta_ijab = np.zeros((nocc, nocc, mo_ene.size - nocc, mo_ene.size - nocc))
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nocc, mo_ene.size):
                for b in range(nocc, mo_ene.size):
                    delta_ijab[i, j, a - nocc, b - nocc] = 1.0 / (
                        mo_ene[a] + mo_ene[b] - mo_ene[i] - mo_ene[j]
                    )

    print("degree = ", laplace.degree)
    print("error  = ", laplace.error)
    # print("delta_full shape = ", laplace.delta_full.shape)
    # print("delta_full = ", laplace.delta_full)
    # print("delta_ijab = ", delta_ijab)
    print(
        "relative error = ",
        np.linalg.norm(laplace.delta_full - delta_ijab) / np.linalg.norm(delta_ijab),
    )
