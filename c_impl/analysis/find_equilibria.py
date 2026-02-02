#!/usr/bin/env python3
"""
Homogeneous steady-state finder for the base RDIM C implementation (no growth layer).
Diffusion terms are ignored; membrane exchange, production/decay are retained.
External C12 can be fixed (default 0). C6 external is solved as part of the system.

Requires: numpy, scipy
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.linalg import eigvals
from scipy.optimize import root

# Parameters mirrored from rdim_model_simulations.c
K_TX = 1e-1
K_TR = 5e0
D_MRNA = 5e-1
D_P = 5e-2

K_1R = 1.0 / 0.1
K_1S = 1.0 / 0.1
K_2R = 1.0 / 0.02
K_2S = 1.0 / 0.02
K_GR = 1.0 / 0.010
K_GS = 1.0 / 0.010
KM_ENZ = 10.0

K_LUXI = 0.04
K_LASI = 0.04

D_HSLI = 0.01
D_HSLE = 0.001

K_TET = 1.0 / 0.001
K_LAC = 1.0 / 0.100
n_R = 2.0
n_S = 2.0
n_REP1 = 2.0
n_REP2 = 4.0
b_0 = 5e-5
b_1 = 2.5e-4

ALPHA_CONST = (K_TX * K_TR) / (D_MRNA * D_P)
C_CONST = ALPHA_CONST * ALPHA_CONST
KTR_OVER_DMRNA = K_TR / D_MRNA

# Coefficient grids (mirrored from C)
DM_COEFFS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 3.0, 5.0, 10.0]
DEXT_COEFFS = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0]
# Baseline C6 table (paired with DM indices)
C6_CONCS = [0.01148090, 0.00237850, 0.00124500, 0.00056350, 0.00033850,
            0.00026450, 0.00022800, 0.00015900, 0.00014450, 0.00013450]


def hill_power(x: np.ndarray, n: float) -> np.ndarray:
    if n == 2.0:
        return x * x
    if n == 4.0:
        return (x * x) * (x * x)
    return np.power(x, n)


def promoter_function(
    hsl: float,
    rep: float,
    k_gene_hsl: float,
    k_dimer: float,
    k_tf_hsl: float,
    k_rep: float,
    n_hsl: float,
    n_rep: float,
) -> float:
    tf_bind = k_tf_hsl * hsl
    frac = tf_bind / (1.0 + tf_bind)
    activator = k_gene_hsl * k_dimer * C_CONST * hill_power(frac, n_hsl)
    rep_term = hill_power(k_rep * rep, n_rep)
    denom = 1.0 + activator + rep_term
    return activator / denom


@dataclass
class Params:
    dm: float
    de: float  # not used in homogeneous steady state
    c6_conc: float
    c12_fixed: float | None = None


def rhs(x: np.ndarray, p: Params) -> np.ndarray:
    """
    State vector x = [C6i, C12i, C6e, C12e, LUXI, LASI, TETR, LACI, RFP, GFP]
    """
    C6i, C12i, C6e, C12e_var, LUXI, LASI, TETR, LACI, RFP, GFP = x
    C12e = p.c12_fixed if p.c12_fixed is not None else C12e_var

    pLT = promoter_function(C6i, TETR, K_GR, K_2R, K_1R, K_TET, n_R, n_REP1)
    pLL = promoter_function(C12i, LACI, K_GS, K_2S, K_1R, K_LAC, n_S, n_REP2)

    enzyme_c6 = (K_LUXI * LUXI) / (KM_ENZ + LUXI)
    enzyme_c12 = (K_LASI * LASI) / (KM_ENZ + LASI)

    expr_LT = (pLT * K_TX + b_0) * KTR_OVER_DMRNA
    expr_LL = (pLL * K_TX + b_1) * KTR_OVER_DMRNA

    dC6i = enzyme_c6 + p.dm * (C6e - C6i) - D_HSLI * C6i
    dC12i = enzyme_c12 + p.dm * (C12e - C12i) - D_HSLI * C12i
    dC6e = p.dm * (C6i - C6e) - D_HSLE * C6e
    dC12e = 0.0 if p.c12_fixed is not None else (p.dm * (C12i - C12e) - D_HSLE * C12e)

    dTETR = expr_LL - D_P * TETR
    dLACI = expr_LT - D_P * LACI
    dLUXI = expr_LT - D_P * LUXI
    dLASI = expr_LL - D_P * LASI
    dRFP = expr_LT - D_P * RFP
    dGFP = expr_LL - D_P * GFP

    return np.array([dC6i, dC12i, dC6e, dC12e, dLUXI, dLASI, dTETR, dLACI, dRFP, dGFP])


def find_roots(p: Params, seeds: int, scale: float) -> Sequence[dict]:
    roots = []
    for s in range(seeds):
        np.random.seed(s)
        x0 = np.abs(np.random.randn(10)) * scale
        x0[2] = p.c6_conc
        if p.c12_fixed is not None:
            x0[3] = p.c12_fixed
        sol = root(rhs, x0, args=(p,), method="hybr")
        if not sol.success:
            continue
        res_norm = np.linalg.norm(sol.fun, ord=np.inf)
        if res_norm > 1e-12:
            continue
        x = sol.x
        # Jacobian
        eps = 1e-6
        J = np.zeros((10, 10))
        f0 = rhs(x, p)
        for i in range(10):
            dx = np.zeros_like(x)
            dx[i] = eps
            fi = rhs(x + dx, p)
            J[:, i] = (fi - f0) / eps
        eigs = eigvals(J)
        unstable = np.any(np.real(eigs) > 1e-6)
        roots.append({"x": x, "res": res_norm, "eigs": eigs, "unstable": unstable})
    return roots


def dedupe_roots(roots: Sequence[dict], tol: float = 1e-4) -> Sequence[dict]:
    clusters = []
    for r in roots:
        x = r["x"]
        for c in clusters:
            if np.linalg.norm(x - c["x"], ord=np.inf) < tol:
                c["count"] += 1
                break
        else:
            new_r = dict(r)
            new_r["count"] = 1
            clusters.append(new_r)
    return clusters


def main():
    ap = argparse.ArgumentParser(
        description="Find homogeneous steady states for RDIM C implementation (no growth)."
    )
    ap.add_argument("--seeds", type=int, default=16, help="random initial guesses")
    ap.add_argument(
        "--scale", type=float, default=0.1, help="scale for random initial guesses"
    )
    ap.add_argument(
        "--c12-fixed",
        type=float,
        default=1e-6,
        help="fix C12 external to this value (set negative to solve for it)",
    )
    ap.add_argument(
        "--tol", type=float, default=1e-8, help="residual tolerance for accepting roots"
    )
    ap.add_argument(
        "--dedupe-tol", type=float, default=1e-4, help="tolerance for clustering roots"
    )
    args = ap.parse_args()

    dedupe_tol = args.dedupe_tol
    c12_fixed = None if args.c12_fixed is not None and args.c12_fixed < 0 else args.c12_fixed

    for de in DEXT_COEFFS:
        for dm, c6_base in zip(DM_COEFFS, C6_CONCS):
            p = Params(dm=dm, de=de, c6_conc=c6_base, c12_fixed=c12_fixed)
            roots = find_roots(p, seeds=args.seeds, scale=args.scale)
            roots = [r for r in roots if r["res"] <= args.tol]
            clusters = dedupe_roots(roots, tol=dedupe_tol)
            unstable_clusters = [c for c in clusters if c["unstable"]]
            print(
                f"DM={dm:.3g}, DE={de:.3g}, roots={len(roots)}, clusters={len(clusters)}, unstable={len(unstable_clusters)}"
            )
            for idx, r in enumerate(unstable_clusters):
                x = r["x"]
                print(
                    f"  Unstable cluster {idx} (n={r['count']}): C6e={x[2]:.6e}, C12e={x[3]:.6e}, GFP={x[9]:.6e}, RFP={x[8]:.6e}, res={r['res']:.2e}"
                )
                unstable_eigs = [np.real(e) for e in r["eigs"] if np.real(e) > 1e-6]
                print(f"    eigs+ (real parts): {unstable_eigs}")
            if unstable_clusters:
                first = unstable_clusters[0]["x"]
                print(f"  -> Suggested C6e start near {first[2]:.6e}")
            print()


if __name__ == "__main__":
    main()
