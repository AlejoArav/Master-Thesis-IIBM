#!/usr/bin/env python3
"""
Numerical steady-state finder for the RDIM contact-process attempt
under a homogeneous (well-mixed) assumption. Diffusion terms are
ignored; membrane exchange, production/decay, nutrient uptake, and
length-based growth are retained. Toxin/public good effects are
disabled (xi=0).

Requires: numpy, scipy
Usage:
    python find_equilibria.py --seeds 32 --growth-rate 0.0231 \
        --dm 1.0 --de 10.0 --nutrient-init 1.0 --nutrient-uptake 0.0333
"""

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.linalg import eigvals
from scipy.optimize import root

# ---- Parameters (mirror C defaults) ----
K_TX = 1e-1
K_TR = 5e0
D_MRNA = 5e-1
D_P = 5e-2
K_1R = 1.0 / 0.1
K_1S = 1.0 / 0.1
K_2R = 1.0 / 0.02
K_2S = 1.0 / 0.02
K_GR = 1.0 / 0.10
K_GS = 1.0 / 0.10
KM_ENZ = 10.0
K_LUXI = 0.04
K_LASI = 0.04
D_HSLI = 0.01
D_HSLE = 0.001
K_TET = 1.0 / 0.005
K_LAC = 1.0 / 0.100
N_R = 2.0
N_S = 2.0
N_REP1 = 2.0
N_REP2 = 4.0
B_0 = 5e-5
B_1 = 2.5e-4

ALPHA_CONST = (K_TX * K_TR) / (D_MRNA * D_P)
C_CONST = ALPHA_CONST * ALPHA_CONST
KTR_OVER_DMRNA = K_TR / D_MRNA

# ---- Helpers ----


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


def clip_nonnegative(x):
    return np.maximum(x, 0.0)


@dataclass
class Params:
    growth_rate: float
    dm: float
    de: float
    nutrient_init: float
    nutrient_diffusion: float  # unused in mean-field
    nutrient_decay: float
    nutrient_uptake: float
    kappa: float
    xi: float = 0.0  # toxin/public good off
    fixed_nutrient: bool = False
    length_fixed: bool = False
    length_init: float = 1.0
    c12e_fixed: float | None = None


def rhs(x: np.ndarray, p: Params) -> np.ndarray:
    """
    State vector x = [C6i, C12i, C6e, C12e, LUXI, LASI, TETR, LACI, RFP, GFP, nutrient, length]
    """
    C6i, C12i, C6e_var, C12e_var, LUXI, LASI, TETR, LACI, RFP, GFP, n_var, l_var = x
    n = p.nutrient_init if p.fixed_nutrient else n_var
    l = p.length_init if p.length_fixed else l_var
    C12e = p.c12e_fixed if p.c12e_fixed is not None else C12e_var
    C6e = C6e_var
    occ = 1.0  # occupied site

    pLT = promoter_function(C6i, TETR, K_GR, K_2R, K_1R, K_TET, N_R, N_REP1)
    pLL = promoter_function(C12i, LACI, K_GS, K_2S, K_1R, K_LAC, N_S, N_REP2)

    enzyme_c6 = occ * (K_LUXI * LUXI) / (KM_ENZ + LUXI + 1e-12)
    enzyme_c12 = occ * (K_LASI * LASI) / (KM_ENZ + LASI + 1e-12)

    expr_LT = occ * ((pLT * K_TX + B_0) * KTR_OVER_DMRNA)
    expr_LL = occ * ((pLL * K_TX + B_1) * KTR_OVER_DMRNA)

    dC6i = enzyme_c6 + occ * p.dm * (C6e - C6i) - D_HSLI * C6i
    dC12i = enzyme_c12 + occ * p.dm * (C12e - C12i) - D_HSLI * C12i
    dC6e = occ * p.dm * (C6i - C6e) - D_HSLE * C6e  # no diffusion term in mean-field
    dC12e = (
        0.0
        if p.c12e_fixed is not None
        else (occ * p.dm * (C12i - C12e) - D_HSLE * C12e)
    )

    dTETR = expr_LL - D_P * TETR
    dLACI = expr_LT - D_P * LACI
    dLUXI = expr_LT - D_P * LUXI
    dLASI = expr_LL - D_P * LASI
    dRFP = expr_LT - D_P * RFP
    dGFP = expr_LL - D_P * GFP

    uptake = p.nutrient_uptake * l * n / (p.kappa + n + 1e-12)
    dn = (
        0.0 if p.fixed_nutrient else (-p.nutrient_decay * n - uptake)
    )  # diffusion ignored here

    if p.length_fixed:
        dl = 0.0
    else:
        growth = p.growth_rate * l * n / (p.kappa + n + 1e-12) * (1.0 - p.xi * 0.0)
        dl = growth  # at steady state dl=0 implies either l=0 or n ~0 or growth=0

    return np.array(
        [dC6i, dC12i, dC6e, dC12e, dLUXI, dLASI, dTETR, dLACI, dRFP, dGFP, dn, dl]
    )


def find_roots(p: Params, seeds: int, scale: float) -> Sequence[dict]:
    roots = []
    for s in range(seeds):
        np.random.seed(s)
        x0 = np.abs(np.random.randn(12)) * scale
        x0[10] = p.nutrient_init  # nutrient initial guess
        x0[11] = p.length_init
        if p.c12e_fixed is not None:
            x0[3] = p.c12e_fixed
        sol = root(rhs, x0, args=(p,), method="hybr")
        if not sol.success:
            continue
        res_norm = np.linalg.norm(sol.fun, ord=np.inf)
        if res_norm > 1e-6:
            continue
        x = sol.x
        # classify stability via Jacobian
        eps = 1e-6
        J = np.zeros((12, 12))
        f0 = rhs(x, p)
        for i in range(12):
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
        description="Find homogeneous steady states for RDIM contact-process attempt."
    )
    ap.add_argument(
        "--seeds", type=int, default=16, help="number of random initial guesses"
    )
    ap.add_argument(
        "--growth-rate", type=float, default=math.log(2.0) / 30.0, help="per minute"
    )
    ap.add_argument(
        "--dm", type=float, default=1.0, help="membrane exchange coefficient"
    )
    ap.add_argument(
        "--de", type=float, default=10.0, help="external diffusion (unused here)"
    )
    ap.add_argument("--nutrient-init", type=float, default=1.0)
    ap.add_argument("--nutrient-diffusion", type=float, default=500.0 / 60.0)
    ap.add_argument("--nutrient-decay", type=float, default=0.0)
    ap.add_argument("--nutrient-uptake", type=float, default=2.0 / 60.0)
    ap.add_argument("--kappa", type=float, default=0.333)
    ap.add_argument(
        "--scale", type=float, default=0.1, help="scale for random initial guesses"
    )
    ap.add_argument(
        "--fixed-n",
        action="store_true",
        help="hold nutrient constant (no uptake/decay dynamics)",
    )
    ap.add_argument(
        "--length-fixed",
        action="store_true",
        help="hold cell length constant (no growth dynamics)",
    )
    ap.add_argument(
        "--length-init", type=float, default=1.0, help="initial/fixed length"
    )
    ap.add_argument(
        "--c12e-fixed",
        type=float,
        default=None,
        help="fix external C12 concentration to this value",
    )
    args = ap.parse_args()

    params = Params(
        growth_rate=args.growth_rate,
        dm=args.dm,
        de=args.de,
        nutrient_init=args.nutrient_init,
        nutrient_diffusion=args.nutrient_diffusion,
        nutrient_decay=args.nutrient_decay,
        nutrient_uptake=args.nutrient_uptake,
        kappa=args.kappa,
        fixed_nutrient=args.fixed_n,
        length_fixed=args.length_fixed,
        length_init=args.length_init,
        c12e_fixed=args.c12e_fixed,
    )

    roots = find_roots(params, seeds=args.seeds, scale=args.scale)
    if not roots:
        print("No roots found.")
        sys.exit(1)

    clusters = dedupe_roots(roots, tol=1e-4)
    print(f"Found {len(roots)} roots, {len(clusters)} clusters (tol=1e-4):")
    for idx, r in enumerate(clusters):
        x = r["x"]
        print(
            f"Cluster {idx} (n={r['count']}): res={r['res']:.2e}, unstable={r['unstable']}"
        )
        print("  C6i={:.3e} C12i={:.3e} C6e={:.3e} C12e={:.3e}".format(*x[:4]))
        print("  LUXI={:.3e} LASI={:.3e} TETR={:.3e} LACI={:.3e}".format(*x[4:8]))
        print("  RFP={:.3e} GFP={:.3e} n={:.3e} len={:.3e}".format(*x[8:]))
        unstable_eigs = [np.real(e) for e in r["eigs"] if np.real(e) > 1e-6]
        if unstable_eigs:
            print(f"  Unstable eigs (real parts): {unstable_eigs}")
        print()


if __name__ == "__main__":
    main()
