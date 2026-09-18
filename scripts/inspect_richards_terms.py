r"""Is `pde_richards` small because the equation is SOLVED, or because every
term in it is individually zero?

A squared-residual loss cannot tell those apart, and this project has now been
caught by that twice from opposite directions:

  Day 25 -- eps_psi = 3e-3 capped psi excursion at 0.5 m of head, psi0 was an
    exact solution of the residual, and pde_richards sat at 5e-08 for 14k
    epochs. Structural, not converged.
  Day 26 -- prod02 reaches pde_richards = 2.98e-05 with an IC residual of
    9.09e-01 starred (150 m of head, 95% of the field's own range) and a mean
    psi at t=0 of -161 m, which is the analytic IC's MINIMUM. The suspicion is
    that the optimiser found the dry corner: K_r ~ 1.9e-07 in Mk and 2.5e-13
    in Tm at -235 m, so both flux terms vanish and the residual is free.

The mixed-form residual is

    R = C*(psi*) dpsi*/dt*  -  Pi_R_diff div*[K* grad* psi*]  -  Pi_R_grav dK*/dz*
        \_____ storage ____/   \________ diffusion _________/   \____ gravity ___/

`residuals.richards_residual(..., return_terms=True)` already hands back the
three pieces. What matters is the size of the FLUXES against STORAGE.

SUPERSEDED (Day 28) — this script used to measure
CANCELLATION = max|term| / |R|, on the reasoning that terms >> |R| means
storage and the fluxes are balancing to some number of digits. That ratio is
pinned at 1.000 on every production row ever measured
(`docs/richards_terms_prod02.json`, all twelve): |R| IS the storage term to
8-9 significant figures, because the fluxes are eight orders below it and
never enter the sum. So `max|term|` is storage, `|R|` is storage, and the
ratio is storage/storage. The `cancel >= 1e3` branch was unreachable and
every verdict was decided by K/K_ic alone.

The measure is now `residuals.flux_fraction`, computed POINTWISE:

    p = max(|diffusion|, |gravity|) / (|storage| + |diffusion| + |gravity|)

reported as `reach` = p / `residuals.attainable_flux_fraction(mat)`, because
the attainable ceiling is a MATERIAL property spanning 4.03e-04 (Mk) to
9.12e-01 (Tm) and absolute p is not comparable across strata. See D-A.2.

THE FLUX FRACTION ALONE IS NOT ENOUGH, and reading it alone gives a false
positive at t = 0. `nondim.py:257` is explicit that a hydrostatic field gives
EXACTLY zero flux -- q*_z groups as -Pi_R_grav K* (Pi_R_hz dpsi*/dz* + 1),
which vanishes at dpsi*/dz* = -1/Pi_R_hz. The initial condition IS
hydrostatic, so both flux terms being tiny at t = 0 is the ansatz being right,
not the field being dead. Exactly the trap that made pde_richards = 5e-08 look
like convergence on Day 25.

So the script carries a CONTROL: K* evaluated on the trained psi*, against K*
evaluated on the analytic IC psi*0 = -(L z* - 197)/H at the same points. The
K/K_ic column is the dry-corner detector.

    K/K_ic ~ 1        the field is near the IC; low flux is hydrostatic and fine
    K/K_ic << 1       the domain has dried PAST the IC and the fluxes are dead
                      for the wrong reason -- the residual is being bought,
                      not solved
    K/K_ic >> 1       wetting, which is what the Tm rain-flux BC should produce

Reported per stratum, because Tm's K_r is five orders below Mk's at the same
suction and a domain-wide mean would hide a Tm-only collapse (Tm is 87.3% of
the domain by area).

    PYTHONPATH=. python scripts/inspect_richards_terms.py runs/prod02/ckpt_002000.pt
    PYTHONPATH=. python scripts/inspect_richards_terms.py CKPT --eps-psi 3e-3   # pre-2 Sep
    PYTHONPATH=. python scripts/inspect_richards_terms.py CKPT --json out.json
"""
import argparse
import json

import torch

from src.config import bounds_from_geometry, full
from src.loss import psi_of
from src.materials import load_materials
from src.model import PINN, NearPhysical
from src.nondim import SCALES
from src.residuals import (attainable_flux_fraction, flux_fraction,
                           richards_residual, swcc_from_material)
from src.sampling import sample_interior


def net_from_ckpt(d, cfg, BOUNDS, eps_psi=None, mode=None, cap_k=None):
    """Rebuild the ansatz the checkpoint was TRAINED with.

    `train.py` stores `cfg: vars(a)` in every checkpoint, so `ansatz`,
    `cap_k` and `eps_psi` are all on disk. Neither diagnostic script read
    them until Day 28, and reconstructing with NearPhysical's defaults is
    not a cosmetic mislabel -- `mode` changes the FORWARD PASS, so an `exp`
    or `cap` checkpoint loaded without it is evaluated as `unbounded` and
    yields a different field from the same weights. Symptom: psi >= 0 points
    appearing in a run of a bounded arm.

    Explicit arguments still override, for checkpoints predating the key.
    """
    saved = d.get("cfg") or {}
    eps = eps_psi if eps_psi is not None else saved.get("eps_psi")
    md = mode if mode is not None else saved.get("ansatz")
    ck = cap_k if cap_k is not None else saved.get("cap_k")
    kw = {}
    if eps is not None:
        kw["eps_psi"] = float(eps)
    if md is not None:
        kw["mode"] = md
    if ck is not None:
        kw["cap_k"] = float(ck)
    uv = saved.get("eps_uv")   # Day 42: rebuild at the
    if uv is not None:         # checkpoint's eps_uv, not the default
        kw["eps_uv"] = float(uv)
    if md is None:
        print("WARNING: checkpoint carries no 'ansatz' key; falling back to "
              "NearPhysical's default. If this was a cap/exp run the field "
              "below is WRONG. Pass --mode explicitly.")
    net = NearPhysical(PINN(cfg, BOUNDS), **kw)
    net.load_state_dict(d["net"])
    return net


def q(v, ps=(50, 90, 99, 100)):
    """Median/tail summary. Means are useless here -- a heavy-tailed residual
    is exactly what Day 26's seed-variance finding was about."""
    a = v.detach().abs().reshape(-1)
    return {f"p{p}": float(torch.quantile(a, p / 100.0)) for p in ps}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--eps-psi", type=float, default=None,
                    help="ansatz prefactor the checkpoint was TRAINED with. "
                         "Default: NearPhysical's current default (0.3). "
                         "Anything before 2 Sep needs 3e-3.")
    ap.add_argument("--mode", default=None, choices=("unbounded", "cap", "exp"),
                    help="override the ansatz mode. Default: read from the "
                         "checkpoint's cfg, which is where it belongs.")
    ap.add_argument("--cap-k", type=float, default=None)
    ap.add_argument("--n", type=int, default=4000,
                    help="interior collocation points (default 4000)")
    ap.add_argument("--sampling-seed", type=int, default=20260808)
    ap.add_argument("--t", type=float, nargs="+", default=[0.0, 1.0, 7.0, 30.0],
                    help="days at which to evaluate")
    ap.add_argument("--p-ref", type=float, default=1e-3,
                    help="reference for the reported live fraction. NOT a\n"
                         "calibrated threshold; see scripts/flux_fraction.py --why.")
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args(argv)

    BOUNDS = bounds_from_geometry(t_max=30.0, s=SCALES)
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = full()
    cfg.device, cfg.dtype = "cpu", "float64"
    net = net_from_ckpt(d, cfg, BOUNDS, eps_psi=a.eps_psi, mode=a.mode,
                        cap_k=a.cap_k)
    net.eval()

    print(f"checkpoint: {a.ckpt}   step: {d.get('step')}")
    print(f"ansatz:     {net.summary()}")
    print(f"Pi_R_diff = {SCALES.Pi_R_diff:.4e}   "
          f"Pi_R_grav = {SCALES.Pi_R_grav:.4e}\n")

    coll = sample_interior(a.n, a.sampling_seed)
    mats = load_materials()
    tags = sorted(set(coll.tag.tolist()))

    def fields(x, z, t):
        return psi_of(net(x, z, t))

    out = {"ckpt": a.ckpt, "step": d.get("step"), "n": a.n,
           "eps_psi": a.eps_psi, "rows": []}

    L, H = SCALES.L_ref, SCALES.H_ref
    hdr = (f"{'t (d)':>6}{'tag':>6}{'n':>6}"
           f"{'|R| p50':>12}{'storage':>12}{'diffusion':>12}{'gravity':>12}"
           f"{'p p50':>10}{'reach':>8}{'K/K_ic':>9}{'psi>=0':>8}  verdict")
    print(hdr)
    print("-" * len(hdr))

    for t_days in a.t:
        for tag in tags:
            m = torch.as_tensor(coll.tag == tag)
            if not m.any():
                continue
            x = coll.x[m].reshape(-1, 1).clone().requires_grad_(True)
            z = coll.z[m].reshape(-1, 1).clone().requires_grad_(True)
            t = torch.full_like(x, t_days).requires_grad_(True)

            R, terms = richards_residual(fields, x, z, t, mats[tag],
                                         s=SCALES, return_terms=True)

            # CONTROL: the same K* on the analytic hydrostatic IC. Without
            # this, a hydrostatic field and a collapsed one look identical.
            psi_ic = -(L * z.detach() - 197.0) / H
            K_ic = swcc_from_material(mats[tag], SCALES).K_star(psi_ic)
            k_ratio = (terms["K_star"].detach().median()
                       / K_ic.median()).item()

            absR = R.detach().abs()
            medR = absR.median().item()

            # THE REGIME MEASURE. Was `cancel = max(median|term|)/median|R|`,
            # which is pinned at 1.000 on every production row measured so far
            # (docs/richards_terms_prod02.json, all twelve): |R| IS storage to
            # 8-9 significant figures, so `max|term|` is storage and the ratio
            # is storage/storage. The `cancel >= 1e3` branch was unreachable
            # and every label was decided by k_ratio alone. `flux_fraction`
            # compares the fluxes to storage instead, which is the quantity
            # the question was always about.
            p = flux_fraction(terms)
            ceiling = attainable_flux_fraction(mats[tag], SCALES)
            # Normalised by what THIS material can reach. Tm's ceiling is
            # 0.912 and the marls' is ~5e-04, so an absolute cut would call
            # the marls dead at every saturation. See D-A.2.
            reach = float(p.median()) / ceiling
            live = float((p > a.p_ref).double().mean())

            psi_now = fields(x, z, t).detach().reshape(-1)
            sat_frac = float((psi_now >= 0).double().mean())

            # Ordered by how badly each condition invalidates the others.
            if sat_frac > 0.0:
                # C* is identically zero at psi = 0, so storage VANISHES and
                # p reads 1.000 by construction. The equation has changed type
                # at those points; nothing else on this row means anything.
                verdict = "UNPHYSICAL"
            elif k_ratio < 0.1:
                verdict = "DRY-COLLAPSE"
            elif reach >= 0.1:
                verdict = "solving"
            elif t_days == 0.0 and 0.5 <= k_ratio <= 2.0:
                # Correct ONLY at t = 0: the IC is hydrostatic and gives zero
                # flux by design (nondim.py:257). Never a valid label later.
                verdict = "hydrostatic"
            elif t_days == 0.0:
                # At t = 0 the field IS the analytic IC, so K/K_ic must be ~1.
                # Checked BEFORE the wetting branch: at t = 0 there has been
                # no time to wet, so k_ratio > 2 there is a misfit, not a
                # front. prod02 reports 0.223 in Mk and the old code labelled
                # it `hydrostatic?`, which it cannot be.
                verdict = "IC MISFIT"
            elif k_ratio > 2.0:
                # Wetter than the IC with the fluxes still dead. The rain BC
                # has moved water in and nothing is transporting it.
                verdict = "WETTING, NO FLUX"
            else:
                verdict = "STALLED"

            row = {"t_days": t_days, "tag": tag, "n": int(m.sum()),
                   "flux_fraction_p50": float(p.median()),
                   "ceiling": ceiling, "reach": reach, "live_frac": live,
                   "psi_ge0_frac": sat_frac,
                   "k_over_k_ic": k_ratio, "verdict": verdict,
                   "R": q(R), "K_star": q(terms["K_star"]),
                   **{k: q(terms[k]) for k in
                      ("storage", "diffusion", "gravity")}}
            out["rows"].append(row)

            print(f"{t_days:>6.1f}{tag:>6}{int(m.sum()):>6}"
                  f"{medR:>12.3e}"
                  f"{terms['storage'].detach().abs().median():>12.3e}"
                  f"{terms['diffusion'].detach().abs().median():>12.3e}"
                  f"{terms['gravity'].detach().abs().median():>12.3e}"
                  f"{float(p.median()):>10.2e}{reach:>8.3f}"
                  f"{k_ratio:>9.2g}{sat_frac:>8.3f}  {verdict}")
        print()

    print("p      = median flux fraction, max(|diff|,|grav|)/(|stor|+|diff|+|grav|),\n"
          "         computed POINTWISE then reduced. Replaces `cancel`, which was\n"
          "         pinned at 1.000 on every production row because |R| is storage.\n"
          "reach  = p / the ceiling THIS material can attain (Mk 4.03e-04,\n"
          "         Mk_d 4.92e-04, Tm 9.12e-01). Absolute p is not comparable\n"
          "         across strata; reach is.\n"
          "K/K_ic = trained K* over K* on the analytic hydrostatic IC.\n"
          "psi>=0 = fraction of points at or above saturation.\n\n"
          "  solving           reach >= 0.1. The fluxes are doing a real share of\n"
          "                    what this material permits.\n"
          "  hydrostatic       t = 0 only, K/K_ic ~ 1. Zero flux is correct there\n"
          "                    BY DESIGN (nondim.py:257).\n"
          "  IC MISFIT         t = 0 but K/K_ic is not ~1. The field does not match\n"
          "                    its own initial condition; prod02 reports 0.223 in Mk.\n"
          "  WETTING, NO FLUX  K/K_ic > 2 with dead fluxes. Water has arrived and\n"
          "                    nothing is moving it. Was mislabelled `hydrostatic?`.\n"
          "  STALLED           t > 0, field near the IC, fluxes dead. The BC has not\n"
          "                    moved anything.\n"
          "  DRY-COLLAPSE      K/K_ic < 0.1. Dried past its own IC, K_r gone with it,\n"
          "                    both flux terms structurally zero. pde_richards is\n"
          "                    being BOUGHT, not solved.\n"
          "  UNPHYSICAL        psi >= 0 present. C* is identically zero at\n"
          "                    saturation, so storage vanishes and p reads 1.000 by\n"
          "                    construction. The equation has changed TYPE at those\n"
          "                    points. Every other column on the row is void.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())