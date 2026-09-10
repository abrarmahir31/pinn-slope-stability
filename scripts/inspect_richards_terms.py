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
three pieces. What matters is their SIZE relative to |R|:

  * all three terms individually ~ |R|  -> degenerate. Nothing is being
    balanced; the equation is trivially satisfied because every flux has
    collapsed. The loss value is meaningless and so is anything downstream.
  * terms >> |R| -> genuine cancellation. Storage is being balanced against
    the fluxes to some number of digits, which is what convergence looks like.

The ratio that decides it is CANCELLATION = max|term| / |R|. One digit of
cancellation is nothing; six digits is a solved equation.

CANCELLATION ALONE IS NOT ENOUGH, and reading it alone gives a false positive
at t = 0. `nondim.py:257` is explicit that a hydrostatic field gives EXACTLY
zero flux -- q*_z groups as -Pi_R_grav K* (Pi_R_hz dpsi*/dz* + 1), which
vanishes at dpsi*/dz* = -1/Pi_R_hz. The initial condition IS hydrostatic, so
both flux terms being tiny at t = 0 is the ansatz being right, not the field
being dead. Exactly the trap that made pde_richards = 5e-08 look like
convergence on Day 25.

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
from src.residuals import richards_residual, swcc_from_material
from src.sampling import sample_interior


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
    ap.add_argument("--n", type=int, default=4000,
                    help="interior collocation points (default 4000)")
    ap.add_argument("--sampling-seed", type=int, default=20260808)
    ap.add_argument("--t", type=float, nargs="+", default=[0.0, 1.0, 7.0, 30.0],
                    help="days at which to evaluate")
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args(argv)

    BOUNDS = bounds_from_geometry(t_max=30.0, s=SCALES)
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = full()
    cfg.device, cfg.dtype = "cpu", "float64"
    kw = {} if a.eps_psi is None else {"eps_psi": a.eps_psi}
    net = NearPhysical(PINN(cfg, BOUNDS), **kw)
    net.load_state_dict(d["net"])
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
           f"{'K* p50':>11}{'K/K_ic':>9}{'cancel':>9}  verdict")
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
            big = max(terms[k].detach().abs().median().item()
                      for k in ("storage", "diffusion", "gravity"))
            medR = absR.median().item()
            cancel = big / medR if medR > 0 else float("inf")

            # Low cancellation is only damning if the field has also left
            # the IC. Hydrostatic + low flux is correct; dry + low flux is
            # the collapse.
            if cancel >= 1e3:
                verdict = "solving"
            elif k_ratio < 0.1:
                verdict = "DRY-COLLAPSE"
            elif cancel < 10:
                verdict = "hydrostatic?"
            else:
                verdict = "weak"

            row = {"t_days": t_days, "tag": tag, "n": int(m.sum()),
                   "cancellation": cancel, "k_over_k_ic": k_ratio,
                   "verdict": verdict,
                   "R": q(R), "K_star": q(terms["K_star"]),
                   **{k: q(terms[k]) for k in
                      ("storage", "diffusion", "gravity")}}
            out["rows"].append(row)

            print(f"{t_days:>6.1f}{tag:>6}{int(m.sum()):>6}"
                  f"{medR:>12.3e}"
                  f"{terms['storage'].detach().abs().median():>12.3e}"
                  f"{terms['diffusion'].detach().abs().median():>12.3e}"
                  f"{terms['gravity'].detach().abs().median():>12.3e}"
                  f"{terms['K_star'].detach().median():>11.3e}"
                  f"{k_ratio:>9.2g}{cancel:>9.2g}  {verdict}")
        print()

    print("cancel = max(median |term|) / median |R|: the digits of "
          "cancellation between\nstorage and the fluxes. K/K_ic = trained K* "
          "over K* on the analytic\nhydrostatic IC at the same points.\n\n"
          "  solving       cancel >= 1e3. Storage and the fluxes genuinely "
          "balance.\n"
          "  weak          one or two digits, field still near the IC. Real "
          "but shallow.\n"
          "  hydrostatic?  cancel < 10 but K/K_ic ~ 1. Every term is tiny "
          "because the\n"
          "                field is still hydrostatic, which gives zero flux "
          "BY DESIGN\n"
          "                (nondim.py:257). Expected at t = 0. At t = 30 it "
          "means the\n"
          "                rain-flux BC has not moved anything.\n"
          "  DRY-COLLAPSE  K/K_ic < 0.1. The domain has dried past its own "
          "initial\n"
          "                condition, K_r has gone with it, and both flux "
          "terms are\n"
          "                structurally zero. pde_richards is then being "
          "BOUGHT, not\n"
          "                solved, and nothing downstream of it means "
          "anything.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())