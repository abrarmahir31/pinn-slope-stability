#!/usr/bin/env python
"""Reconcile Phase 1/isikdere_phase1_dataset.json with nondim.py.

Run with --dry-run first (the default). It reports every divergence and only
writes when passed --write.

Scope
-----
FIXES (unambiguous):
  * H_ref_m 30 -> 165, with justification
  * the E_ref justification string, which omits Tm and names a lower bound no
    stratum has
  * a revision_note recording that Step 3.2 corrected these

REPORTS ONLY (needs your judgement):
  * the `groups` block, which uses a different nondimensionalisation from
    nondim.py -- Pi_R_grav omits dtheta_ref, Pi_R_diff omits both H_ref and
    dtheta_ref
  * `diffusive_timescale`, computed as L^2/K, likewise
  * the `caveat` paragraph, whose argument rests on Pi_R_diff ~ 3e-6 and a
    916-year timescale. Under the current definitions those become ~1.5e-3 and
    ~1.8 years, which changes the conclusion rather than just the numbers.

The caveat is methods-section material. Rewriting it is a physics decision, not
a bookkeeping one, so this script prints what changes and leaves it alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [".", "src", "src/step21_geometry"]
from src.nondim import SCALES as S            # noqa: E402
from src.materials import load_materials      # noqa: E402

DATASET = Path("Phase 1/isikdere_phase1_dataset.json")

E_REF_JUSTIFICATION = (
    "1e8 Pa, a MARL scale. u* = u/U_ref = E_ref/E_actual, so E_ref must sit "
    "within about an order of magnitude of the strata that actually deform. "
    "Adopted rock-mass moduli are Mk 2.09e8, Mk_d 3.81e7 and Tm 4.26e9 Pa -- a "
    "span of 112x, so no single E_ref puts all three within an order. Tm is "
    "excluded: it is the crystallised limestone basement, effectively rigid, "
    "and small u* there is physically correct rather than a scaling failure; "
    "the failure mechanism runs through the marls. The Mk/Mk_d geometric mean "
    "is 8.9e7, i.e. the adopted 1e8. The original 1e9 gave u* = 26 for Mk_d, "
    "stiffer than every deforming stratum in the model."
)

H_REF_JUSTIFICATION = (
    "165 m, the largest suction head in the domain. psi over the initial "
    "condition spans -3 to -161 m, so this puts psi* in [-1, 0]: the field "
    "fills the network's output range without saturating the activation. The "
    "original 30 m (max piezometric head above the coal seam) gave psi* down "
    "to -5.4."
)

REVISION_NOTE = (
    "Scales corrected during Step 3.2 (2026-08-13). H_ref 30 -> 165 m and the "
    "E_ref justification rewritten; E_ref itself was already 1e8 here but "
    "nondim.py still carried the superseded 1e9. Nothing bound the two, so "
    "they had drifted in opposite directions. tests/test_scale_consistency.py "
    "now provides that binding: every difference must appear in "
    "KNOWN_DIVERGENCES with a written reason. NOTE: the groups block and "
    "diffusive_timescale below still use the earlier nondimensionalisation "
    "and have NOT been recomputed -- see the caveat."
)


def live_groups():
    """What nondim.py actually uses, for comparison."""
    out = {}
    for name in ("Pi_R_diff", "Pi_R_grav", "Pi_R_hz", "Pi_M_body",
                 "Pi_M_couple"):
        if hasattr(S, name):
            out[name] = float(getattr(S, name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="apply the safe fixes (default is dry-run)")
    args = ap.parse_args()

    if not DATASET.exists():
        sys.exit(f"not found: {DATASET}")

    with open(DATASET) as f:
        d = json.load(f)

    nd = d["nondimensionalisation"]
    scales, just, groups = nd["scales"], nd["scale_justification"], nd["groups"]
    mats = load_materials()

    print("=" * 72)
    print("SCALES: dataset vs nondim.py")
    print("=" * 72)
    mapping = {
        "L_ref_m": S.L_ref, "T_ref_s": S.T_ref, "H_ref_m": S.H_ref,
        "sig_ref_Pa": S.sig_ref, "K_ref_mps": S.K_ref, "E_ref_Pa": S.E_ref,
        "rho_b_ref_kgpm3": S.rho_b_ref, "U_ref_m": S.U_ref,
    }
    for k, live in mapping.items():
        rec = scales.get(k)
        flag = "  " if rec is not None and abs(rec - live) < 1e-6 * abs(live) \
            else "->"
        print(f"{flag} {k:20s} dataset {rec!s:>14}   nondim.py {live:>14.6g}")

    print()
    print("=" * 72)
    print("GROUPS: dataset vs nondim.py  (REPORT ONLY -- not written)")
    print("=" * 72)
    for k, live in live_groups().items():
        rec = groups.get(k)
        flag = "  " if rec is not None and abs(rec - live) < 1e-4 * abs(live) \
            else "->"
        print(f"{flag} {k:20s} dataset {rec!s:>14}   nondim.py {live:>14.6g}")

    print()
    print("The dataset's groups omit dtheta_ref (and H_ref in Pi_R_diff).")
    print("Consequences for the caveat paragraph, which you must decide on:")
    t_diff_s = S.L_ref ** 2 * S.dtheta_ref / (S.K_ref * S.H_ref)
    print(f"  dataset:   Pi_R_diff 2.9896e-06,  T_diff 916.4 years")
    print(f"  current:   Pi_R_diff {getattr(S, 'Pi_R_diff', float('nan')):.4g},"
          f"  T_diff {t_diff_s / 86400 / 365.25:.2f} years")
    print("  The caveat argues from '~3e-6' and '~916 years'. Both change.")

    print()
    print("=" * 72)
    print("MODULI referenced by the E_ref justification")
    print("=" * 72)
    for tag, m in sorted(mats.items()):
        print(f"   {tag:6s} E = {m.E:.4g} Pa")
    print("   dataset says the moduli 'span 8e6 to 2.09e8' -- omits Tm, and no")
    print("   stratum has E = 8e6.")

    if not args.write:
        print("\n[dry run] nothing written. Re-run with --write to apply the "
              "H_ref, E_ref-justification and revision_note changes.")
        return

    scales["H_ref_m"] = float(S.H_ref)
    just["H_ref"] = H_REF_JUSTIFICATION
    just["E_ref"] = E_REF_JUSTIFICATION
    nd["revision_note"] = REVISION_NOTE

    with open(DATASET, "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("\nwritten. groups / diffusive_timescale / caveat left untouched.")


if __name__ == "__main__":
    main()