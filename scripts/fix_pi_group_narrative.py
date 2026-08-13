#!/usr/bin/env python
"""Correct the stale Pi-group narrative left over from H_ref = 30.

Three files describe the Richards residual's balance using numbers that were
computed when H_ref was 30 m and E_ref was 1e9. Raising H_ref to 165 changed
Pi_R_diff by 5.5x (it enters linearly) and the descriptions no longer match the
code's own output.

What actually changed
---------------------
    Pi_R_diff = T*K*H/(L^2*dth) = 1.4948e-03      (was 2.72e-4 at H_ref = 30)
    Pi_R_grav = K*T/(L*dth)     = 1.5401e-03      (unchanged; no H_ref in it)
    ratio grav/diff = L/H       = 1.0303

Gravity and capillary diffusion are now balanced to 3%. At H_ref = 30 the ratio
was 5.67 and gravity outweighed diffusion nearly six-fold.

Run with --write to apply. Anything it cannot match is listed for hand editing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EDITS = [
    # ---------------------------------------------------------------- residuals.py
    (
        "src/residuals.py",
        '"none"     - as written above; the diffusion term is O(1e-6)',
        '"none"     - as written above. RESOLVED: this is the setting to use.\n'
        '                         Pi_R_diff 1.4948e-3 and Pi_R_grav 1.5401e-3\n'
        '                         differ by 3%, so no term dominates INSIDE the\n'
        '                         Richards residual and there is nothing for a\n'
        '                         constant divisor to fix. Cross-equation\n'
        '                         weighting against the O(1) mechanical groups\n'
        '                         is what total_loss weights and GradNorm are\n'
        '                         for; dividing here rescales residual and\n'
        '                         gradient together and only relabels it.',
    ),
    (
        "src/residuals.py",
        "and diffusion O(H_ref/L_ref) = 0.176",
        "and diffusion O(H_ref/L_ref) = 0.971",
    ),
    (
        "src/residuals.py",
        "The Step 1.6 scaling\n        decision is still open, so it is a flag, "
        "not a hard-coded constant:",
        "The Step 1.6 scaling decision is\n        CLOSED as \"none\" (Step 3.2, "
        "see DECISIONS.md D-S.4). Kept as a flag\n        because the branches "
        "are cheap and the choice is worth being able\n        to re-test:",
    ),
    # ------------------------------------------------------------ validate_nondim.py
    (
        "scripts/validate_nondim.py",
        '"Pi_R_diff (T*K/L^2)"',
        '"Pi_R_diff (T*K*H/(L^2*dth))"',
    ),
    (
        "scripts/validate_nondim.py",
        '"Pi_R_grav (K*T/L)"',
        '"Pi_R_grav (K*T/(L*dth))"',
    ),
    (
        "scripts/validate_nondim.py",
        "With T_ref = 1 day the Richards diffusion group is ~3e-6, i.e. the\n"
        "        per-day timescale leaves the seepage term six orders below storage.\n"
        "        The script shows how choosing T_ref = L_ref^2 / K_ref (the diffusive\n"
        "        timescale) rebalances that term to O(1), which is what actually lets\n"
        "        GradNorm work rather than merely relabelling the imbalance.",
        "With T_ref = 1 day, Pi_R_diff = 1.4948e-3 and Pi_R_grav = 1.5401e-3:\n"
        "        capillary diffusion and gravity drainage are balanced to 3%, so\n"
        "        neither dominates the Richards residual. Both sit ~3 orders below\n"
        "        the O(1) mechanical groups, but that is a CROSS-equation weighting\n"
        "        problem for total_loss and GradNorm, not something a constant\n"
        "        divisor inside the residual can fix.\n"
        "        (This paragraph previously read '~3e-6 ... six orders below\n"
        "        storage'. That was computed at H_ref = 30 m; H_ref is now 165 and\n"
        "        Pi_R_diff carries it linearly.)",
    ),
    (
        "scripts/validate_nondim.py",
        "diffusion term is correctly tiny on a 1-day residual.",
        "diffusion term is small but NOT negligible on a 1-day residual --\n"
        "    over the 30-day window a signal diffuses ~4.5% of the pit depth.",
    ),
    (
        "scripts/validate_nondim.py",
        "With this choice Pi_R_diff = 1 exactly; the gravity group",
        "NOTE: Pi_R_diff = 1 would need T_ref = L^2*dth/(K*H) = 669 days, not\n"
        "    L^2/K. Against a 30-day window that abandons the rainfall transient\n"
        "    entirely. The gravity group",
    ),
]

CAVEAT_KEY = ("nondimensionalisation", "caveat")
NEW_CAVEAT = (
    "Both Richards groups are balanced: Pi_R_diff = 1.4948e-3 and "
    "Pi_R_grav = 1.5401e-3 differ by 3% (their ratio is exactly L_ref/H_ref = "
    "1.030). Neither capillary diffusion nor gravity drainage dominates, so "
    "T_ref = 1 day is kept -- it is the physically meaningful scale for a "
    "rainfall transient -- and the residual is used unnormalised "
    "(normalise='none'). Both groups sit about three orders below the O(1) "
    "mechanical groups; that is a cross-equation weighting problem for the "
    "loss weights and GradNorm, not something a constant divisor inside the "
    "Richards residual can fix, since dividing rescales the residual and its "
    "gradient together. Forcing Pi_R_diff = 1 would need T_ref = "
    "L^2*dtheta/(K*H) = 669 days, which abandons the transient the model "
    "exists to resolve. Over the 30-day window a pressure signal diffuses "
    "about 4.5% of the pit depth: small, not negligible. "
    "REVISED Step 3.2 -- the previous text argued from Pi_R_diff ~ 3e-6 and a "
    "916-year timescale, both computed with H_ref = 30 m and without "
    "dtheta_ref. State the choice in the methods section."
)

DATASET = Path("Phase 1/isikdere_phase1_dataset.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    applied, missed = [], []
    buffers: dict[str, str] = {}

    for path, old, new in EDITS:
        p = Path(path)
        if not p.exists():
            missed.append((path, "FILE NOT FOUND", old[:60]))
            continue
        if path not in buffers:
            buffers[path] = p.read_text()
        if old in buffers[path]:
            buffers[path] = buffers[path].replace(old, new, 1)
            applied.append((path, old.splitlines()[0][:64]))
        else:
            missed.append((path, "no match", old.splitlines()[0][:64]))

    caveat_ok = False
    if DATASET.exists():
        d = json.loads(DATASET.read_text())
        node = d
        for k in CAVEAT_KEY[:-1]:
            node = node.get(k, {})
        if CAVEAT_KEY[-1] in node:
            caveat_ok = True
        else:
            missed.append((str(DATASET), "caveat key not found", ""))
    else:
        missed.append((str(DATASET), "FILE NOT FOUND", ""))

    print(f"{len(applied)} replacement(s) matched:")
    for path, frag in applied:
        print(f"   OK   {path:32s} {frag}")
    if caveat_ok:
        print(f"   OK   {str(DATASET):32s} caveat")

    if missed:
        print(f"\n{len(missed)} NOT applied -- edit these by hand:")
        for path, why, frag in missed:
            print(f"   --   {path:32s} [{why}] {frag}")

    if not args.write:
        print("\n[dry run] nothing written. Re-run with --write.")
        return

    for path, text in buffers.items():
        Path(path).write_text(text)
    if caveat_ok:
        d = json.loads(DATASET.read_text())
        d["nondimensionalisation"]["caveat"] = NEW_CAVEAT
        DATASET.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    print("\nwritten.")


if __name__ == "__main__":
    main()