r"""Write Phase 6 arm files for `ssr_sweep.py --overrides`. DECISIONS REQUIRED.

    python scripts/make_arms.py --ks-stratum Tm --gsi-stratum Mk --out configs/arms

`--ks-stratum` replaces the plan's coal-seam K_s arm (Section 5 has no coal
seam) and `--gsi-stratum` picks the marl whose GSI moves. Neither has a
default. Rainfall arms are written only with --include-rainfall, because
tests/test_arms.py::test_rainfall_arms_are_currently_identical_boundary_conditions
shows 10/20/40 mm/hr are bit-identical BCs under the current infiltration
mode; running them spends GPU time to reproduce the baseline.
"""
import argparse
import sys

from src import arms


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks-stratum", required=True, choices=arms.SECTION5_TAGS)
    ap.add_argument("--gsi-stratum", required=True, choices=arms.SECTION5_TAGS)
    ap.add_argument("--ks-frac", type=float, default=0.20)
    ap.add_argument("--gsi-levels", default="40,50,60")
    ap.add_argument("--include-rainfall", action="store_true")
    ap.add_argument("--no-coupling", action="store_true")
    ap.add_argument("--out", default="configs/arms")
    a = ap.parse_args(argv)

    oat = arms.section5_oat_arms(
        ks_stratum=a.ks_stratum, gsi_stratum=a.gsi_stratum, ks_frac=a.ks_frac,
        gsi_levels=tuple(float(g) for g in a.gsi_levels.split(",")))
    oat = [x for x in oat if not x.is_base]
    if not a.include_rainfall:
        oat = [x for x in oat if x.factor != "rainfall"]
    coupling = [] if a.no_coupling else [
        x for x in arms.section5_coupling_arms() if not x.is_base]
    paths = arms.write_arm_files(oat + coupling, a.out)
    for arm, p in zip(oat + coupling, paths):
        flag = "" if arms.sigma0_consistent(arm.overrides) else \
            "   [moves E: sigma_0 not re-equilibrated]"
        print(f"{p}{flag}")
    print(f"\n{len(paths)} arm files. The baseline is the plain sweep "
          f"(no --overrides), so it has no file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
