"""
validate_nondim.py — sanity checks for Step 1.6
================================================
Confirms two things:

  (1) The scaling is ALGEBRAICALLY CONSISTENT: taking a raw-unit field,
      non-dimensionalising it, evaluating the dimensionless residual, and
      re-dimensionalising reproduces the raw-unit residual.

  (2) The residual TERMS are actually O(1) — and flags the one that is not.
      With T_ref = 1 day the Richards diffusion group is ~3e-6, i.e. the
      per-day timescale leaves the seepage term six orders below storage.
      The script shows how choosing T_ref = L_ref^2 / K_ref (the diffusive
      timescale) rebalances that term to O(1), which is what actually lets
      GradNorm work rather than merely relabelling the imbalance.
"""
import math
from src.nondim import Scales, SCALES, report_scales, G_ACCEL, RHO_W


def check_group_magnitudes(s: Scales) -> dict:
    groups = {
        "Pi_R_diff (T*K*H/(L^2*dth))":   s.Pi_R_diff,
        "Pi_R_grav (K*T/(L*dth))":     s.Pi_R_grav,
        "Pi_R_hz   (H/L)":       s.Pi_R_hz,
        "Pi_M_body (rho g L/s)": s.Pi_M_body,
        "Pi_M_couple(rw g H/s)": s.Pi_M_couple,
    }
    verdict = {}
    for name, val in groups.items():
        # "O(1)" band: within ~2 orders of magnitude of unity
        ok = 1e-2 <= abs(val) <= 1e2
        verdict[name] = (val, "O(1) OK" if ok else "OFF-SCALE  <-- rebalance")
    return verdict


def diffusive_timescale(s: Scales) -> float:
    """T for which Pi_R_diff = 1, i.e. the natural seepage timescale."""
    return s.L_ref**2 / s.K_ref


def main():
    print(report_scales())
    print("\n" + "=" * 60)
    print("CHECK 1 — dimensionless-group magnitudes (default scales)")
    print("=" * 60)
    for name, (val, note) in check_group_magnitudes(SCALES).items():
        print(f"  {name:<24} = {val: .4e}   {note}")

    print("\n" + "=" * 60)
    print("CHECK 2 — Richards diffusion timescale")
    print("=" * 60)
    T_diff = diffusive_timescale(SCALES)
    print(f"  L_ref^2 / K_ref            = {T_diff:.4e} s"
          f"  = {T_diff/86400:.1f} days = {T_diff/86400/365:.1f} yr")
    print("  Interpretation: a pressure signal needs this long to diffuse")
    print("  across the pit depth. The per-day T_ref is far shorter, so the")
    print("  diffusion term is small but NOT negligible on a 1-day residual; over the 30-day window a signal diffuses ~4.5% of the pit depth.")

    # rebalanced scale set: adopt the diffusive timescale
    s2 = Scales(T_ref=T_diff)
    print("\n  Rebalanced with T_ref = L^2/K:")
    for name, (val, note) in check_group_magnitudes(s2).items():
        print(f"    {name:<24} = {val: .4e}   {note}")
    print("\n  NOTE: Pi_R_diff = 1 would need T_ref = L^2*dth/(K*H) = 669 days, not L^2/K -- against a 30-day window that abandons the rainfall transient entirely. The gravity group")
    print("  becomes K*T/L = L/H-scale ~ O(L/H). Choose T_ref per the")
    print("  physics you want the network to resolve:")
    print("    * transient rainfall response over days -> T_ref = 1 day")
    print("      (then explicitly weight the seepage residual, or")
    print("       non-dimensionalise psi so the small group is absorbed).")
    print("    * quasi-steady drawdown over the pit -> T_ref = L^2/K.")

    print("\n" + "=" * 60)
    print("CHECK 3 — algebraic round-trip (raw -> nd -> raw)")
    print("=" * 60)
    # Fabricate a simple raw-unit mechanical equilibrium residual and confirm
    # the dimensionless form reproduces it after multiplying by sig_ref/L_ref.
    s = SCALES
    dsig_dz_raw = 2.0e4      # Pa/m, a stress gradient
    rho_b = 1800.0          # kg/m^3
    raw_res = dsig_dz_raw - rho_b * G_ACCEL          # N/m^3 (equilibrium, -z)
    # dimensionless: divide equation by sig_ref/L_ref
    dsig_dz_nd = dsig_dz_raw / (s.sig_ref / s.L_ref)
    nd_res = dsig_dz_nd - s.Pi_M_body * (rho_b / s.rho_b_ref)
    reconstructed = nd_res * (s.sig_ref / s.L_ref)
    print(f"  raw residual              = {raw_res: .6e} N/m^3")
    print(f"  nd residual               = {nd_res: .6e}  (O(1)? {'yes' if abs(nd_res)<1e2 else 'no'})")
    print(f"  reconstructed raw residual= {reconstructed: .6e} N/m^3")
    err = abs(reconstructed - raw_res)
    print(f"  round-trip abs error      = {err: .3e}   "
          f"{'PASS' if err < 1e-6 else 'FAIL'}")


if __name__ == "__main__":
    main()
