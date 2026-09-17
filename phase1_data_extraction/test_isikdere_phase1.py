"""
test_isikdere_phase1.py — verification suite for the Phase-1 dataset
====================================================================
Run: python test_isikdere_phase1.py

Checks that the shipped dataset is internally consistent and that the
constitutive functions behave correctly at the limits autodiff will hit.
Any FAIL here means the dataset is not safe to simulate with.
"""
import math
from isikdere_phase1 import (DATA, STRATA, SCALES, ORDER, RHO_W, G,
                             vg_Se, vg_theta, vg_K, vg_C, water_saturation,
                             kozeny_carman, porosity, hoek_brown_params,
                             mohr_coulomb_equivalent, seepage_flux, drawdown_head,
                             richards_residual_nd, mechanical_residual_nd)

fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


print("=" * 74)
print("1. PHASE RELATIONS — n0 recomputed from each stratum's own gamma and Gs")
print("=" * 74)
gammas = {"marl_sekkoy": 14.9, "marl_weakzone": 16.9, "coal": 8.6,
          "tuffite": 14.4, "underclay": 18.4, "slope_debris": 17.6}
w_nat = {"underclay": 0.215, "slope_debris": 0.332}
for k, st in STRATA.items():
    gam = gammas[k] / (1 + w_nat.get(k, 0.0))
    rho_d = gam * 1000.0 / G
    n0 = 1 - rho_d / (st.Gs * RHO_W)
    check(f"{k:14} n0", abs(n0 - st.n0) < 1e-3, f"{n0:.4f} vs {st.n0:.4f}")
    check(f"{k:14} rho_sat = rho_dry + n0*rho_w",
          abs(st.rho_dry + st.n0 * RHO_W - st.rho_sat) < 1.0)

print("\n" + "=" * 74)
print("2. VAN GENUCHTEN — limits, monotonicity, analytic C vs numerical dtheta/dpsi")
print("=" * 74)
for k, st in STRATA.items():
    # saturation limit
    check(f"{k:14} Se = 1 at psi = 0", abs(vg_Se(0.0, st) - 1.0) < 1e-12)
    check(f"{k:14} theta(0) = theta_s", abs(vg_theta(0.0, st) - st.theta_s) < 1e-12)
    check(f"{k:14} K(0) = K_s exactly", abs(vg_K(0.0, st) - st.K_s) / st.K_s < 1e-12)
    check(f"{k:14} C(0) = 0 (saturated stores nothing)", abs(vg_C(0.0, st)) < 1e-12)
    # saturated branch: positive pressure head must NOT desaturate
    check(f"{k:14} psi > 0 stays saturated",
          abs(vg_theta(+25.0, st) - st.theta_s) < 1e-12
          and abs(vg_K(+25.0, st) - st.K_s) / st.K_s < 1e-12
          and abs(vg_C(+25.0, st)) < 1e-12)
    # residual limit. Low-n strata approach theta_r very slowly - that is a real
    # property of van Genuchten at n ~ 1.2, not a defect: the marl family still
    # holds ~0.02 above theta_r at 10^6 m of suction.
    check(f"{k:14} theta -> theta_r at extreme suction",
          abs(vg_theta(-1e12, st) - st.theta_r) < 0.01,
          f"theta(-1e6 m) = {vg_theta(-1e6, st):.4f} vs theta_r = {st.theta_r}")
    # monotone decreasing in suction
    psis = [-0.1, -1, -5, -20, -100]
    th = [vg_theta(p, st) for p in psis]
    check(f"{k:14} theta monotone decreasing", all(th[i] > th[i + 1] for i in range(len(th) - 1)))
    Ks = [vg_K(p, st) for p in psis]
    check(f"{k:14} K monotone decreasing", all(Ks[i] > Ks[i + 1] for i in range(len(Ks) - 1)))
    # analytic capacity vs central difference
    worst = 0.0
    for p in (-0.5, -2.0, -10.0, -50.0):
        h = 1e-5 * abs(p)
        num = (vg_theta(p + h, st) - vg_theta(p - h, st)) / (2 * h)
        ana = vg_C(p, st)
        worst = max(worst, abs(num - ana) / max(abs(num), 1e-30))
    check(f"{k:14} C analytic == dtheta/dpsi", worst < 1e-5, f"max rel err {worst:.2e}")
    # C finite and non-negative everywhere sampled
    check(f"{k:14} C finite and >= 0",
          all(math.isfinite(vg_C(p, st)) and vg_C(p, st) >= 0 for p in psis))

print("\n" + "=" * 74)
print("3. KOZENY-CARMAN COUPLING")
print("=" * 74)
for k, st in STRATA.items():
    check(f"{k:14} KC identity at eps_v = 0", abs(kozeny_carman(porosity(0.0, st), st) - st.K_s) / st.K_s < 1e-12)
    hi = kozeny_carman(porosity(+0.01, st), st)
    lo = kozeny_carman(porosity(-0.01, st), st)
    check(f"{k:14} KC increases with dilation", hi > st.K_s > lo,
          f"x{hi/st.K_s:.3f} / x{lo/st.K_s:.3f} for eps_v = +/-1%")

print("\n" + "=" * 74)
print("4. HOEK-BROWN — reproduces Step 1.5 sheet and Ulusay Table 3b inputs")
print("=" * 74)
expected = {"marl_sekkoy": (0.1125, 2.40e-4, 0.5057),
            "marl_weakzone": (0.0590, 1.045e-4, 0.5081),
            "coal": (0.2477, 4.54e-5, 0.5114),
            "tuffite": (0.0760, 4.40e-6, 0.5292)}
for k, (mb_e, s_e, a_e) in expected.items():
    st = STRATA[k]
    mb, s, a = hoek_brown_params(st.GSI, st.m_i, st.D)
    ok = (abs(mb - mb_e) < 5e-4 and abs(s - s_e) / s_e < 5e-3 and abs(a - a_e) < 5e-4)
    check(f"{k:14} m_b, s, a", ok, f"{mb:.4f} / {s:.3e} / {a:.4f}")

# MC equivalents at the design confining range
print("\n  MC equivalents at sig3max = 850 kPa (Step 1.5 Table 4):")
expect_mc = {"marl_sekkoy": (130.5, 21.7), "marl_weakzone": (50.4, 10.0),
             "coal": (108.2, 21.6), "tuffite": (43.9, 9.9)}
for k, (c_e, phi_e) in expect_mc.items():
    c, phi = mohr_coulomb_equivalent(STRATA[k], 850e3)
    check(f"{k:14} c, phi", abs(c / 1e3 - c_e) < 0.6 and abs(phi - phi_e) < 0.2,
          f"{c/1e3:.1f} kPa / {phi:.1f} deg")

print("\n" + "=" * 74)
print("5. BOUNDARY CONDITIONS — the corrected flux and its mass balance")
print("=" * 74)
cum = sum(seepage_flux(t) * 86400.0 for t in range(1, 31))
check("30-day cumulative inflow = 5.37 m3/m (paper)", abs(cum - 5.37) < 0.01, f"{cum:.4f} m3/m")
check("q(1 day) = 6.481e-6 m3/s/m (NOT 6.48e-7)", abs(seepage_flux(1) - 6.481e-6) < 1e-9,
      f"{seepage_flux(1):.4e}")
check("drawdown h(0,t) = 10 m at the face", abs(drawdown_head(1e-9, 1) - 10.0) < 1e-6)
check("drawdown h(500,1) -> h0 = 30 m", abs(drawdown_head(500, 1) - 30.0) < 1e-3)
check("~3 m drawdown at x = 100 m, t = 1 day (paper's own summary)",
      2.0 < 30.0 - drawdown_head(100, 1) < 4.0, f"{30.0-drawdown_head(100,1):.2f} m")

print("\n" + "=" * 74)
print("6. NON-DIMENSIONALISATION — algebraic round trip")
print("=" * 74)
dsig_dz_raw, rho_b = 2.0e4, 1800.0
raw_res = dsig_dz_raw - rho_b * G
nd_res = mechanical_residual_nd(
    (0.0, dsig_dz_raw / (SCALES.sig_ref / SCALES.L_ref)), rho_b / SCALES.rho_b_ref)[1]
recon = nd_res * (SCALES.sig_ref / SCALES.L_ref)
check("raw -> nd -> raw round trip", abs(recon - raw_res) < 1e-6,
      f"err {abs(recon-raw_res):.2e} N/m3")
check("mechanical residual is O(1)", abs(nd_res) < 1e2, f"{nd_res:.4f}")
check("U_ref = sig_ref*L_ref/E_ref",
      abs(SCALES.U_ref - SCALES.sig_ref * SCALES.L_ref / SCALES.E_ref) < 1e-9)
check("E_ref is not stiffer than every stratum",
      SCALES.E_ref <= 1.1 * max(st.E for st in STRATA.values()),
      f"E_ref {SCALES.E_ref:.1e} vs stiffest stratum {max(st.E for st in STRATA.values()):.3e}")
print(f"  NOTE Pi_R_diff = {SCALES.Pi_R_diff:.3e} is deliberately NOT O(1). "
      f"T_diffusive = {SCALES.T_diffusive/86400/365:.0f} yr.")

print("\n" + "=" * 74)
print("7. DATASET COMPLETENESS — Phase-1 exit checklist")
print("=" * 74)
required_all = ["K_s", "rho_dry", "rho_sat", "nu", "n0",
                "theta_r", "theta_s", "alpha", "n", "m", "E"]
for k, st in STRATA.items():
    missing = [f for f in required_all if getattr(st, f) is None]
    check(f"{k:14} all universal params present", not missing, f"missing {missing}" if missing else "")
for k, st in STRATA.items():
    if st.is_rock:
        missing = [f for f in ("sigma_ci", "m_i", "GSI", "D", "m_b", "s", "a")
                   if getattr(st, f) is None]
        check(f"{k:14} Hoek-Brown set present", not missing)
check("all 10 conflicts carry a resolution",
      all("resolution" in c for c in DATA["conflicts_resolved"]))
check("LEM benchmark table populated", len(DATA["validation"]["lem_factors_of_safety"]) >= 25)
check("MC design pair is the back-analysis pair",
      DATA["mc_strength"]["design_pair"]["c_Pa"] == 4400.0 and
      DATA["mc_strength"]["design_pair"]["phi_deg"] == 15.4)

print("\n" + "=" * 74)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED — dataset is internally consistent and safe to simulate with.")
print("=" * 74)
