import pathlib
"""
Step 1.2-1.3: SWCC (van Genuchten) + elastic parameters for Isikdere strata.
Source data: Ulusay et al. (2014), Eng. Geol. 181, 261-280, Tables 2 & site GSI/UCS.
"""
import json, math
import numpy as np
from rosetta import rosetta, SoilData, SoilData

G = 9.81  # m/s^2, consistent with Step 1.1 sheet

# ---------------------------------------------------------------- Step 1.2
# Dry bulk density from bulk unit weight and water content:
#   gamma_d = gamma / (1 + w);  rho_d = gamma_d * 1000 / g  -> g/cm^3
def dry_bd(gamma_kNm3, w_frac):
    gd = gamma_kNm3 / (1.0 + w_frac)
    return gd * 1000.0 / G / 1000.0  # g/cm^3

soils = {
    "Underclay (Turgut Fm.)": dict(sand=22.5, silt=46.0, clay=31.5,
                                   gamma=18.4, w=0.215, Gs=2.50),
    "Slope debris":           dict(sand=3.5,  silt=53.5, clay=43.0,
                                   gamma=17.6, w=0.332, Gs=2.65),  # Gs assumed
}
gouge = dict(sand=20.0, silt=80.0, clay=0.0)  # fault gouge, 1 specimen, SSC only

print("=" * 78)
print("STEP 1.2 - ROSETTA3 (Zhang & Schaap 2017) PEDOTRANSFER RUNS")
print("=" * 78)

def run_rosetta(rows, version=3):
    """rows: [sand, silt, clay, (bd)]; estimate_type='log' -> cols 2-5 are log10 means.
    Cols: 0 theta_r, 1 theta_s, 2 log10 alpha(1/cm), 3 log10 n, 4 log10 Ksat(cm/day)"""
    mean, stdev, codes = rosetta(version, SoilData.from_array(rows))
    return mean, stdev, codes

results = {}
for name, p in soils.items():
    bd_dry = dry_bd(p["gamma"], p["w"])
    bd_moist = p["gamma"] * 1000.0 / G / 1000.0
    # Adopted run: SSC + DRY bulk density (Rosetta was trained on oven-dry BD)
    m, s, c = run_rosetta([[p["sand"], p["silt"], p["clay"], bd_dry]])
    # Sensitivity run: SSC + moist BD (what the roadmap draft implicitly used)
    m2, _, _ = run_rosetta([[p["sand"], p["silt"], p["clay"], bd_moist]])
    # Texture-only run (model 2) for comparison
    m3, _, _ = run_rosetta([[p["sand"], p["silt"], p["clay"]]])

    def unpack(mm):
        tr, ts, la, ln_, lk = mm[0][:5]
        return dict(theta_r=tr, theta_s=ts,
                    alpha_cm=10**la, alpha_m=10**la * 100.0,
                    n=10**ln_, m_vg=1 - 1/10**ln_,
                    Ksat_cmday=10**lk, Ksat_ms=10**lk / 100.0 / 86400.0)

    r = unpack(m)
    r_sd = dict(theta_r_sd=s[0][0], theta_s_sd=s[0][1],
                log10_alpha_sd=s[0][2], log10_n_sd=s[0][3], log10_Ksat_sd=s[0][4])
    porosity = 1.0 - dry_bd(p["gamma"], p["w"]) / p["Gs"]
    results[name] = dict(inputs=dict(**p, bd_dry=round(bd_dry, 3),
                                     bd_moist=round(bd_moist, 3),
                                     model_code=int(c[0])),
                         adopted=r, stdev=r_sd,
                         moistBD=unpack(m2), textureOnly=unpack(m3),
                         porosity_n0=porosity)

    print(f"\n--- {name} ---")
    print(f"  SSC = {p['sand']}/{p['silt']}/{p['clay']} %   "
          f"BD_dry = {bd_dry:.3f} g/cm3 (moist {bd_moist:.3f})   Rosetta model {int(c[0])}")
    print(f"  ADOPTED (SSC+BD_dry): th_r={r['theta_r']:.3f}  th_s={r['theta_s']:.3f}  "
          f"alpha={r['alpha_m']:.3f} 1/m  n={r['n']:.3f}  m={r['m_vg']:.3f}")
    print(f"    Ksat = {r['Ksat_cmday']:.2f} cm/day = {r['Ksat_ms']:.2e} m/s")
    print(f"    1-sd (log10 space): alpha {r_sd['log10_alpha_sd']:.3f}, n {r_sd['log10_n_sd']:.3f}, "
          f"Ksat {r_sd['log10_Ksat_sd']:.3f}; th_r {r_sd['theta_r_sd']:.3f}, th_s {r_sd['theta_s_sd']:.3f}")
    m2u, m3u = results[name]["moistBD"], results[name]["textureOnly"]
    print(f"  sens. moist-BD:  th_s={m2u['theta_s']:.3f} alpha={m2u['alpha_m']:.3f} n={m2u['n']:.3f}")
    print(f"  sens. SSC-only:  th_s={m3u['theta_s']:.3f} alpha={m3u['alpha_m']:.3f} n={m3u['n']:.3f}")
    print(f"  Phase-relation porosity n0 = {porosity:.3f}  (theta_s cross-check)")

# Fault gouge (supplementary; texture only, clay not reported -> 0)
mg, sg, cg = run_rosetta([[gouge["sand"], gouge["silt"], gouge["clay"]]])
tr, ts, la, ln_, lk = mg[0][:5]
print(f"\n--- Fault gouge (supplementary, model {int(cg[0])}, SSC only, clay taken as 0) ---")
print(f"  th_r={tr:.3f} th_s={ts:.3f} alpha={10**la*100:.3f} 1/m n={10**ln_:.3f} "
      f"Ksat={10**lk:.2f} cm/day = {10**lk/100/86400:.2e} m/s")
results["Fault gouge (suppl.)"] = dict(
    inputs=dict(**gouge, model_code=int(cg[0])),
    adopted=dict(theta_r=tr, theta_s=ts, alpha_m=10**la*100, n=10**ln_,
                 m_vg=1-1/10**ln_, Ksat_ms=10**lk/100/86400))

# Rock strata: porosity-tied theta_s (theta_s ~ 0.9 * n0, entrapped-air allowance)
print("\n" + "=" * 78)
print("ROCK-STRATA POROSITY -> theta_s TIE-IN (bulk unit wt ~ dry for rock units)")
print("=" * 78)
rock_poro = {}
for name, gam, Gs in [("Marl (Sekkoy Fm.)", 14.9, 2.65),
                      ("Weak-zone marl",    16.9, 2.65),
                      ("Tuffite (Yatagan Fm.)", 14.4, 2.50)]:
    rho_d = gam * 1000.0 / G / 1000.0
    n0 = 1.0 - rho_d / Gs
    rock_poro[name] = dict(rho_d=rho_d, Gs=Gs, n0=n0, theta_s_09=0.9*n0)
    print(f"  {name:24s} rho_d={rho_d:.3f} g/cm3  Gs={Gs} (assumed)  "
          f"n0={n0:.3f}  0.9*n0={0.9*n0:.3f}")

# ---------------------------------------------------------------- Step 1.3
print("\n" + "=" * 78)
print("STEP 1.3 - HOEK-DIEDERICHS (2006) ROCK-MASS MODULUS, D = 1 (production blasting)")
print("=" * 78)

def hd_full(Ei, GSI, D=1.0):
    return Ei * (0.02 + (1 - D/2) / (1 + math.exp((60 + 15*D - GSI) / 11)))

def hd_simplified(GSI, D=1.0):
    return 1e5 * ((1 - D/2) / (1 + math.exp((75 + 25*D - GSI) / 11)))  # MPa

def vasarhelyi(GSI, mi):
    return -0.002*GSI - 0.003*mi + 0.457

rocks = {  # UCS in MPa; MR (lo, adopted, hi); nu adopted (Gercek-2007-consistent)
    "Marl (Sekkoy Fm.)":     dict(UCS=17.9, GSI=50, mi=4,  MR=(150, 175, 200), nu=0.28),
    "Weak-zone marl":        dict(UCS=4.29, GSI=45, mi=3,  MR=(150, 175, 200), nu=0.30),
    "Coal seam (lignite)":   dict(UCS=7.80, GSI=40, mi=18, MR=(200, 300, 400), nu=0.32),
    "Tuffite (Yatagan Fm.)": dict(UCS=3.59, GSI=26, mi=15, MR=(200, 350, 400), nu=0.33),
}
elastic = {}
for name, p in rocks.items():
    lo, ad, hi = p["MR"]
    Ei = ad * p["UCS"]
    Erm = hd_full(Ei, p["GSI"])
    Erm_lo, Erm_hi = hd_full(lo*p["UCS"], p["GSI"]), hd_full(hi*p["UCS"], p["GSI"])
    Erm_simp = hd_simplified(p["GSI"])
    factor = Erm / Ei
    nu_v = vasarhelyi(p["GSI"], p["mi"])
    nu = p["nu"]
    E_Pa = Erm * 1e6
    Gmod = E_Pa / (2*(1+nu))
    lam = E_Pa * nu / ((1+nu)*(1-2*nu))
    elastic[name] = dict(UCS=p["UCS"], GSI=p["GSI"], mi=p["mi"], D=1,
                         MR_lo=lo, MR_adopted=ad, MR_hi=hi,
                         Ei_MPa=Ei, HD_factor=factor,
                         Erm_MPa=Erm, Erm_lo_MPa=Erm_lo, Erm_hi_MPa=Erm_hi,
                         Erm_simplified_MPa=Erm_simp,
                         nu_adopted=nu, nu_vasarhelyi=nu_v,
                         E_Pa=E_Pa, G_Pa=Gmod, lambda_Pa=lam)
    print(f"\n--- {name} ---  GSI={p['GSI']}, UCS={p['UCS']} MPa, mi={p['mi']}, D=1")
    print(f"  MR = {lo}-{hi} (adopted {ad})  ->  E_i = {Ei:,.0f} MPa")
    print(f"  HD reduction factor = {factor:.4f}")
    print(f"  E_rm = {Erm:,.1f} MPa   [MR range: {Erm_lo:,.1f} - {Erm_hi:,.1f} MPa]")
    print(f"  simplified-HD cross-check (ignores UCS): {Erm_simp:,.0f} MPa")
    print(f"  nu adopted = {nu:.2f}   (Vasarhelyi-2009 rock-mass estimate: {nu_v:.3f})")
    print(f"  G = {Gmod:.3e} Pa   lambda = {lam:.3e} Pa")

# Soil strata (drained parameters; literature-typical, weakest-constrained values)
soils_E = {
    "Underclay (Turgut Fm.)": dict(E_MPa=15.0, E_lo=10, E_hi=25, nu=0.35),
    "Slope debris":           dict(E_MPa=8.0,  E_lo=5,  E_hi=15, nu=0.35),
}
print("\n--- SOIL STRATA (drained E', nu'; literature-typical, flag for sensitivity) ---")
for name, p in soils_E.items():
    E_Pa = p["E_MPa"]*1e6; nu = p["nu"]
    Gm = E_Pa/(2*(1+nu)); lam = E_Pa*nu/((1+nu)*(1-2*nu))
    elastic[name] = dict(Erm_MPa=p["E_MPa"], Erm_lo_MPa=p["E_lo"], Erm_hi_MPa=p["E_hi"],
                         nu_adopted=nu, E_Pa=E_Pa, G_Pa=Gm, lambda_Pa=lam)
    print(f"  {name:24s} E' = {p['E_MPa']} MPa ({p['E_lo']}-{p['E_hi']})  nu' = {nu}"
          f"   G = {Gm:.2e} Pa  lambda = {lam:.2e} Pa")

json.dump(dict(step12=results, rock_porosity=rock_poro, step13=elastic),
          open(pathlib.Path(__file__).resolve().parent.parent / "data" / "derived_raw.json", "w"), indent=1, default=float)
print("\nSaved ../data/derived_raw.json")
