r"""Evidence pack for the Phase 5 supervisor decisions. MEASURES OPTIONS; DOES
NOT PICK ONE.

    python scripts/decision_evidence.py runs/ansatz/exp_seed7/ckpt_final.pt

Every number is a POINTWISE check of the frozen elastic baseline stress
against a strength envelope -- the fraction of the slope whose stress already
lies outside the criterion. It is NOT a factor of safety, and no training is
done. It answers "does this option give the SSR sweep a sane starting state,
and how fast does pointwise yielding spread with SRF?", which is what the
decision needs before GPU time is spent.

Stress is the quantity `loss.L_yield` checks: total stress sigma_0 + D:eps
with the Bishop increment, principal values compression-positive, on
`sample_interior` points with their random t in [0, 30] d, area-weighted by
coll.w. Admissible means `plasticity.yield_violation <= 1e-9`, as in L_yield.

Options measured:
  D2  MC strength: bedding residual pair (current) vs Hoek (2002) rock-mass
      MC equivalents at several sig3max
  D3  GHB fit range: reduced GHB (Hammah et al. via strength.reduce_ghb) at
      SRF 1.25 and 1.5 for several ranges
  D4  disturbance D: GHB strength re-derived at D = 1, 0.7, 0 (strength only;
      E_rm and sigma_0 are NOT re-derived -- see output note)
  D5  admissible metric: area, min_tag, tag:Mk_d at each SRF
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from scripts.ssr_sweep import MC_DESIGN, _admissible_metric
from src import plasticity as pl
from src import sensitivity as sens
from src import strength as st
from src.config import BOUNDS, tiny
from src.materials import load_materials
from src.mechanics import total_stress_star
from src.model import PINN, NearPhysical
from src.nondim import SCALES
from src.properties import UNIT_MAP
from src.sampling import sample_interior
from src.sigma0 import attach_sigma0

import dataclasses


def hoek2002_mc(p: st.GHBParams, sig3max: float) -> st.MCParams:
    """Hoek, Carranza-Torres & Corkum (2002) equivalent c, phi over
    0 <= sig3 <= sig3max. `p` already carries D through m_b and s."""
    s3n = sig3max / p.sigma_ci
    a, mb, s = p.a, p.m_b, p.s
    k = 6.0 * a * mb * (s + mb * s3n) ** (a - 1.0)
    d = (1.0 + a) * (2.0 + a)
    phi = math.asin(k / (2.0 * d + k))
    c = (p.sigma_ci * ((1.0 + 2.0 * a) * s + (1.0 - a) * mb * s3n)
         * (s + mb * s3n) ** (a - 1.0) / (d * math.sqrt(1.0 + k / d)))
    return st.MCParams(c=c, phi=phi)


def admissible_by_tag(sig, tags, params_by_tag, criterion) -> dict:
    """Fraction admissible per tag. `sig` maps tag -> (sig1, sig3) numpy, Pa."""
    out = {}
    for t in tags:
        s1, s3 = sig[t]
        v = pl.yield_violation(s1, s3, params_by_tag[t], criterion,
                               sigma_scale=SCALES.sig_ref)
        out[t] = float(np.mean(np.asarray(v) <= 1e-9))
    return out


def metrics(adm, shares) -> dict:
    return {m: _admissible_metric(adm, shares, m)
            for m in ("area", "min_tag", "tag:Mk_d")}


def ghb_at_D(tag, D):
    b = sens.BASELINE[UNIT_MAP.get(tag, tag)]
    return st.ghb_from_gsi(sigma_ci=b["sigma_ci"], gsi=b["gsi"],
                           m_i=b["m_i"], D=D)


def baseline_stress(ckpt, n, seed):
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = d["cfg"]
    pc = dataclasses.replace(tiny(), n_layers=cfg["layers"],
                             n_neurons=cfg["width"], device="cpu")
    net = NearPhysical(PINN(pc, BOUNDS), eps_psi=cfg["eps_psi"],
                       mode=cfg["ansatz"], cap_k=cfg.get("cap_k", 100.0))
    net.load_state_dict(d["net"])
    net.eval()
    coll = attach_sigma0(sample_interior(n, seed))
    mats = load_materials()
    sig, shares = {}, {}
    w = coll.w.detach().numpy().ravel()
    for tag in sorted(set(coll.tag.tolist())):
        m = torch.as_tensor(coll.tag == tag)
        x = coll.x[m].reshape(-1, 1).clone().requires_grad_(True)
        z = coll.z[m].reshape(-1, 1).clone().requires_grad_(True)
        t = coll.t[m].reshape(-1, 1).clone().requires_grad_(True)
        s0 = coll.sigma0[m]
        (sxx, szz, sxz), _, _ = total_stress_star(
            net, x, z, t, mats[tag],
            sigma0_star=(s0[:, 0:1], s0[:, 1:2], s0[:, 2:3]))
        s1, s3 = st.principal_stresses_from_cartesian(
            sxx.detach(), szz.detach(), sxz.detach())
        sig[tag] = (s1.numpy().ravel() * SCALES.sig_ref,
                    s3.numpy().ravel() * SCALES.sig_ref)
        shares[tag] = float(w[coll.tag == tag].sum() / w.sum())
    return sig, shares, mats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--sampling-seed", type=int, default=20260808)
    ap.add_argument("--json", default="docs/results/decision_evidence.json")
    a = ap.parse_args(argv)

    sig, shares, mats = baseline_stress(a.ckpt, a.n, a.sampling_seed)
    tags = sorted(sig)
    out = {"ckpt": a.ckpt, "n": a.n, "area_shares": shares,
           "note": "pointwise admissibility of the frozen elastic baseline; "
                   "not a factor of safety"}
    ghb0 = {t: st.GHBParams(sigma_ci=mats[t].sigma_ci, m_b=mats[t].m_b,
                            s=mats[t].s, a=mats[t].a) for t in tags}

    def show(title, rows):
        print(f"\n{title}")
        print(f"  {'option':<34}{'Mk':>7}{'Mk_d':>7}{'Tm':>7}"
              f"{'area':>8}{'min':>7}")
        for name, adm in rows:
            mt = metrics(adm, shares)
            print(f"  {name:<34}{adm['Mk']:>7.3f}{adm['Mk_d']:>7.3f}"
                  f"{adm['Tm']:>7.3f}{mt['area']:>8.3f}{mt['min_tag']:>7.3f}")

    # D2 -- MC strength at SRF 1 and 1.25
    d2 = []
    for srf in (1.0, 1.25):
        red = st.reduce_mc(MC_DESIGN, srf)
        d2.append((f"bedding pair, SRF {srf}",
                   admissible_by_tag(sig, tags, {t: red for t in tags}, "MC")))
        for s3m in (1.0e5, 4.0e5, 8.5e5):
            eq = {t: st.reduce_mc(hoek2002_mc(ghb0[t], s3m), srf) for t in tags}
            d2.append((f"rock-mass MC s3max {s3m/1e3:.0f}k, SRF {srf}",
                       admissible_by_tag(sig, tags, eq, "MC")))
    show("D2  MC strength options (admissible fraction)", d2)
    out["D2_mc"] = [{"option": n, "admissible_by_tag": adm,
                     **metrics(adm, shares)} for n, adm in d2]
    out["D2_mc_equivalents"] = {
        f"{s3m:.0f}": {t: {"c_Pa": hoek2002_mc(ghb0[t], s3m).c,
                           "phi_deg": math.degrees(hoek2002_mc(ghb0[t], s3m).phi)}
                       for t in tags} for s3m in (1.0e5, 4.0e5, 8.5e5)}

    # D3 -- GHB fit range
    d3 = [("GHB unreduced, SRF 1", admissible_by_tag(sig, tags, ghb0, "GHB"))]
    for srf in (1.25, 1.5):
        for rng in ((0.0, 1.31e5), (0.0, 1.79e5), (0.0, 4.0e5), (0.0, 8.5e5),
                    (0.0, 1.25e6)):
            red = {t: st.reduce_ghb(ghb0[t], srf, sig3_range=rng) for t in tags}
            d3.append((f"range 0-{rng[1]/1e3:.0f}k, SRF {srf}",
                       admissible_by_tag(sig, tags, red, "GHB")))
    show("D3  GHB sig3 fit range (admissible fraction)", d3)
    out["D3_ghb_range"] = [{"option": n, "admissible_by_tag": adm,
                            **metrics(adm, shares)} for n, adm in d3]

    # D4 -- disturbance factor
    d4, d4_params = [], []
    for D in (1.0, 0.7, 0.0):
        p = {t: ghb_at_D(t, D) for t in tags}
        d4.append((f"D = {D}, SRF 1", admissible_by_tag(sig, tags, p, "GHB")))
        d4_params.append({t: dataclasses.asdict(p[t]) for t in tags})
    show("D4  disturbance factor, GHB at SRF 1 (strength only)", d4)
    out["D4_disturbance"] = [{"option": n, "admissible_by_tag": adm,
                              **metrics(adm, shares), "params": prm}
                             for (n, adm), prm in zip(d4, d4_params)]
    print("  NOTE: D also moves E_rm (Hoek-Diederichs) and therefore sigma_0; "
          "only the strength is changed here.")

    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
