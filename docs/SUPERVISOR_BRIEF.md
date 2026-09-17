# Phase 5 decisions — brief for the supervisor meeting

Abrar Mahir · PINN hydro-mechanical SSR, Işıkdere Section 5 · Day 41

**Outcome (Day 41):** recorded in `DECISIONS.md` as D-5.1 to D-5.5. Soft
constraint; MC-SSR by rock-mass equivalents or not at all (bedding pair
rejected); GHB fit range 0–179 kPa; D = 1 retained; admissible metric
`min_tag`. Open: (a) vs (c), `--yield-norm`, the drop threshold.

**What this is.** Five decisions block the factor-of-safety sweeps. The code
for all options exists and is tested (530+ tests). The sweeps will not run
until each is made. Every number below is a **pointwise** check of the frozen
elastic baseline (`baseline-v1`) against a strength envelope: the fraction of
each stratum whose stress already lies inside the criterion. **None is a
factor of safety.** Source: `scripts/decision_evidence.py`,
`docs/results/decision_evidence.json`.

Area shares: Mk 3.0% · Mk_d 9.8% · Tm 87.2%.

---

## D1 — Soft constraint or return mapping?

| | Soft constraint (implemented) | Return mapping |
|---|---|---|
| What it is | penalty `w_yield · ‖max(f,0)‖²` in the loss | stress projected onto the yield surface with a flow rule |
| FOS depends on | `w_yield` — must show a plateau over 0.3/1/3 | the constitutive model only |
| Dilatancy, associated vs non-associated | not representable | representable |
| Cost | ready now | weeks of implementation and testing |

**Decision:** ______________________

---

## D2 — Which strength for MC-SSR?

The Phase-1 plan reduces the bedding-plane **residual** pair
(c = 4.4 kPa, φ = 15.4°) in the MC sweep. The sweep applies it as a continuum
strength at every point. Admissible fraction on the baseline:

| option | Mk | Mk_d | Tm | area |
|---|---|---|---|---|
| bedding pair, SRF 1.0 | 0.000 | 0.069 | 0.019 | 0.023 |
| rock-mass MC, σ₃max 100 kPa, SRF 1.0 | 0.999 | 0.424 | 1.000 | 0.944 |
| rock-mass MC, σ₃max 400 kPa, SRF 1.0 | 0.999 | 0.533 | 1.000 | 0.954 |
| rock-mass MC, σ₃max 850 kPa, SRF 1.0 | 0.999 | 0.655 | 1.000 | 0.966 |
| *for comparison: GHB, SRF 1.0* | 0.999 | 0.389 | 1.000 | 0.940 |

With the bedding pair, 98% of the slope is outside the envelope before any
reduction, including the limestone basement. A CPU smoke sweep confirmed that
admissibility then **rises** with SRF as the penalty reshapes the field, so no
failure can be detected. Rock-mass equivalents (Hoek et al. 2002, reproducing
the Phase-1 stored values to 0.03%) start from a state comparable to GHB.

Options:
- (a) rock-mass MC equivalents: a like-for-like MC vs GHB comparison; state σ₃max
- (b) bedding pair on a bedding interface/zone only: needs new code (a joint or ubiquitous-joint model)
- (c) drop MC-SSR; compare GHB-PINN against the LEM instead
- (d) keep the bedding pair as a continuum: not recommended by the evidence above

**Decision:** ______________________

---

## D3 — GHB σ₃ fit range

Baseline σ₃ (compression +, area-weighted): Mk_d p50 51 kPa, p95 131 kPa,
p99 179 kPa; Tm p50 561 kPa. The old default (850 kPa) is about 6× the marl
p95.

Mk_d admissible fraction after GHB reduction:

| fit range | SRF 1.25 | SRF 1.5 |
|---|---|---|
| 0–131 kPa | 0.192 | 0.117 |
| 0–850 kPa | 0.190 | 0.117 |
| 0–1250 kPa | 0.190 | 0.117 |

Pointwise, the range hardly matters. The reduced envelopes agree to about
3 kPa over Mk_d's stress range. They differ mainly at σ₃ ≈ 0
(27.5 vs 24.4 kPa), which affects mechanisms exiting near the surface. The
range must still be reported with the FOS.

**Decision:** σ₃ lo = ______ Pa, hi = ______ Pa, basis: ______________

---

## D4 — Disturbance factor D = 1 for all strata

All strata are stored at D = 1, the value Hoek gives for heavy production
blasting. Admissible fraction at SRF 1 (strength re-derived; E_rm and σ₀ not):

| D | Mk | Mk_d | Tm | area |
|---|---|---|---|---|
| 1.0 (current) | 0.999 | 0.389 | 1.000 | 0.940 |
| 0.7 | 0.999 | 0.906 | 1.000 | 0.991 |
| 0.0 | 1.000 | 0.989 | 1.000 | 0.999 |

D controls whether 61% or 9% of Mk_d is already overstressed at SRF = 1. It
is the largest single lever found. Changing it also changes E_rm and would
require re-solving σ₀ and retraining the baseline.

**Decision:** ______________________

---

## D5 — How is "admissible fraction dropped" measured?

The failure test needs loss plateau **and** displacement jump **and** a drop
in admissibility relative to SRF = 1. At GHB, D = 1, pointwise:

| metric | SRF 1 | SRF 1.25 | SRF 1.5 | max drop if only marls yield |
|---|---|---|---|---|
| area | 0.940 | 0.919 | 0.907 | **0.068** |
| min over strata | 0.389 | 0.192 | 0.117 | 0.389 |
| Mk_d only | 0.389 | 0.192 | 0.117 | 0.389 |

Under `area`, Tm's 87% share caps the signal: any threshold ≥ 0.07 can never
fire for a marl mechanism. `min_tag` and `tag:Mk_d` carry the signal. The
latter builds an expectation about where failure occurs into the detector.

**Decision:** metric ______ drop ______

---

## Also for the record

- The rainfall arm of the sensitivity study cannot change the result:
  10/20/40 mm/hr give bit-identical boundary conditions (all rainfall surfaces
  are Mk_d at K_s = 1e-9 m/s, capped). Drop it or change the infiltration mode.
- The plan's "one-way overestimates FOS by ~21.5%" predates any result and
  should be struck; the coupling arm reports a signed number.
- `baseline-v1` is a 2,000-epoch run; the fixed base moves ~1.4 mm by
  t = 30 d. A production baseline may be needed before headline numbers.
- LEM reference for Section 5 is F = 0.94 (residual bedding strength), not
  1.25.
