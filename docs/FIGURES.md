# Thesis figures: style, sources and caption obligations

One command regenerates every figure (lab PC, repo root, after `go.bat`):

    scripts\make_thesis_figs.bat

Each figure is written to `docs\figs\` as a 300 dpi PNG and a vector PDF with
the same name. Insert the PDF into Word or LaTeX at 100%. The figures are
drawn at the final 6.25 in width, so do not rescale them. Rescaling is what
makes 12 pt text end up at a different size from the body text.

## House style (`src/plot_style.py`)

| rule | value |
|---|---|
| font | Times New Roman (fallback: TeX Gyre Termes / Liberation Serif), mathtext STIX |
| size | 12 pt for every text element |
| width | 6.25 in after the tight crop (`ps.save` resizes the figure to hit it and warns if it cannot) |
| export | PNG 300 dpi + PDF, `bbox_inches="tight"`, `pad_inches=0.05` |
| spines / ticks | 0.9 pt, ticks inward on all four sides |
| grid | `#E5E7EB`, dashed, alpha 0.5; off on field plots |
| criterion colour | GHB cobalt `#1D4ED8`, MC vermilion `#E4572E` (every figure) |
| strata colour | Mk amber, Mk_d purple, Tm teal (lines); light tints for fills |
| w_yield | marker shape, never colour |
| fields | ψ `viridis`; departures `RdBu_r` about 0; \|d\|, γ_max `turbo`; FS `RdYlBu` centred on 1 |

## Sources

| fig | script | data |
|---|---|---|
| 2 | `make_fig2.py` | baseline-v2 checkpoint cfg; refuses to write if any label overflows its box |
| 3 | `make_fig3.py` | live geometry, BC and sampler code |
| 4 | `make_fig4.py` | baseline-v2 `log.jsonl` |
| 5 | `make_fig5.py` | baseline-v2 checkpoint |
| 6 | `make_fig6.py` | baseline-v2 checkpoint (+ `--no-bishop` diagnostic) |
| 7 | `verify_nguyen_raudkivi.py` | retrains the verification PINN, R² 0.99931 |
| 8, 10, 11, 12, S2 | `make_results_figs.py` | `docs/results/final_results.json` + `*_sweep.jsonl`, **adopted criterion** (D-5.18/D-5.19) |
| 9 | `make_fig9.py` | `runs\ssr_ghb_w1`, `runs\ssr_mc_w1` states (lab PC only), `failed` |

`make_ssr_figs.py` still plots the **as-run** FOS from `runs\`, and those
disagree with FINAL_RESULTS for several runs (Mk_d GSI 35: 3.215 as run,
1.043 adopted). It is kept for checking a live sweep. Do not use its output
in the thesis.

Fig S1 has been folded into Fig 10(b) (O-5); the old file is in `_superseded/`.
Fig S2 is the new criterion-dependence figure (revision 1).

## What each caption must say

- **Fig 3**: the collocation cloud is stratified (D-Samp.1). Mk_d is 9.8% of
  the area but 35% of the points.
- **Fig 4**: the losses are unweighted and shown relative to step 0. The run is
  Adam only, 100k epochs. The interface term is frozen (D-W.4) and drops below
  the axis.
- **Fig 5**: the hydraulic field is essentially static over 30 d (O-21). The
  departure colour scale is clipped at p99.5, and the base strip saturates.
- **Fig 6**: SRF = 1, the unreduced baseline. σ₃ < 0 is clipped to 0 in FS,
  which credits tensile points with the rock-mass UCS (O-14). |d| is in µm and
  is limited by `eps_uv`.
- **Fig 8**: L is normalised by L(SRF 1) of the same run. Filled markers are
  the 0.25 grid and open markers are bisection points.
- **Fig 9**: it shows the `failed` state, with the reason (D-5.15), LI/LI₀ for
  each run, and the statement that a continuum without bedding planes cannot
  produce Ulusay et al.'s bedding-plane slides.
- **Figs 8–12, S2**: every FOS is calibrated (D-5.13). The error bars and bands
  are the GHB collocation-draw replicate SD (0.023, n = 5), and the same SD is
  applied to MC. Fig 10 has no LEM bar (O-15).
- **Fig 12**: the ranges come from data uncertainty, not a uniform ±20%. The
  GSI arms use the baseline σ₀ (O-11).
