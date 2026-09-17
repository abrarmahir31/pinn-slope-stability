# PINN Slope Stability — Isikdere Open-Pit Mine

Physics-Informed Neural Network (PINN) modelling of coupled hydro-mechanical
slope stability for the Isikdere open-pit mine (Ulusay et al. 2014, Eng.
Geol. 181, 261–280). The project builds a rainfall-infiltration /
geomechanics model of a real mine cross-section (Section 5), non-dimensionalises
it, and is working toward training a PINN that solves the coupled Richards
(seepage) and mechanical equilibrium equations simultaneously, subject to the
site's real boundary conditions, initial conditions, and material properties.

## Overview

Classical slope-stability analysis (limit equilibrium, FEM) solves seepage
and mechanics as separate, often decoupled, steps. This project instead poses
the coupled hydro-mechanical (HM) problem as a single PINN: a fully-connected
network that maps dimensionless space-time coordinates `(x*, z*, t*)` to the
dimensionless pressure head and displacement field `(psi*, u*, v*)`, trained
by minimising the PDE residuals of the Richards equation (unsaturated seepage)
and mechanical equilibrium (with Bishop's effective-stress coupling), together
with the site's boundary/initial condition losses.

Getting to a trainable network requires a substantial physics-setup pipeline
first — digitising the real geometry and material data from the source paper,
deriving constitutive parameters (Hoek-Brown, van Genuchten SWCC), choosing a
consistent non-dimensionalisation, and building/verifying an autograd-based
residual layer — before any training loop is meaningful. **That is the current
state of this repository: the physics foundation (Phases 1–4 below) is built
and verified; the PINN architecture and training loop (Phase 5) are the next
step, not yet implemented.**

## Prerequisites / Requirements

- Python 3.10 (see `environment.yml` for the conda environment named `pinn`)
- Core dependencies (installed via conda/pip from `environment.yml`):
  `numpy`, `scipy`, `matplotlib`, `pandas`, `pymupdf`, `pillow`, `pytest`,
  and `rosetta-soil==0.1.2` (pedotransfer functions for the SWCC derivation).
- **PyTorch is intentionally *not* pinned in `environment.yml`** because the
  two development machines need different builds:
  - Laptop (CPU only): `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  - Lab PC (CUDA): `pip install torch --index-url https://download.pytorch.org/whl/cu130`
  - Run `python tools/gpu_check.py` after installing to confirm which device
    PyTorch will use.
- A local copy of the source PDF (Ulusay et al. 2014) is required to
  regenerate the digitised figures in `phase3_geometry_bc_ic/`; it is not
  checked into the repository (copyrighted).

Setup:

```bash
conda env create -f environment.yml
conda activate pinn
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the cu130 URL on the lab PC
```

## Execution Guide

The pipeline is organised chronologically. Each phase's scripts are prefixed
with a run order where relevant; run them from the repository root unless a
script's own docstring says otherwise.

### Phase 1 — Data extraction (`phase1_data_extraction/`)
Extracts and structures the raw site data (stratigraphy, material properties)
from the source paper into a single JSON dataset.
```bash
python phase1_data_extraction/isikdere_phase1.py
pytest phase1_data_extraction/test_isikdere_phase1.py
```

### Phase 2 — Parameter derivation (`phase2_parameter_derivation/`)
Derives constitutive parameters from the Phase 1 data: Hoek-Brown/Mohr-Coulomb
strength, van Genuchten SWCC + elastic parameters (via Rosetta pedotransfer
functions), and validates the non-dimensionalisation scheme used everywhere
downstream.
```bash
python phase2_parameter_derivation/01_hoek_brown_mc.py
python phase2_parameter_derivation/02_swcc_elastic_params.py
python phase2_parameter_derivation/03_validate_nondim.py
```

### Phase 3 — Geometry, boundary & initial conditions (`phase3_geometry_bc_ic/`)
Digitises the mine cross-section geometry from the source figure, defines
material regions, and derives/verifies the boundary conditions (seepage +
mechanical) and initial conditions (hydrostatic pressure head, geostatic
stress). Scripts are numbered `00`–`16` in run order; each numbered `_checks`
script is a gate that must pass before moving to the next step.
```bash
cd phase3_geometry_bc_ic
python 00_setup_check.py
# ... 01 through 08: figure extraction / digitisation from the source PDF
python 09_checks.py        # geometry gate
python 11_bc_checks.py     # boundary-condition gate
python 13_mech_checks.py   # mechanical BC gate (exits non-zero on failure)
python 14_ic_checks.py     # initial-condition gate
python 16_ic_plot.py       # Fig. 3 — domain, contacts, collocation cloud, IC fields
```

### Phase 4 — Materials & PDE residuals (`phase4_materials_and_residuals/`)
The core physics library: the van Genuchten constitutive model (`vg.py`), the
per-unit material adapter (`materials.py`), the non-dimensionalisation
(`nondim.py`), the autograd differentiation layer (`derivatives.py`), the
Richards + mechanical residuals (`residuals.py`, `loss.py`), a collocation
sampler (`sampling.py`), and the PINN network definition (`model.py`,
`config.py`). This is a Python package — import it, don't run it directly.
```bash
pytest tests/ -v
```

### Phase 5 — Training & evaluation (not yet implemented)
The training loop and post-training evaluation/plotting scripts do not exist
yet. Once implemented, they belong in a new `phase5_training/` (or similar)
folder, building on the `PINN` model in `phase4_materials_and_residuals/model.py`
and the residuals in `phase4_materials_and_residuals/residuals.py`.

### Supporting directories
- `data/` — derived/reference datasets (CSV/JSON) produced by Phases 1–2.
- `docs/` — design decisions (`open_items.md` tracks open physics/numerical
  questions), verification logs, and reference documents (proposal,
  planning, master parameter sheets).
- `tools/` — environment/hardware checks (`verify_env.py`, `gpu_check.py`).
- `tests/` — pytest suite for the Phase 4 library (some tests require
  PyTorch; see Prerequisites).

## Project Directory Structure

```
pinn-slope-stability/
├── README.md
├── environment.yml
├── .githooks/
│   └── pre-commit                     # compiles all phase folders
├── data/                               # derived/reference CSV & JSON
├── docs/
│   ├── open_items.md                  # live log of open physics/numerical questions
│   ├── *_verification.txt             # residual/nondim verification logs
│   ├── *.docx                         # step-by-step parameter master sheets
│   ├── proposal/                      # thesis proposal
│   ├── planning/                      # roadmap, workflow docs
│   └── reference/                     # framework tables, slides, demo
├── phase1_data_extraction/
│   ├── isikdere_phase1.py             # extraction script
│   ├── isikdere_phase1_dataset.json   # structured site dataset (source of truth)
│   ├── isikdere_phase1_strata.csv
│   ├── inspect_dataset.py             # quick dataset inspector
│   ├── test_isikdere_phase1.py
│   └── PHASE1_README.md
├── phase2_parameter_derivation/
│   ├── 01_hoek_brown_mc.py            # Hoek-Brown -> Mohr-Coulomb strength
│   ├── 02_swcc_elastic_params.py      # van Genuchten SWCC + elastic params
│   ├── 03_validate_nondim.py          # non-dimensionalisation sanity checks
│   └── paths.py                       # shared path helper
├── phase3_geometry_bc_ic/
│   ├── 00_setup_check.py .. 08_split.py       # PDF figure extraction/digitisation
│   ├── 09_checks.py .. 16_ic_plot.py          # geometry / BC / IC gates and plots
│   ├── geometry.py, boundaries.py, initial.py # domain, BC and IC libraries
│   ├── digitised/                             # digitised contact CSVs
│   ├── figures/                               # source reference figures
│   ├── output/                                # generated verification plots
│   └── docs/DECISIONS.md                      # geometry/BC/IC design log
├── phase4_materials_and_residuals/
│   ├── vg.py, materials.py, properties.py     # constitutive model + material set
│   ├── nondim.py                              # non-dimensionalisation (Pi groups)
│   ├── derivatives.py                         # autograd differentiation layer
│   ├── residuals.py, loss.py                  # PDE residuals
│   ├── sampling.py                             # collocation point sampling
│   └── model.py, config.py                    # PINN architecture (untrained)
├── tests/                              # pytest suite for phase4
├── tools/                              # environment/GPU checks
└── All files/                          # original raw handoff archives (unextracted)
```

## Notes on this reorganization

- All internal imports and file-path references were updated to match the
  new layout (see `git log` for the commit that performed this pass).
- Four legacy top-level folders (`Hoek-Brown parameters/`, `SWCC & elastic
  parameters/`, `Porosity & remaining parameters/`, `Non-dimensionalisation/`)
  were removed: their contents were already fully duplicated into `docs/`,
  `data/`, and the current phase folders (one file, the SWCC derivation
  script, had a corrupted line fixed during this pass — see `git log`).
- A duplicated `step21_geometry/` folder (at repo root and under `src/`) was
  merged into the single `phase3_geometry_bc_ic/` folder; the root copy's
  unique figures, output plots, and `check_tags.py` were preserved.
- `docs/open_items.md` is a live research log of open **physics/numerical**
  questions (e.g. reference porosity provenance, an elastic-modulus choice)
  and was deliberately left untouched — those require domain judgement, not
  a code fix.
