
Active branch: `step-3.2-loss`. Clone with `git clone -b step-3.2-loss <url>` — `main` lags behind.

## First run after a fresh clone

Two caches are gitignored and 11 tests fail / 42 error without them:

    cd src/step21_geometry
    python 14_ic_checks.py       # writes ic_cache.npz
    python 17_sigma0_solve.py    # writes sigma0_cache.npz
    cd ../..
    PYTHONPATH=. python -m pytest -q     # 307 passed, 4 skipped
