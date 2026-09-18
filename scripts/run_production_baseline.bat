@echo off
setlocal
REM D-5.6: train the production baseline that headline FOS numbers must use.
REM baseline-v1 (2,000 epochs) is for code validation and smoke sweeps only.
REM   scripts\run_production_baseline.bat train     about 16 h at 100k epochs on the RTX 4060 Ti
REM   scripts\run_production_baseline.bat check     base drift, psi, loss ratios vs baseline-v1
REM   scripts\run_production_baseline.bat fig6      elastic check on the new checkpoint (freeze gate input)
REM   scripts\run_production_baseline.bat freeze    archive as baseline-v2 with hashes
REM   scripts\run_production_baseline.bat figs      thesis Figs 2 3 4 5 6 into docs\figs
REM
REM ===== DECIDED - DECISIONS.md D-5.7 =====
set ANSATZ=cap
set LBFGS_EPOCHS=500
set SEED=7
set OUT=runs\ansatz_epsuv1e-2
set TAG=baseline-v2
REM
REM ===== OPEN - DECISIONS.md D-5.7 =====
REM SEED   configs\production_labpc.args sets 20250812. In the D-A.1 ablation
REM        the cap ansatz FAILED on exactly that seed: pde_richards ratio 0.418
REM        against about 1e-5 on seeds 1234 and 7, psi pinned at the cap.
REM        Set a seed deliberately. 20250812 is allowed if you accept that.
REM ===========================================================================


if "%ANSATZ%"=="CHANGE_ME" (echo DECISION MISSING: ANSATZ cap or exp - D-A.1 & exit /b 1)
if "%LBFGS_EPOCHS%"=="CHANGE_ME" (echo DECISION MISSING: LBFGS_EPOCHS & exit /b 1)
if "%SEED%"=="CHANGE_ME" (echo DECISION MISSING: SEED - read the note above, cap failed on 20250812 & exit /b 1)

if "%1"=="train" (
  if exist %OUT%\log.jsonl (echo REFUSING: %OUT% already has a log. Use a new OUT or resume by hand. & exit /b 1)
  python scripts\train.py @configs\production_labpc.args --ansatz %ANSATZ% --lbfgs-epochs %LBFGS_EPOCHS% --seed %SEED% --out %OUT%
  if errorlevel 1 exit /b 1
  exit /b 0
)

if "%1"=="check" (
  python scripts\check_baseline.py %OUT% --compare runs\ansatz\exp_seed7 --json docs\results\baseline_v2_check.json
  exit /b 0
)

if "%1"=="figs" (
  python scripts\make_fig2.py %OUT%\ckpt_final.pt --out docs\figs\fig2.png
  if errorlevel 1 exit /b 1
  python scripts\make_fig3.py --out docs\figs\fig3.png
  if errorlevel 1 exit /b 1
  python scripts\make_fig4.py %OUT%\log.jsonl --out docs\figs\fig4.png
  if errorlevel 1 exit /b 1
  python scripts\make_fig5.py %OUT%\ckpt_final.pt --out docs\figs\fig5.png
  if errorlevel 1 exit /b 1
  python scripts\make_fig6.py %OUT%\ckpt_final.pt --out docs\figs\fig6.png --json docs\figs\fig6.json
  if errorlevel 1 exit /b 1
  python scripts\make_fig6.py %OUT%\ckpt_final.pt --no-bishop --out docs\figs\fig6_nobishop.png --json docs\figs\fig6_nobishop.json
  if errorlevel 1 exit /b 1
  exit /b 0
)

if "%1"=="fig6" (
  python scripts\make_fig6.py %OUT%\ckpt_final.pt --out docs\fig6.png --json docs\fig6_elastic_check.json
  if errorlevel 1 exit /b 1
  exit /b 0
)

if "%1"=="freeze" (
  python scripts\freeze_baseline.py %OUT% --tag %TAG% --accept-elastic-overstress
  if errorlevel 1 exit /b 1
  echo Frozen. Point BASELINE in run_phase5.bat and run_phase6.bat at %OUT%\ckpt_final.pt
  exit /b 0
)

echo Usage: scripts\run_production_baseline.bat train, check, fig6, freeze or figs
exit /b 1
