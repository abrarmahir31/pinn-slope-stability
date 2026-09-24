@echo off
setlocal
REM Phase 6 on the lab PC. Run only after Phase 5 has a reportable FOS.
REM   scripts\run_phase6.bat arms        write configs\arms\*.json
REM   scripts\run_phase6.bat sweeps      one sweep per arm file
REM   scripts\run_phase6.bat npde        N_PDE 5000 and 20000 at D-5.17 scaled w_yield
REM
REM The whole remaining queue, unattended, is scripts\run_remaining.bat.
REM   scripts\run_phase6.bat draws       collocation-draw replicates
REM   scripts\run_phase6.bat figs        collect results, Figs 11 12
REM
REM ===== D-5.3 and D-5.5 are filled in. The rest stay open until Phase 5 is done. =====
set "BASELINE=runs\ansatz_epsuv1e-2\ckpt_final.pt"
set "YIELD_NORM=raw"
set ADM_METRIC=min_tag
REM D-5.13 calibrated criterion - must match run_phase5.bat exactly
set ADM_MODE=floor
set ADM_FLOOR=0.50
set PLATEAU_FACTOR=5
set DISP_FACTOR=5
set SIG3_LO=0
set SIG3_HI=179000
REM D-5.8: w_yield 1.0, conditional on the Phase 5 plateau holding
set W_REPORT=1
REM D-6.1 (proposed 24 Sep, supervisor to ratify): both arms on Mk_d.
REM K_s x0.1 / x10 (marl K_s was never measured); GSI 35 / 55 (Table 3b 45 +/- 10).
set KS_STRATUM=Mk_d
set GSI_STRATUM=Mk_d
set KS_FACTORS=0.1,10
set GSI_LEVELS=35,45,55
REM ===========================================================================

if "%BASELINE%"=="CHANGE_ME" (echo DECISION MISSING: BASELINE & exit /b 1)
if "%YIELD_NORM%"=="CHANGE_ME" (echo DECISION MISSING: YIELD_NORM & exit /b 1)
if "%ADM_METRIC%"=="CHANGE_ME" (echo DECISION MISSING: ADM_METRIC & exit /b 1)
if "%ADM_FLOOR%"=="CHANGE_ME" (echo DECISION MISSING: ADM_FLOOR - D-5.13 & exit /b 1)
if "%SIG3_LO%"=="CHANGE_ME" (echo DECISION MISSING: SIG3_LO & exit /b 1)
if "%SIG3_HI%"=="CHANGE_ME" (echo DECISION MISSING: SIG3_HI & exit /b 1)
if "%W_REPORT%"=="CHANGE_ME" (echo DECISION MISSING: W_REPORT - the w_yield chosen from the Phase 5 plateau & exit /b 1)

set RUN=python scripts\ssr_sweep.py --criterion GHB --sig3-lo %SIG3_LO% --sig3-hi %SIG3_HI% --baseline %BASELINE% --yield-norm %YIELD_NORM% --admissible-metric %ADM_METRIC% --admissible-mode %ADM_MODE% --admissible-floor %ADM_FLOOR% --plateau-factor %PLATEAU_FACTOR% --disp-factor %DISP_FACTOR% --w-yield %W_REPORT% --srf-step 0.25 --bisect-tol 0.01 --srf-max 4.0

if "%1"=="arms" (
  if "%KS_STRATUM%"=="CHANGE_ME" (echo DECISION MISSING: KS_STRATUM Mk, Mk_d or Tm & exit /b 1)
  if "%GSI_STRATUM%"=="CHANGE_ME" (echo DECISION MISSING: GSI_STRATUM Mk or Mk_d & exit /b 1)
  python scripts\make_arms.py --ks-stratum %KS_STRATUM% --gsi-stratum %GSI_STRATUM% --ks-factors %KS_FACTORS% --gsi-levels %GSI_LEVELS% --out configs\arms
  if errorlevel 1 exit /b 1
  exit /b 0
)

if "%1"=="sweeps" (
  for %%F in (configs\arms\*.json) do (
    %RUN% --overrides %%F --out runs\ssr_arm_%%~nF
    if errorlevel 1 exit /b 1
  )
  exit /b 0
)

if "%1"=="npde" (
  REM D-5.17: legacy L_yield is a per-stratum sum, so w_yield scales as 10000/N_PDE.
  REM RUN already carries --w-yield %W_REPORT%; argparse keeps the LAST value.
  %RUN% --n-pde 5000 --w-yield 2 --out runs\ssr_ghb_n5000_wscaled
  if errorlevel 1 exit /b 1
  %RUN% --n-pde 20000 --w-yield 0.5 --out runs\ssr_ghb_n20000_wscaled
  if errorlevel 1 exit /b 1
  exit /b 0
)

if "%1"=="draws" (
  echo NOTE: varies the collocation draw only. Network-seed replicates need
  echo retrained baselines, see docs\PHASE5_6_RUNBOOK.md section 3.
  for %%S in (11 12 13 14) do (
    %RUN% --sampling-seed %%S --out runs\ssr_ghb_draw%%S
    if errorlevel 1 exit /b 1
  )
  exit /b 0
)

if "%1"=="figs" (
  python scripts\collect_results.py --runs "runs\ssr*" --out docs\results
  if errorlevel 1 exit /b 1
  python scripts\make_ssr_figs.py --runs "runs\ssr*" --out docs\figs --only fig11,fig12
  exit /b 0
)

echo Usage: scripts\run_phase6.bat arms, sweeps, npde, draws or figs
exit /b 1
