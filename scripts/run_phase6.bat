@echo off
setlocal
REM Phase 6 on the lab PC. Run only after Phase 5 has a reportable FOS.
REM   scripts\run_phase6.bat arms        write configs\arms\*.json
REM   scripts\run_phase6.bat sweeps      one sweep per arm file
REM   scripts\run_phase6.bat npde        N_PDE levels 5000 and 20000
REM   scripts\run_phase6.bat draws       collocation-draw replicates
REM   scripts\run_phase6.bat figs        collect results, Figs 11 12
REM
REM ===== DECISIONS - copy the Phase 5 values, then set the rest. =====
set BASELINE=CHANGE_ME
set YIELD_NORM=CHANGE_ME
set ADM_METRIC=CHANGE_ME
set ADM_DROP=CHANGE_ME
set SIG3_LO=CHANGE_ME
set SIG3_HI=CHANGE_ME
set W_REPORT=CHANGE_ME
set KS_STRATUM=CHANGE_ME
set GSI_STRATUM=CHANGE_ME
REM ===========================================================================

if "%BASELINE%"=="CHANGE_ME" (echo DECISION MISSING: BASELINE & exit /b 1)
if "%YIELD_NORM%"=="CHANGE_ME" (echo DECISION MISSING: YIELD_NORM & exit /b 1)
if "%ADM_METRIC%"=="CHANGE_ME" (echo DECISION MISSING: ADM_METRIC & exit /b 1)
if "%ADM_DROP%"=="CHANGE_ME" (echo DECISION MISSING: ADM_DROP & exit /b 1)
if "%SIG3_LO%"=="CHANGE_ME" (echo DECISION MISSING: SIG3_LO & exit /b 1)
if "%SIG3_HI%"=="CHANGE_ME" (echo DECISION MISSING: SIG3_HI & exit /b 1)
if "%W_REPORT%"=="CHANGE_ME" (echo DECISION MISSING: W_REPORT - the w_yield chosen from the Phase 5 plateau & exit /b 1)

set RUN=python scripts\ssr_sweep.py --criterion GHB --sig3-lo %SIG3_LO% --sig3-hi %SIG3_HI% --baseline %BASELINE% --yield-norm %YIELD_NORM% --admissible-metric %ADM_METRIC% --admissible-drop %ADM_DROP% --w-yield %W_REPORT%

if "%1"=="arms" (
  if "%KS_STRATUM%"=="CHANGE_ME" (echo DECISION MISSING: KS_STRATUM Mk, Mk_d or Tm & exit /b 1)
  if "%GSI_STRATUM%"=="CHANGE_ME" (echo DECISION MISSING: GSI_STRATUM Mk or Mk_d & exit /b 1)
  python scripts\make_arms.py --ks-stratum %KS_STRATUM% --gsi-stratum %GSI_STRATUM% --out configs\arms
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
  for %%N in (5000 20000) do (
    %RUN% --n-pde %%N --out runs\ssr_ghb_n%%N
    if errorlevel 1 exit /b 1
  )
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
