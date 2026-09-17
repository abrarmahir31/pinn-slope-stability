@echo off
setlocal
REM Phase 5 on the lab PC. Usage, from the repo root after go.bat:
REM   scripts\run_phase5.bat smoke      GHB smoke under BOTH yield norms - evidence for YIELD_NORM and ADM_DROP
REM   scripts\run_phase5.bat smoke_mc   MC rock-mass smoke - evidence for D-5.2 option a vs c
REM   scripts\run_phase5.bat sweeps     GHB, and MC if RUN_MC=yes, at w_yield 0.3 1 3
REM   scripts\run_phase5.bat cold       cold-start check at w_yield 1
REM   scripts\run_phase5.bat figs       collect results, Figs 8 10, Fig 9
REM
REM ===== DECIDED - Day 41 supervisor meeting, DECISIONS.md Step 5 =====
REM D-5.1 soft constraint, w_yield sweep 0.3 1 3 is built into the sweeps stage
REM D-5.3 GHB fit range 0 to 179 kPa
REM D-5.4 D = 1 is in the material data, nothing to set here
REM D-5.5 admissible metric: min_tag primary, tag:Mk_d secondary - recorded, not used for detection
REM D-5.6 baseline-v1 is for code validation and smoke sweeps ONLY. Before headline
REM       numbers: scripts\run_production_baseline.bat, then point BASELINE at it.
REM       Results on baseline-v1 are marked validation_only by collect_results.py.
set BASELINE=runs\ansatz\exp_seed7\ckpt_final.pt
set SIG3_LO=0
set SIG3_HI=179000
set ADM_METRIC=min_tag
set ADM_SECONDARY=tag:Mk_d
REM D-5.2 option a: rock-mass MC equivalents over 0 to 400 kPa
REM D-5.5 drop threshold 0.15 - confirm against the smoke trajectories first
REM D-5.8 Fig 9 at t = 30 d
set RUN_MC=yes
set MC_SIG3MAX=400000
set ADM_DROP=0.15
set FIG9_T=30
REM
REM ===== STILL OPEN - see DECISIONS.md "Still open after Day 41" =====
REM YIELD_NORM   raw or ref ONLY. These are the two modes ssr_sweep.py has.
REM              The smoke stage runs both.
REM FIG9_STATE   failed or stable - which end of the SRF bracket, not a time
set YIELD_NORM=CHANGE_ME
set FIG9_STATE=CHANGE_ME
REM ===========================================================================

set BASE=--baseline %BASELINE% --admissible-metric %ADM_METRIC% --admissible-secondary %ADM_SECONDARY%
set GHB=--criterion GHB --sig3-lo %SIG3_LO% --sig3-hi %SIG3_HI%
REM Smoke runs never bisect, so their drop threshold only decides an early stop.
REM 0.5 is a placeholder recorded in result.json with smoke=true, and
REM collect_results.py excludes smoke runs. It is NOT the ADM_DROP decision.
set SMOKE=--smoke --w-yield 1 --admissible-drop 0.5

if "%1"=="smoke" (
  python scripts\ssr_sweep.py %GHB% %BASE% %SMOKE% --yield-norm ref --out runs\ssr_smoke_ghb_ref
  if errorlevel 1 exit /b 1
  python scripts\ssr_sweep.py %GHB% %BASE% %SMOKE% --yield-norm raw --out runs\ssr_smoke_ghb_raw
  if errorlevel 1 exit /b 1
  echo.
  echo Smoke done. Send both logs. Compare the Mk_d column under ref and raw,
  echo and the sec per SRF, before setting YIELD_NORM and ADM_DROP.
  exit /b 0
)

if "%1"=="smoke_mc" (
  if "%MC_SIG3MAX%"=="CHANGE_ME" (echo DECISION MISSING: MC_SIG3MAX in Pa & exit /b 1)
  python scripts\ssr_sweep.py --criterion MC --mc-strength rockmass --mc-sig3max %MC_SIG3MAX% %BASE% %SMOKE% --yield-norm ref --out runs\ssr_smoke_mc_rockmass
  if errorlevel 1 exit /b 1
  exit /b 0
)

if "%YIELD_NORM%"=="CHANGE_ME" (echo DECISION MISSING: YIELD_NORM raw or ref - run the smoke stage first & exit /b 1)
if not "%YIELD_NORM%"=="raw" if not "%YIELD_NORM%"=="ref" (echo YIELD_NORM must be raw or ref, got %YIELD_NORM% & exit /b 1)
if "%ADM_DROP%"=="CHANGE_ME" (echo DECISION MISSING: ADM_DROP - run the smoke stage first & exit /b 1)
if "%RUN_MC%"=="CHANGE_ME" (echo DECISION MISSING: RUN_MC yes or no - D-5.2 option a or c & exit /b 1)
if "%RUN_MC%"=="yes" if "%MC_SIG3MAX%"=="CHANGE_ME" (echo DECISION MISSING: MC_SIG3MAX in Pa & exit /b 1)

set COMMON=%BASE% --yield-norm %YIELD_NORM% --admissible-drop %ADM_DROP%
set MC=--criterion MC --mc-strength rockmass --mc-sig3max %MC_SIG3MAX%

if "%1"=="sweeps" (
  for %%W in (0.3 1 3) do (
    python scripts\ssr_sweep.py %GHB% %COMMON% --w-yield %%W --out runs\ssr_ghb_w%%W
    if errorlevel 1 exit /b 1
  )
  if "%RUN_MC%"=="yes" (
    for %%W in (0.3 1 3) do (
      python scripts\ssr_sweep.py %MC% %COMMON% --w-yield %%W --out runs\ssr_mc_w%%W
      if errorlevel 1 exit /b 1
    )
  )
  exit /b 0
)

if "%1"=="cold" (
  python scripts\ssr_sweep.py %GHB% %COMMON% --w-yield 1 --cold-start --out runs\ssr_ghb_w1_cold
  if errorlevel 1 exit /b 1
  exit /b 0
)

if "%1"=="figs" (
  if "%FIG9_STATE%"=="CHANGE_ME" (echo DECISION MISSING: FIG9_STATE failed or stable & exit /b 1)
  python scripts\collect_results.py --runs "runs\ssr*" --out docs\results
  if errorlevel 1 exit /b 1
  python scripts\make_ssr_figs.py --runs "runs\ssr*" --out docs\figs --only fig8,fig10
  if "%RUN_MC%"=="yes" (
    python scripts\make_fig9.py --run runs\ssr_ghb_w1 --run runs\ssr_mc_w1 --state %FIG9_STATE% --t %FIG9_T% --out docs\figs\fig9.png
  ) else (
    python scripts\make_fig9.py --run runs\ssr_ghb_w1 --state %FIG9_STATE% --t %FIG9_T% --out docs\figs\fig9.png
  )
  exit /b 0
)

echo Usage: scripts\run_phase5.bat smoke, smoke_mc, sweeps, cold or figs
exit /b 1
