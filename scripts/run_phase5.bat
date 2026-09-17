@echo off
setlocal
REM Phase 5 on the lab PC. Usage, from the repo root after go.bat:
REM   scripts\run_phase5.bat smoke      first execution, minutes
REM   scripts\run_phase5.bat sweeps     GHB, and MC if enabled, at w_yield 0.3 1 3
REM   scripts\run_phase5.bat cold       cold-start check at w_yield 1
REM   scripts\run_phase5.bat figs       collect results, Figs 8 10, Fig 9
REM
REM ===== DECISIONS - every value must be set. There are no defaults. =====
REM See docs\PHASE5_6_RUNBOOK.md section 1 for what each one means.
set BASELINE=CHANGE_ME
set YIELD_NORM=CHANGE_ME
set ADM_METRIC=CHANGE_ME
set ADM_DROP=CHANGE_ME
set SIG3_LO=CHANGE_ME
set SIG3_HI=CHANGE_ME
set RUN_MC=CHANGE_ME
set FIG9_STATE=CHANGE_ME
REM ===========================================================================

if "%BASELINE%"=="CHANGE_ME" (echo DECISION MISSING: BASELINE & exit /b 1)
if "%YIELD_NORM%"=="CHANGE_ME" (echo DECISION MISSING: YIELD_NORM raw or ref & exit /b 1)
if "%ADM_METRIC%"=="CHANGE_ME" (echo DECISION MISSING: ADM_METRIC area, min_tag or tag:Mk_d & exit /b 1)
if "%ADM_DROP%"=="CHANGE_ME" (echo DECISION MISSING: ADM_DROP & exit /b 1)
if "%SIG3_LO%"=="CHANGE_ME" (echo DECISION MISSING: SIG3_LO in Pa & exit /b 1)
if "%SIG3_HI%"=="CHANGE_ME" (echo DECISION MISSING: SIG3_HI in Pa & exit /b 1)
if "%RUN_MC%"=="CHANGE_ME" (echo DECISION MISSING: RUN_MC yes or no - read runbook section 2.1 first & exit /b 1)
if "%FIG9_STATE%"=="CHANGE_ME" (echo DECISION MISSING: FIG9_STATE failed or stable & exit /b 1)

set COMMON=--baseline %BASELINE% --yield-norm %YIELD_NORM% --admissible-metric %ADM_METRIC% --admissible-drop %ADM_DROP%
set GHB=--criterion GHB --sig3-lo %SIG3_LO% --sig3-hi %SIG3_HI%

if "%1"=="smoke" (
  python scripts\ssr_sweep.py %GHB% %COMMON% --w-yield 1 --smoke --out runs\ssr_smoke_ghb
  if errorlevel 1 exit /b 1
  if "%RUN_MC%"=="yes" python scripts\ssr_sweep.py --criterion MC %COMMON% --w-yield 1 --smoke --out runs\ssr_smoke_mc
  if errorlevel 1 exit /b 1
  echo Smoke done. Read the adm column per stratum before launching sweeps.
  exit /b 0
)

if "%1"=="sweeps" (
  for %%W in (0.3 1 3) do (
    python scripts\ssr_sweep.py %GHB% %COMMON% --w-yield %%W --out runs\ssr_ghb_w%%W
    if errorlevel 1 exit /b 1
  )
  if "%RUN_MC%"=="yes" (
    for %%W in (0.3 1 3) do (
      python scripts\ssr_sweep.py --criterion MC %COMMON% --w-yield %%W --out runs\ssr_mc_w%%W
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
  python scripts\collect_results.py --runs "runs\ssr*" --out docs\results
  if errorlevel 1 exit /b 1
  python scripts\make_ssr_figs.py --runs "runs\ssr*" --out docs\figs --only fig8,fig10
  if "%RUN_MC%"=="yes" (
    python scripts\make_fig9.py --run runs\ssr_ghb_w1 --run runs\ssr_mc_w1 --state %FIG9_STATE% --out docs\figs\fig9.png
  ) else (
    python scripts\make_fig9.py --run runs\ssr_ghb_w1 --state %FIG9_STATE% --out docs\figs\fig9.png
  )
  exit /b 0
)

echo Usage: scripts\run_phase5.bat smoke, sweeps, cold or figs
exit /b 1
