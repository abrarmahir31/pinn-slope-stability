@echo off
REM Every remaining Phase 5/6 run, unattended, then the result tables.
REM Run from a Miniforge Prompt with the pinn env active:
REM   conda activate C:\conda-envs\pinn
REM   cd /d C:\work\pinn-slope-stability
REM   scripts\run_remaining.bat
REM Waits for the GPU to be free first. Safe to stop (Ctrl+C) and rerun: it resumes.
REM   scripts\run_remaining.bat --dry-run     show the plan, run nothing
setlocal
cd /d "%~dp0.."
set PYTHONPATH=.
python -u scripts\run_remaining.py %*
exit /b %errorlevel%
