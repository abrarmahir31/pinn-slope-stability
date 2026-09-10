@echo off
REM Applies the Day 26 changes to a Day 25 tree (HEAD = b126dba).
REM Run from the repo root: C:\work\pinn-slope-stability

git rev-parse --show-toplevel >nul 2>&1 || (echo Not a git repo. cd to the repo root first. & exit /b 1)

echo == checking the patch applies ==
git apply --check --whitespace=nowarn day26.patch || (echo Patch does not apply cleanly. Nothing changed. & exit /b 1)

echo == applying ==
git apply --whitespace=nowarn day26.patch || exit /b 1

echo == running the suite ==
python -m pytest -q
echo.
echo Expect: 307 passed, 4 skipped
echo If you get errors about ic_cache.npz / sigma0_cache.npz, regenerate them:
echo   cd src\step21_geometry ^&^& python 14_ic_checks.py ^&^& python 17_sigma0_solve.py ^&^& cd ..\..
