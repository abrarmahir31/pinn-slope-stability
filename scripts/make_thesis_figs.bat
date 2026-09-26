@echo off
REM ==========================================================================
REM  make_thesis_figs.bat -- every thesis figure, thesis style, one command.
REM
REM    scripts\make_thesis_figs.bat            (from the repo root, after go.bat)
REM
REM  Style: src\plot_style.py -- Times New Roman 12 pt, 6.25 in wide (A4, 1-in
REM  margins), PNG 300 dpi + vector PDF next to each PNG, in docs\figs\.
REM
REM  Sources (all committed, so this runs on the laptop too, except Fig 9):
REM    Figs 2-6   baselines\baseline-v2\run\  (frozen checkpoint + log, D-5.6)
REM    Fig 7      retrains the small verification PINN (~2-3 min CPU)
REM    Figs 8, 10, 11, 12, S2
REM               docs\results\final_results.json + *_sweep.jsonl
REM               (ADOPTED criterion, D-5.18 / D-5.19 -- not the as-run FOS)
REM    Fig 9      runs\ssr_ghb_w1 and runs\ssr_mc_w1 states (lab PC only;
REM               runs\ is gitignored), state = failed (D-5.15)
REM  Fig S1 is absorbed into Fig 10(b) (O-5). Fig 1 is redrawn by hand (O-19).
REM ==========================================================================
setlocal
set BASE=baselines\baseline-v2\run
set OUT=docs\figs
set PYTHONPATH=.

python scripts\make_fig2.py %BASE%\ckpt_final.pt --out %OUT%\fig2.png            || goto :fail
python scripts\make_fig3.py --out %OUT%\fig3.png                                  || goto :fail
python scripts\make_fig4.py %BASE%\log.jsonl --out %OUT%\fig4.png                 || goto :fail
python scripts\make_fig5.py %BASE%\ckpt_final.pt --out %OUT%\fig5.png             || goto :fail
python scripts\make_fig6.py %BASE%\ckpt_final.pt --out %OUT%\fig6.png --json %OUT%\fig6.json || goto :fail
python scripts\make_fig6.py %BASE%\ckpt_final.pt --no-bishop --out %OUT%\fig6_nobishop.png --json %OUT%\fig6_nobishop.json || goto :fail
python scripts\verify_nguyen_raudkivi.py --out %OUT%\fig7.png --json %OUT%\fig7.json || goto :fail
python scripts\make_results_figs.py --results docs\results --out %OUT% --require  || goto :fail

if exist runs\ssr_mc_w1\states (
   python scripts\make_fig9.py --run runs\ssr_ghb_w1 --run runs\ssr_mc_w1 --state failed --out %OUT%\fig9.png || goto :fail
) else (
  echo SKIPPED Fig 9: runs\ssr_ghb_w1\states and runs\ssr_mc_w1\states not on this machine
)

echo.
echo All figures written to %OUT% ^(PNG + PDF^). Check them, then:
echo   git add src\plot_style.py scripts tests docs\figs
echo   git commit -m "Thesis figure style: Times 12 pt, 6.25 in, PNG+PDF; adopted-criterion Figs 8 10 11 12 S2"
exit /b 0

:fail
echo.
echo FAILED -- the command above did not complete. Nothing after it was run.
exit /b 1
