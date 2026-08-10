#!/usr/bin/env bash
cd ~/thesis-pinn
git log --oneline -6
git status -sb
pytest -q 2>&1 | tail -3
python -m scripts.verify_dtheta_propagation 2>/dev/null | tail -4
