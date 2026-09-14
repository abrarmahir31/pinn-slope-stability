"""Tests for the L-BFGS polish phase (`--lbfgs-epochs`).

The load-bearing claim is that the balancer weights are FROZEN across the
handover. That is not a tuning preference: L-BFGS builds its inverse-Hessian
estimate from (s, y) gradient-difference pairs and the strong-Wolfe line search
compares objective values across several evaluations inside one iteration. If
`bal.w` moved underneath, the curvature pairs would come from different
objectives and the line search would compare values of different functions.
Neither failure raises -- the run would complete and descend on nothing in
particular -- so it is pinned here.

Also pinned: the default is Adam-only, so every run and artifact from before
Day 27 is reproducible; and the handover checkpoint always exists, because
Fig. 4 marks that step and the Adam-only field has to stay recoverable if the
polish turns out to make things worse.
"""
import json
import pathlib

import pytest

from scripts.train import main


def _run(tmp_path, *extra):
    argv = ["--epochs", "6", "--n-pde", "60", "--n-bc", "60", "--n-iface", "60",
            "--log-every", "2", "--ckpt-every", "0",
            "--out", str(tmp_path)] + list(extra)
    main(argv)
    return [json.loads(l) for l in
            (tmp_path / "log.jsonl").read_text().splitlines() if l.strip()]


def test_default_is_adam_only(tmp_path):
    """No L-BFGS unless asked. Every pre-Day-27 run depends on this."""
    recs = _run(tmp_path)
    assert {r["phase"] for r in recs} == {"adam"}
    assert recs[-1]["step"] == 6


def test_lbfgs_phase_runs_and_is_labelled(tmp_path):
    recs = _run(tmp_path, "--lbfgs-epochs", "4")
    phases = [r["phase"] for r in recs]
    assert "lbfgs" in phases, "the L-BFGS phase produced no log records"
    assert phases.index("lbfgs") > 0, "L-BFGS records appear before Adam's"
    # Steps continue past --epochs rather than restarting, so Fig. 4 can plot
    # one x-axis with the handover marked at --epochs.
    lb = [r["step"] for r in recs if r["phase"] == "lbfgs"]
    assert min(lb) > 6, f"L-BFGS steps {lb} overlap the Adam phase"


def test_weights_are_frozen_across_the_handover(tmp_path):
    """The correctness requirement. See the module docstring."""
    recs = _run(tmp_path, "--lbfgs-epochs", "4", "--every", "1")
    adam = [r for r in recs if r["phase"] == "adam"]
    lb = [r for r in recs if r["phase"] == "lbfgs"]
    assert lb, "no L-BFGS records to check"

    handover = adam[-1]["w"]
    for r in lb:
        assert r["w"] == handover, (
            f"w changed during L-BFGS at step {r['step']}: "
            f"{r['w']} vs {handover}. The balancer is being stepped inside "
            "the polish, so the line search is comparing values of different "
            "objectives and the curvature pairs are inconsistent.")
        assert r["c"] == adam[-1]["c"]


def test_stale_balancer_diagnostics_are_nulled_not_repeated(tmp_path):
    """`spread` in the L-BFGS phase would be a cache, not a measurement.

    bal._last_norms still holds the final Adam update's gradient norms. Echoing
    them would draw a flat line in Fig. 4 that looks measured. Nulls force the
    reader (and the plotting code) to notice the phase boundary.
    """
    recs = _run(tmp_path, "--lbfgs-epochs", "4")
    for r in recs:
        if r["phase"] == "lbfgs":
            assert r["spread"] is None
            assert r["grad_norms"] == {}
        else:
            assert "grad_norms" in r


def test_the_handover_checkpoint_always_exists(tmp_path):
    """Independent of --ckpt-every, which is 0 here.

    Fig. 4 marks the handover step; if the polish makes the field worse, the
    comparison is against this file. Losing it means rerunning the Adam phase.
    """
    _run(tmp_path, "--lbfgs-epochs", "3")
    adam_ckpt = tmp_path / "ckpt_adam_final.pt"
    assert adam_ckpt.exists(), "no ckpt_adam_final.pt written at the handover"
    assert (tmp_path / "ckpt_final.pt").exists()


def test_lbfgs_actually_reduces_the_objective(tmp_path):
    """Sanity, not a quality claim.

    A second-order method on a fixed objective should not INCREASE it. This
    says nothing about whether the minimum is physically meaningful -- see the
    warning in `_lbfgs_phase`'s docstring about w[pde_richards] ~ 8e3 on a
    satisfaction-limited term.
    """
    recs = _run(tmp_path, "--lbfgs-epochs", "6")
    lb = [r for r in recs if r["phase"] == "lbfgs"]
    adam_end = [r for r in recs if r["phase"] == "adam"][-1]["total"]
    assert lb[-1]["total"] <= adam_end * 1.001, (
        f"L-BFGS raised the weighted total from {adam_end:.6g} to "
        f"{lb[-1]['total']:.6g} on a frozen objective")