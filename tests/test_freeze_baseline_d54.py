"""freeze_baseline --accept-elastic-overstress overrides ONE gate (D-5.4)."""
import json
import os

import pytest
import torch

from scripts import freeze_baseline as fb


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "runs" / "prod"
    run.mkdir(parents=True)
    torch.save({"net": {}, "step": 5, "cfg": {"ansatz": "exp", "eps_psi": 0.3}},
               run / "ckpt_final.pt")
    (tmp_path / "docs").mkdir()
    json.dump({"frac_yielded_domain": 0.1925},
              open(tmp_path / "docs" / "fig6_elastic_check.json", "w"))
    state = {"dirty": ""}
    monkeypatch.setattr(fb, "git", lambda *a: state["dirty"] if a[:1] == ("status",) else "")
    return run, state


def _freeze(run, *flags):
    return fb.main([str(run), "--tag", "t", "--dest", "baselines", "--skip-tests",
                    *flags])


def test_overstressed_baseline_is_refused_without_the_flag(workspace):
    run, _ = workspace
    assert _freeze(run) == 1


def test_flag_accepts_the_overstress_and_records_it(workspace):
    run, _ = workspace
    assert _freeze(run, "--accept-elastic-overstress") == 0
    man = json.load(open(os.path.join("baselines", "t", "MANIFEST.json")))
    assert any("D-5.4" in o for o in man["overrides"])


def test_flag_does_not_override_a_dirty_tree(workspace):
    run, state = workspace
    state["dirty"] = " M src/loss.py"
    assert _freeze(run, "--accept-elastic-overstress") == 1
