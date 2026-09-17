"""scripts/decision_evidence.py pure helpers."""
import math

import numpy as np
import pytest

from scripts import decision_evidence as E
from src import strength as st
from src.materials import load_materials

# Phase-1 dataset hoek_brown_mc_equivalents (c Pa, phi deg)
STORED = {"Mk": {1e5: (61785, 36.57), 8.5e5: (130540, 21.72)},
          "Mk_d": {1e5: (16835, 21.69), 8.5e5: (50363, 10.04)}}


@pytest.mark.parametrize("tag", ["Mk", "Mk_d"])
def test_hoek2002_reproduces_the_stored_equivalents(tag):
    m = load_materials()[tag]
    p = st.GHBParams(sigma_ci=m.sigma_ci, m_b=m.m_b, s=m.s, a=m.a)
    for s3max, (c, phi) in STORED[tag].items():
        q = E.hoek2002_mc(p, s3max)
        assert q.c == pytest.approx(c, rel=5e-4)
        assert math.degrees(q.phi) == pytest.approx(phi, abs=0.02)


def test_hoek2002_phi_falls_with_the_fit_range():
    m = load_materials()["Mk_d"]
    p = st.GHBParams(sigma_ci=m.sigma_ci, m_b=m.m_b, s=m.s, a=m.a)
    assert E.hoek2002_mc(p, 1e5).phi > E.hoek2002_mc(p, 8.5e5).phi


def test_ghb_at_D1_reproduces_the_solver_material():
    m = load_materials()["Mk_d"]
    g = E.ghb_at_D("Mk_d", 1.0)
    assert g.m_b == pytest.approx(m.m_b, rel=1e-3)
    assert g.s == pytest.approx(m.s, rel=1e-3)


def test_lower_disturbance_is_stronger():
    assert E.ghb_at_D("Mk_d", 0.7).s > E.ghb_at_D("Mk_d", 1.0).s


def test_admissible_by_tag_counts_points_inside_the_envelope():
    p = st.MCParams(c=1e4, phi=math.radians(30))
    s3 = np.array([1e5, 1e5])
    s1_env = st.mc_sigma1(s3, p)
    sig = {"Mk": (np.array([0.5, 2.0]) * s1_env, s3)}
    assert E.admissible_by_tag(sig, ["Mk"], {"Mk": p}, "MC")["Mk"] == 0.5
