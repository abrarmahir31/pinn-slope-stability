"""
test_weighting.py -- Step 3.4 guards.

The failure this file exists to catch is not "the weights are slightly off".
It is "the balancer emitted inf on update 1 because one term was 1e-32, and
40 000 epochs were spent training a dead network". Each test is a mutation
test against a specific way of writing the module wrong; the docstring names
the mutation.

The fixture builds terms in BOTH regimes of D-W.0 on purpose:

  prefactor-limited     r = c * f(theta)                 -> p = 1
  satisfaction-limited  r = (f - f.detach()) + eps       -> p = 0.5

The second is the trick that makes a residual with a SMALL VALUE and an
O(1) GRADIENT. It is what ic_head / ic_disp / the constrained bc_mech
segments look like at the near-equilibrium initialisation.
"""
import math

import pytest
import torch

from weighting import (BASELINE_LOSSES, PI_SEED, BalancerConfig, LossBalancer,
                       TERMS, format_table, grad_norm, grad_norm_table,
                       pi_seed, regime_exponents, weight_bounds)


# --------------------------------------------------------------------------
class Toy(torch.nn.Module):
    def __init__(self, n=16):
        super().__init__()
        self.lin = torch.nn.Linear(2, n, dtype=torch.float64)
        self.out = torch.nn.Linear(n, 3, dtype=torch.float64)

    def forward(self, x):
        return self.out(torch.tanh(self.lin(x)))


#: the toy has no Pi-group structure, so the D-W.7 seed is switched off for
#: balancer tests and exercised separately in test_pi_seed_*.
FLAT = {k: 1.0 for k in TERMS}

#: which regime each term is built in, mirroring the real diagnosis
REGIME = {"bc_mech": "prefactor", "pde_mech": "prefactor",
          "pde_richards": "prefactor", "bc": "satisfaction",
          "ic_disp": "satisfaction", "ic_head": "satisfaction"}


@pytest.fixture
def toy():
    torch.manual_seed(0)
    net = Toy()
    x = torch.randn(300, 2, dtype=torch.float64)
    y = net(x)

    def prefactor(target_L, col):
        f = y[:, col]
        c = math.sqrt(target_L / float((f ** 2).mean().detach()))
        return ((c * f) ** 2).mean()

    def satisfaction(target_L, col):
        f = y[:, col]
        eps = math.sqrt(target_L)
        r = (f - f.detach()) + eps       # value eps, gradient df/dtheta
        return (r ** 2).mean()

    losses = {}
    for i, (k, L) in enumerate(BASELINE_LOSSES.items()):
        if k == "interface":
            losses[k] = torch.zeros((), dtype=torch.float64)
        elif REGIME[k] == "prefactor":
            losses[k] = prefactor(L, 0)
        else:
            losses[k] = satisfaction(L, 0)
    return net, losses


@pytest.fixture
def two_snapshots():
    """(losses, norms) at two network states -- what D-W.0b asks for in the
    real run: measure at the initialisation, take ~500 Adam steps, measure
    again. Here the second state is produced by perturbing theta."""
    def snap(seed, drift, decades):
        torch.manual_seed(0)
        net = Toy()
        with torch.no_grad():
            torch.manual_seed(seed)
            for prm in net.parameters():
                prm.add_(drift * torch.randn_like(prm))
        x = torch.randn(300, 2, dtype=torch.float64)
        y = net(x)

        def prefactor(target_L):
            f = y[:, 0]
            c = math.sqrt(target_L / float((f ** 2).mean().detach()))
            return ((c * f) ** 2).mean()

        def satisfaction(target_L):
            f = y[:, 0]
            r = (f - f.detach()) + math.sqrt(target_L)
            return (r ** 2).mean()

        losses = {}
        for k, L in BASELINE_LOSSES.items():
            if k == "interface":
                continue
            # the SECOND state has each term at a different magnitude; that
            # displacement is what the exponent is fitted on
            Lk = L * (10.0 ** decades)
            losses[k] = (prefactor(Lk) if REGIME[k] == "prefactor"
                         else satisfaction(Lk))
        norms = grad_norm_table(losses, list(net.parameters()))
        return ({k: float(v.detach()) for k, v in losses.items()}, norms)

    # theta held fixed, only the residual magnitudes move: this isolates the
    # estimator's algebra. `noisy_snapshots` adds the parameter drift that a
    # real 500-step gap also has.
    return snap(1, 0.0, 0.0), snap(1, 0.0, 0.6)





# --------------------------------------------------------------------------
# 1. measurement
# --------------------------------------------------------------------------
def test_grad_norm_matches_manual_backward(toy):
    """Mutation: summing |g| instead of sqrt(sum g^2), or missing a param."""
    net, losses = toy
    L = losses["pde_mech"]
    g_fn = grad_norm(L, list(net.parameters()))
    net.zero_grad()
    L.backward(retain_graph=True)
    g_manual = math.sqrt(sum(float(p.grad.pow(2).sum())
                             for p in net.parameters() if p.grad is not None))
    assert abs(g_fn - g_manual) / g_manual < 1e-12


def test_zero_term_gives_zero_not_error(toy):
    """`interface` has no graph at t = 0. A naive autograd.grad raises
    'does not require grad'; returning 0.0 is the contract D-W.4 needs."""
    net, losses = toy
    assert grad_norm(losses["interface"], list(net.parameters())) == 0.0


def test_regime_exponents_separate_the_two_causes(two_snapshots):
    """D-W.0, the whole point of the step. A term that is small because it is
    weakly scaled must be distinguishable from one that is small because it
    is nearly satisfied -- otherwise the weights are picked from a bracket
    four orders wide."""
    snap_a, snap_b = two_snapshots
    p = regime_exponents(snap_a, snap_b)
    assert set(p) >= set(REGIME)
    for k, pk in p.items():
        expect = 1.0 if REGIME[k] == "prefactor" else 0.5
        assert pk == pytest.approx(expect, abs=1e-6), f"{k}: p={pk:.6f}"


def test_cross_term_exponent_would_have_been_wrong(two_snapshots):
    """Records the bug this estimator replaced. Fitting p ACROSS terms folds
    the per-term constant k_i into the exponent: the satisfaction terms come
    out near 0.2-0.3, not 0.5, and the weights derived from that are wrong by
    orders. If someone reintroduces a cross-term fit, this is the evidence."""
    (La, ga), _ = two_snapshots
    anchor = "bc_mech"
    bad = {k: math.log(ga[k] / ga[anchor]) / math.log(La[k] / La[anchor])
           for k in La if k != anchor and La[k] > 0 and ga[k] > 0}
    sat = [bad[k] for k in bad if REGIME.get(k) == "satisfaction"]
    assert sat and max(sat) < 0.35, \
        "cross-term fit no longer misreports the satisfaction terms"


def test_format_table_labels_the_regimes(toy, two_snapshots):
    net, losses = toy
    norms = grad_norm_table(losses, list(net.parameters()))
    txt = format_table(norms, p=regime_exponents(*two_snapshots))
    assert "prefactor" in txt and "satisfaction" in txt
    assert "inactive" in txt          # interface


def test_weight_bounds_are_not_actually_bounds(toy):
    """weight_bounds() is orientation, NOT a bracket the measurement lands
    inside -- g_i = k_i L_i^p carries a per-term constant k_i that the loss
    table cannot see. Here `bc` measures w^ = 3.9 against a nominal bracket
    of [7.0e1, 4.8e3]. This is the strongest form of D-W.0: the note's loss
    table cannot set the weights to within four orders, let alone one.
    Anyone tempted to skip scripts/grad_norm_table.py should read this."""
    net, losses = toy
    norms = grad_norm_table(losses, list(net.parameters()))
    ga = norms["bc_mech"]
    outside = [k for k, (lo, hi) in weight_bounds().items()
               if k != "bc_mech" and not lo <= ga / norms[k] <= hi]
    assert outside, ("every term happened to land inside the nominal "
                     "bracket; that is luck, not a licence to use it")


# --------------------------------------------------------------------------
# 2. the balancer
# --------------------------------------------------------------------------
def test_interface_is_frozen_not_amplified(toy):
    """THE test. A 1/g rule on interface returns ~1e14-1e29 and kills
    training on update 1. Its weight must be untouched and finite."""
    net, losses = toy
    bal = LossBalancer(TERMS, net.parameters(),
                       BalancerConfig(warmup=0, every=1, seed=FLAT, log_clip=(-9, 9)))
    w = bal.update(losses)
    assert w["interface"] == 1.0
    assert "interface" in bal._frozen
    assert all(math.isfinite(v) for v in w.values())


def test_anchor_never_moves(toy):
    """D-W.2. If bc_mech drifts, the scheme can suppress the cut-face
    constraint, which the baseline note explicitly forbids."""
    net, losses = toy
    bal = LossBalancer(TERMS, net.parameters(),
                       BalancerConfig(warmup=0, every=1, lam=0.9, seed=FLAT, log_clip=(-9, 9)))
    for _ in range(20):
        bal.update(losses)
    assert bal.w["bc_mech"] == 1.0


def test_weights_move_the_right_way_and_close_the_gap(toy):
    """Mutation: w^ = g_i / tgt (inverted) passes almost every other test."""
    net, losses = toy
    bal = LossBalancer(TERMS, net.parameters(),
                       BalancerConfig(warmup=0, every=1, lam=0.5, seed=FLAT, log_clip=(-9, 9)))
    before = bal.w["pde_richards"]
    for _ in range(15):
        bal.update(losses)
    assert bal.w["pde_richards"] > before
    norms = bal._last_norms
    live = [k for k in TERMS if norms.get(k, 0) > 0]
    raw = max(norms[k] for k in live) / min(norms[k] for k in live)
    wtd = (max(bal.w[k] * norms[k] for k in live)
           / min(bal.w[k] * norms[k] for k in live))
    assert wtd < raw / 100.0
    assert wtd == pytest.approx(1.0, abs=0.5)   # converged to balance


def test_geomean_would_demote_the_anchor(toy):
    """D-W.2's rejection, made concrete. target='geomean' is the tidier rule
    and it is rejected precisely because it drives w_bc_mech below 1 -- so
    the rejection is tested, not merely asserted in a docstring. If this ever
    fails, the spread has changed and D-W.2 can be re-argued on data."""
    net, losses = toy
    bal = LossBalancer(TERMS, net.parameters(),
                       BalancerConfig(warmup=0, every=1, lam=1.0,
                                      target="geomean", seed=FLAT, log_clip=(-9, 9)))
    bal.update(losses)
    tgt = bal.history[-1]["target"]
    assert tgt < bal._last_norms["bc_mech"], \
        "geomean no longer sits below the anchor; revisit D-W.2"
    # under anchor targeting the same term gets a strictly larger weight
    bal2 = LossBalancer(TERMS, net.parameters(),
                        BalancerConfig(warmup=0, every=1, lam=1.0,
                                       target="anchor", seed=FLAT, log_clip=(-9, 9)))
    bal2.update(losses)
    assert bal2.w["pde_richards"] > bal.w["pde_richards"]


def test_clipping_is_reported_not_silent(toy):
    """A weight pinned to the clip is a diagnostic. If it is invisible the
    scheme looks converged while one term is still effectively unweighted."""
    net, losses = toy
    bal = LossBalancer(TERMS, net.parameters(),
                       BalancerConfig(warmup=0, every=1, lam=1.0,
                                      log_clip=(-0.1, 0.1), seed=FLAT))
    bal.update(losses)
    assert bal._clipped
    assert "CLIPPED" in bal.report()


def test_log_ema_actually_travels(toy):
    """Mutation: a LINEAR EMA. At lam = 0.1 it needs ~1e5 updates to go from
    1 to 1e4, i.e. the weights never arrive within a training run. Log-space
    must be within an order of the target after ~50 updates."""
    net, losses = toy
    bal = LossBalancer(TERMS, net.parameters(),
                       BalancerConfig(warmup=0, every=1, lam=0.1, seed=FLAT, log_clip=(-9, 9)))
    for _ in range(50):
        bal.update(losses)
    tgt = bal._last_norms["bc_mech"] / bal._last_norms["pde_richards"]
    assert abs(math.log10(bal.w["pde_richards"] / tgt)) < 1.0


def test_cadence_and_warmup(toy):
    """maybe_update must be free off-cadence -- it is called every step."""
    net, losses = toy
    bal = LossBalancer(TERMS, net.parameters(),
                       BalancerConfig(warmup=500, every=100))
    for s in (0, 99, 100, 499):
        bal.maybe_update(s, losses)
    assert bal._n_updates == 0
    bal.maybe_update(500, losses)
    assert bal._n_updates == 1


def test_total_refuses_a_partial_objective(toy):
    """A dropped physics term trains fine and answers wrong. Fail loudly."""
    net, losses = toy
    bal = LossBalancer(TERMS, net.parameters())
    partial = {k: v for k, v in losses.items() if k != "pde_richards"}
    with pytest.raises(KeyError):
        bal.total(partial)


def test_no_gradient_flows_through_the_weights(toy):
    """w must be a constant of the step. If it were trainable the optimiser
    would minimise the objective by shrinking w, not by satisfying physics."""
    net, losses = toy
    bal = LossBalancer(TERMS, net.parameters(),
                       BalancerConfig(warmup=0, every=1, seed=FLAT, log_clip=(-9, 9)))
    bal.update(losses)
    bal.total(losses).backward()
    assert all(isinstance(v, float) for v in bal.w.values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in net.parameters())


def test_per_stratum_is_measured_and_not_acted_on(toy):
    """D-W.5. The split must appear in the log and must not appear in w."""
    net, losses = toy
    losses = dict(losses)
    for u in ("Mk", "Mk_d", "Tm"):
        losses[f"pde_mech::{u}"] = losses["pde_mech"] * 1.0
    bal = LossBalancer(TERMS, net.parameters(),
                       BalancerConfig(warmup=0, every=1, seed=FLAT, log_clip=(-9, 9)),
                       subkeys={"pde_mech": ("Mk", "Mk_d", "Tm")})
    bal.update(losses)
    assert set(bal.history[-1]["sub"]) == {
        "pde_mech::Mk", "pde_mech::Mk_d", "pde_mech::Tm"}
    assert set(bal.w) == set(TERMS)
    assert "per-stratum" in bal.report()


def test_weighting_cannot_move_the_zero():
    """D-W.6. Every residual is zero at the exact initial state, so no weight
    vector can make the objective nonzero there. Complements
    tests/test_initial_equilibrium.py, which pins the state itself."""
    net = Toy()
    zeros = {k: torch.zeros((), dtype=torch.float64) for k in TERMS}
    bal = LossBalancer(TERMS, net.parameters(),
                       BalancerConfig(warmup=0, every=1, seed=FLAT, log_clip=(-9, 9)))
    bal.w.update({k: 10.0 ** i for i, k in enumerate(TERMS)})
    bal.w["bc_mech"] = 1.0
    assert float(bal.total(zeros)) == 0.0
    w_before = dict(bal.w)
    bal.update(zeros)                     # degenerate state: hold, do not 0/0
    assert bal.w == w_before


def test_state_dict_round_trip(toy):
    """Checkpoints must carry the weights. Resuming with w = 1 after 40 000
    epochs silently restarts the balancing."""
    net, losses = toy
    bal = LossBalancer(TERMS, net.parameters(),
                       BalancerConfig(warmup=0, every=1, lam=1.0))
    bal.update(losses)
    d = bal.state_dict()
    bal2 = LossBalancer(TERMS, net.parameters())
    bal2.load_state_dict(d)
    assert bal2.w == bal.w


def test_bounds_exclude_interface():
    b = weight_bounds()
    assert "interface" not in b
    assert b["pde_richards"][0] == pytest.approx(1.095e4, rel=0.02)
    assert b["pde_richards"][1] == pytest.approx(1.199e8, rel=0.02)


# --------------------------------------------------------------------------
# 3. the static seed (D-W.7)
# --------------------------------------------------------------------------
def test_pi_seed_is_the_squared_sensitivity_ratio():
    """The seed is arithmetic, not a tuned number. If Pi_M_body, Pi_R_grav or
    either ansatz prefactor changes, this must change with it, or the
    objective silently stops matching the scaling.

    The eps factors are half the formula: what multiplies the network inside
    each residual is Pi * eps, not Pi. See pi_seed's docstring."""
    from weighting import _PI_M_BODY, _PI_R_GRAV, _EPS_PSI, _EPS_UV
    assert PI_SEED["pde_richards"] == pytest.approx(
        ((_PI_M_BODY * _EPS_UV) / (_PI_R_GRAV * _EPS_PSI)) ** 2)
    assert PI_SEED["pde_richards"] == pytest.approx(4.221e1, rel=0.01)
    for k in ("bc_mech", "pde_mech", "bc", "ic_disp", "ic_head"):
        assert PI_SEED[k] == 1.0


def test_pi_seed_eps_literals_match_nearphysical():
    """weighting.py duplicates the ansatz prefactors as literals so it imports
    in a bare environment. They must not drift from the wrapper that actually
    applies them -- the Day 25 failure was exactly a stale eps_psi."""
    import inspect
    from src.model import NearPhysical
    from weighting import _EPS_PSI, _EPS_UV
    p = inspect.signature(NearPhysical.__init__).parameters
    assert _EPS_PSI == p["eps_psi"].default
    assert _EPS_UV == p["eps_uv"].default


def test_omitting_eps_was_a_factor_of_nine_under_the_old_ansatz():
    """Why the bug survived to Day 25. At eps_psi = 3e-3 the omitted ratio
    eps_uv/eps_psi was 1/3, so the seed was wrong by 9x and sat inside the
    clip. At eps_psi = 0.3 the same omission is 9e4."""
    from weighting import _PI_M_BODY, _PI_R_GRAV, pi_seed
    naive = (_PI_M_BODY / _PI_R_GRAV) ** 2
    assert naive / pi_seed(eps_psi=3e-3)["pde_richards"] == pytest.approx(9.0)
    assert naive / pi_seed(eps_psi=0.3)["pde_richards"] == pytest.approx(9e4)


def test_seed_brings_the_p1_weight_inside_the_default_clip():
    """The reason D-W.7 exists. Without the seed, pde_richards at p = 1 wants
    w = 1.2e8, which the default clip of 1e3 on the correction cannot reach
    and a clip on w would have hidden. With the seed the correction is ~284.

    BASELINE_LOSSES was measured at the Step 3.3 state, which is eps_psi =
    3e-3, so the seed it must be checked against is the one for that ansatz.
    Checking it against the eps_psi = 0.3 seed compares a loss table and a
    weight taken under two different networks -- which is the Day 25 error in
    miniature."""
    need = BASELINE_LOSSES["bc_mech"] / BASELINE_LOSSES["pde_richards"]
    c = need / pi_seed(eps_psi=3e-3)["pde_richards"]
    lo, hi = BalancerConfig().log_clip
    assert not lo <= math.log10(need) <= hi          # unseeded: out of reach
    assert lo <= math.log10(c) <= hi                 # seeded: comfortable
    assert c == pytest.approx(284.0, rel=0.02)


def test_seed_mismatch_on_resume_is_an_error(toy):
    """Resuming a checkpoint against a different seed silently changes the
    objective. Loud failure beats a quietly different experiment."""
    net, losses = toy
    bal = LossBalancer(TERMS, net.parameters(),
                       BalancerConfig(warmup=0, every=1, seed=FLAT))
    bal.update(losses)
    d = bal.state_dict()
    other = LossBalancer(TERMS, net.parameters())      # default PI_SEED
    with pytest.raises(ValueError, match="seed differs"):
        other.load_state_dict(d)

def test_pi_seed_8x64_matches_the_measurement_file():
    """PI_SEED_8X64 is no longer arithmetic -- it is a measurement, so it can
    go stale silently in a way pi_seed() cannot. Bind the literals to the JSON
    they were read from. The 16 Aug seed went stale exactly this way: it
    outlived the eps_psi it was measured under by three weeks."""
    import json
    import pathlib
    from weighting import GRADNORM_8X64_EPSPSI03
    p = (pathlib.Path(__file__).resolve().parents[1]
         / "docs" / "gradnorm_epspsi03.json")
    meas = json.loads(p.read_text())["grad_norms"]
    assert set(meas) == set(GRADNORM_8X64_EPSPSI03)
    for k, v in meas.items():
        assert GRADNORM_8X64_EPSPSI03[k] == pytest.approx(v, rel=1e-12)


def test_pi_seed_8x64_is_the_geometric_mean_of_both_snapshots():
    """D-W.0b in code. The seed must not be the t = 0 column alone: at t = 0
    `pde_richards` is satisfaction-limited and asks for w^ = 3.47, by step 500
    it asks for 3.87e+03, and a seed fitted to the first is clipped within 400
    epochs (runs/prod02_smoke, Day 26).

    The two held terms are the point of the seed, not an exception to it. A
    satisfaction-limited term returns a huge w^ and must not be seeded with
    it -- D-W.4 for `interface`, p = 0.57 for `ic_disp`."""
    from weighting import (GRADNORM_8X64_EPSPSI03,
                           GRADNORM_8X64_EPSPSI03_STEP500, PI_SEED_8X64)
    g0, g1 = GRADNORM_8X64_EPSPSI03, GRADNORM_8X64_EPSPSI03_STEP500
    for k in ("pde_richards", "pde_mech", "bc", "ic_head"):
        w0 = g0["bc_mech"] / g0[k]
        w1 = g1["bc_mech"] / g1[k]
        assert PI_SEED_8X64[k] == pytest.approx(math.sqrt(w0 * w1))
    assert PI_SEED_8X64["bc_mech"] == 1.0
    assert PI_SEED_8X64["ic_disp"] == PI_SEED["ic_disp"]
    assert PI_SEED_8X64["interface"] == PI_SEED["interface"]
    # the regime change the geometric mean exists to straddle, on record
    assert (g1["bc_mech"] / g1["pde_richards"]) / \
           (g0["bc_mech"] / g0["pde_richards"]) == pytest.approx(1117, rel=0.01)


def test_pi_seed_8x64_corrections_are_inside_the_clip_except_ic_disp():
    """The seed exists so the balancer never runs clipped -- at EITHER end of
    the regime change, which is what the geometric mean buys.

    `ic_disp` is the exception and this test records it rather than hiding it.
    Its gradient norm is identical under both ansaetze (4.7618e-05, it has no
    psi dependence) while the bc_mech anchor grew 1154x when eps_psi went to
    0.3. So w^ went 4.09e+01 -> 4.71e+04 without ic_disp changing at all, and
    a term held at w0 = 1 now needs a correction 1.7 orders past the clip.
    Confirmed in runs/prod02_smoke: clipped from step 100 onward.

    Seeding it at 4.71e+04 is NOT obviously the fix -- u* = v* = 0 is the
    correct answer at t = 0 -- but neither is leaving it pinned. The decision
    is upstream, in the choice of anchor. docs/open_items.md, Day 26."""
    from weighting import (GRADNORM_8X64_EPSPSI03,
                           GRADNORM_8X64_EPSPSI03_STEP500, PI_SEED_8X64)
    lo, hi = BalancerConfig().log_clip
    for snap in (GRADNORM_8X64_EPSPSI03, GRADNORM_8X64_EPSPSI03_STEP500):
        ga = snap["bc_mech"]
        for k in ("pde_richards", "pde_mech", "bc", "ic_head", "bc_mech"):
            c = (ga / snap[k]) / PI_SEED_8X64[k]
            assert lo <= math.log10(c) <= hi, f"{k} clipped at c = {c:.3e}"

    # pde_richards is the term the geometric mean was introduced for: it
    # straddles the regime change symmetrically instead of failing at one end.
    g0, g1 = GRADNORM_8X64_EPSPSI03, GRADNORM_8X64_EPSPSI03_STEP500
    c0 = (g0["bc_mech"] / g0["pde_richards"]) / PI_SEED_8X64["pde_richards"]
    c1 = (g1["bc_mech"] / g1["pde_richards"]) / PI_SEED_8X64["pde_richards"]
    assert math.log10(c0) == pytest.approx(-math.log10(c1), rel=1e-6)
    assert abs(math.log10(c0)) == pytest.approx(1.52, abs=0.01)

    ga = GRADNORM_8X64_EPSPSI03["bc_mech"]
    c_disp = (ga / GRADNORM_8X64_EPSPSI03["ic_disp"]) / PI_SEED_8X64["ic_disp"]
    assert math.log10(c_disp) > hi                  # known, tracked, not fixed
    assert c_disp == pytest.approx(4.71e4, rel=0.01)


def test_pi_seed_literals_match_nondim():
    """PI_SEED hardcodes the groups so weighting.py imports in a bare test
    env. E_ref and dtheta_ref are both open items; if either moves, the seed
    silently stops matching the scaling it was derived from."""
    from src.nondim import SCALES
    from src.weighting import _PI_M_BODY, _PI_R_GRAV
    assert _PI_M_BODY == pytest.approx(SCALES.Pi_M_body, rel=1e-4)
    assert _PI_R_GRAV == pytest.approx(SCALES.Pi_R_grav, rel=1e-4)


def test_the_measured_pde_seeds_are_not_reproducible_across_draws():
    """Day 26. The reason `--seed-source` defaults to `formula`.

    D-W.7 is only legitimate if w^ is a property of the objective rather than
    of the collocation draw. scripts/seed_variance.py measures it: the two PDE
    residual terms move by more than the entire +/-3 clip across three
    sampling seeds, while every boundary and IC term stays inside 3x. A seed
    taken from a single draw of the first group is fitting a heavy tail.

    This test pins the recorded numbers so the claim in the module docstring
    cannot quietly stop being true. It reads the artifact rather than
    re-measuring; regenerate with
        PYTHONPATH=. python scripts/seed_variance.py \\
            --json docs/seed_variance_day26.json
    """
    import json
    import pathlib
    p = (pathlib.Path(__file__).resolve().parents[1]
         / "docs" / "seed_variance_day26.json")
    spread = json.loads(p.read_text())["spread"]["1200"]
    lo, hi = BalancerConfig().log_clip
    reach = 10 ** hi              # the most the balancer can correct upward

    # Two draws of the same quantity disagree by more than the balancer can
    # correct, so which draw you seeded from decides whether you end up
    # clipped -- which is exactly what happened to `t0` and `geomean`.
    assert spread["pde_richards"] > reach, "the finding has changed"
    assert spread["pde_richards"] == pytest.approx(1264, rel=0.05)
    assert spread["pde_mech"] > 10

    for k in ("bc", "ic_head", "ic_disp", "bc_mech"):
        assert spread[k] < 5, f"{k} was stable across draws on Day 26"

    # ic_disp's 4.7e+04 is therefore a real requirement, not a sampling
    # artifact -- which is why it needs the anchor decision and not a seed.
    assert spread["ic_disp"] < 5
