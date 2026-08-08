
## Step 2.3 — Initial conditions (Day 13)

**Z_WT = 197.0 m a.s.l.** Karstic bedrock aquifer, Ulusay et al. (2014) §5.3, range
113–197 over six wells. Maximum taken because (i) Section 5 is the northern wall and
regional flow is N→SSW so heads are highest there, (ii) a higher table means less
suction and a lower F, i.e. conservative, (iii) it matches the F1 far-field Dirichlet
datum. Z_WT < Z_BASE = 200, so the domain is entirely unsaturated at t = 0 and the
saturated branch of psi_initial is provably empty (asserted, not assumed).
Alternative Z_WT = 113 shifts psi_0 by a rigid −84 m but leaves the gradient, and
therefore the driving force for infiltration, unchanged.

**No suction cap.** psi_0 runs −160.97 to −3.00 m (z_top_domain = 358.07). A cap at
−100 m would engage above z = 297 — the bench, all natural ground and the upper cut
face, i.e. exactly where the infiltration physics lives. K_r at −100 and −168 m are
both in the residual tail, so the cap moves the conditioning problem rather than
fixing it. Instead: floor K_r and non-dimensionalise psi. Revisit only if training
will not start.

**sigma_v_geostatic: analytic, not quadrature.** The trapz implementation carried
node-alignment jitter that varied with x and got *worse* on refinement (−278 Pa at
n=400, −324 Pa at n=1600). Every column is at most two layers because the Mk/Mk_d
divide is vertical, so the closed form is exact to machine precision. The contact is
clipped to z_ground: digitisation puts z_tm_top 0.14 m above ground at x = 0, which
would integrate a negative marl thickness at the toe.

**K0_SMOOTH_W = 2.0 m.** K0 is genuinely discontinuous at the Tm contact (0.429 →
0.333); sigma_v is continuous, sigma_h is not. A tanh network cannot represent a jump,
so K0 is blended linearly over ±2 m. Numerical device applied to a quantity that is
already an idealisation. The 22% jump becomes a 27.24 kPa ramp; ratio inside the band
spans 0.3341–0.4280, strictly interior, no overshoot.

**Known artefact: 50.07 kPa sigma_v stripe at x = X_MK_DIVIDE.** Density contrast
across the vertical Mk/Mk_d divide (Δρ = 204 kg/m³ over ~25 m of marl). The 1-D column
formula propagates it to the domain floor *undiminished* — measured at 50.07 kPa at
every depth from 300 down to 200 m, i.e. 12.83% of sigma_v at z=300 and 1.67% at
z=200. NOT smoothed: unlike K0 this is real geology, and it is the undiffused
propagation that is the artefact, not the contrast itself. Same root cause as the K0
column not being an equilibrium state (see below). Expected to diffuse over a
St. Venant distance under the gravity warm-up. **Acceptance test for Phase 3: confirm
the x = 90.7 stripe has diffused after the nil step.**

**K0 column is not in equilibrium.** Vertical equilibrium is satisfied identically;
horizontal is not, because sigma_h inherits the slope of the overburden. On the 31.06°
face the unbalanced horizontal body force is 0.20–0.26 × rho*g. Treat the K0 field as
a reference to be equilibrated, not as truth. Add a gravity warm-up stage to Step 3.4:
freeze psi = psi_0, train the mechanical outputs alone against div(sigma) + rho*g = 0,
initialised from the K0 field, then store the equilibrated sigma_0 as the lookup and
set u(t=0) = 0 from that state.

**Incremental stress convention.** sigma_total = sigma_0(lookup) + delta_sigma(u,v),
u = 0 at t = 0. Better conditioned than absolute. Consequence: sigma_0 is not only an
L_IC target — it enters the Bishop coupling (Step 3.3) and the SRF evaluation
(Phase 4) at every point and every time.

**ic_cache.npz.** 4000 points, seed 20260813, stratified Mk/Mk_d/Tm = 0.25/0.35/0.40
against area fractions 3.07/9.68/87.25%, with per-point weights so L_IC stays an
unbiased estimate of the domain integral. Geometry SHA recorded; gitignored as a
generated artefact.

**Open, escalated to Phase 3:** rainfall duration (20 mm/hr sustained for 30 days is
14.4 m of rain against ~1.2 m regional annual) and infiltration capacity
(q_rain/K_s(marl) = 5560, Green–Ampt ponding in ~11 min, so the ground-surface Neumann
is over-specified and needs the same complementarity treatment as the seepage face).

## E_ref = 1e9 Pa — retained (8 Aug)
The Phase-1 note "drop E_ref from 1 GPa to 100 MPa, no stratum is near 1 GPa"
is a SECTION 7 statement and does not apply here. Section 5 rock-mass moduli
(src/properties.py, authoritative):
    Mk 2.0895e8   Mk_d 3.8057e7   Tm 4.2639e9 Pa
Tm is 87.25% of the domain by area; the area-weighted mean is 3.73 GPa.

Candidates, as ratios E_unit/E_ref (the coefficient that actually appears in
the constitutive law) and the resulting U_ref = sig_ref*L_ref/E_ref:
    1.000e8   Mk 2.090  Mk_d 0.381  Tm 42.639   U_ref 1.700 m
    1.000e9   Mk 0.209  Mk_d 0.038  Tm  4.264   U_ref 0.170 m   <- ADOPTED
    4.264e9   Mk 0.049  Mk_d 0.009  Tm  1.000   U_ref 0.040 m

No choice makes all three O(1): the Tm/Mk_d stiffness contrast is 112x and is
geology, not a scaling freedom — structurally the same situation as L/H = 5.67
in the Richards groups. 1e9 gives the most balanced spread (geometric centre
0.40, nothing beyond 4.3) and is within 3.7x of the area-weighted mean.
4.264e9 optimises for the 87% unit but pushes Mk_d, the unit that FAILS, to
0.009. U_ref = 0.17 m against expected displacements of a few cm gives
u* ~ 0.25, which is the right order for tanh.

E_ref enters the physics ONLY through U_ref and hence the stiffness ratio.
It does NOT appear in Pi_M_body (rho_b_ref*g*L_ref/sig_ref) or Pi_M_couple
(rho_w*g*H_ref/sig_ref). Verified: sig_ref*L_ref/E_ref = 0.17 = SCALES.U_ref.

Also removed here: boundaries.py:130 held E_RM = {Mk 4.78e8, Mk_d 8.65e7,
Tm 4.264e9}, a pre-correction copy (both marls high by 2.286 = 400/175, the
MR revision). Dead constant, referenced nowhere. Deleted rather than fixed.
src/properties.py is the single source of truth for stiffness.
