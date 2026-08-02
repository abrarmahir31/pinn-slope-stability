
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
