"""Material properties for Section 5 (Tm / Mk / Mk_d), Isikdere.

Units: SI throughout. E in Pa, K_s in m/s, rho in kg/m3, alpha in 1/m.
Sign convention: psi < 0 = unsaturated (suction), psi >= 0 = saturated.

Mualem-van Genuchten constitutive model. Theta is effective saturation,
theta() returns volumetric water content.
"""
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

from src.nondim import SCALES

# Grain density used in the Phase 1 porosity derivation: n0 = 1 - rho_dry/(Gs*rho_w)
GS_RHO_W = 2650.0  # kg/m3  (Gs = 2.65)

_EPS = 1e-8        # suction floor: |psi| + _EPS avoids the singularity at psi = 0
_THETA_CLAMP = 1e-12   # keeps effective saturation strictly inside (0, 1)

_JSON = Path(__file__).resolve().parents[1] / "data" / "phase1_reference_state.json"


@dataclass(frozen=True)
class Material:
    tag: str            # "Tm" | "Mk" | "Mk_d"
    K_s: float          # m/s
    rho_dry: float      # kg/m3
    E: float            # Pa
    nu: float
    n0: float
    vg_alpha: float     # 1/m
    vg_n: float
    theta_s: float      # := n0 (see load_materials)
    theta_r: float
    sigma_ci: float     # Pa
    m_b: float
    s: float
    a: float
    provisional: bool
    provenance: str

    @property
    def vg_m(self) -> float:
        return 1.0 - 1.0 / self.vg_n

    @property
    def dtheta(self) -> float:
        return self.theta_s - self.theta_r


# Section 5 tag -> (key in phase1_reference_state.json, provenance note)
SECTION5_MAPPING = {
    "Mk":   ("marl_sekkoy",   "direct: Sekkoy marl"),
    "Mk_d": ("marl_weakzone", "PROXY: deformed member represented by weak-zone marl"),
    "Tm":   (None,            "PLACEHOLDER: no Turgut Fm. entry in Phase 1 data"),
}

# Invented. Fractured crystallized limestone, order-of-magnitude only.
TM_PLACEHOLDER = Material(
    tag="Tm",
    K_s=1e-5, rho_dry=2400.0, E=20.0e9, nu=0.25, n0=0.05,
    vg_alpha=1.0, vg_n=2.0, theta_s=0.05, theta_r=0.01,
    sigma_ci=60.0e6, m_b=2.0, s=1e-3, a=0.5,
    provisional=True,
    provenance="PLACEHOLDER - no source. Replace before any reported result.",
)


def load_materials(path: Path = _JSON) -> dict:
    """Build the Section 5 material set from the Phase 1 reference state.

    Conversions and overrides applied here (all deliberate):
      - E_MPa -> Pa
      - theta_s := n0   (the JSON's literature theta_s = 0.38 disagrees with
        the derived porosity 0.427; the derived value is traceable)
      - n0 is cross-checked against 1 - rho_dry/2650 and a warning raised
        on disagreement > 1%
    """
    strata = json.loads(path.read_text())["strata"]
    out = {}

    for tag, (key, provenance) in SECTION5_MAPPING.items():
        if key is None:
            out[tag] = TM_PLACEHOLDER
            continue

        r = strata[key]
        n0_file = float(r["n0"])
        n0_calc = 1.0 - float(r["rho_dry"]) / GS_RHO_W
        if abs(n0_calc - n0_file) > 0.01:
            warnings.warn(
                f"{tag}: n0 in file = {n0_file:.3f} but rho_dry/{GS_RHO_W:.0f} "
                f"implies {n0_calc:.3f}. Using file value; see open_items.md.",
                stacklevel=2,
            )

        out[tag] = Material(
            tag=tag,
            K_s=float(r["K_s_mps"]),
            rho_dry=float(r["rho_dry"]),
            E=float(r["E_MPa"]) * 1e6,
            nu=float(r["nu"]),
            n0=n0_file,
            vg_alpha=float(r["vg_alpha_1pm"]),
            vg_n=float(r["vg_n"]),
            theta_s=n0_file,
            theta_r=float(r["vg_theta_r"]),
            sigma_ci=float(r["sigma_ci_Pa"]),
            m_b=float(r["m_b"]),
            s=float(r["s"]),
            a=float(r["a"]),
            provisional=False,
            provenance=provenance,
        )

    for m in out.values():
        if m.provisional:
            warnings.warn(f"{m.tag} is PROVISIONAL: {m.provenance}", stacklevel=2)

    return out


def _effective_saturation(psi: torch.Tensor, mat: Material) -> torch.Tensor:
    """Theta(psi) for the unsaturated branch, clamped strictly inside (0, 1)."""
    S = mat.vg_alpha * (psi.abs() + _EPS)
    Theta = (1.0 + S ** mat.vg_n) ** (-mat.vg_m)
    return Theta.clamp(_THETA_CLAMP, 1.0 - _THETA_CLAMP)


def theta(psi: torch.Tensor, mat: Material) -> torch.Tensor:
    """Volumetric water content [-]."""
    unsat = mat.theta_r + mat.dtheta * _effective_saturation(psi, mat)
    return torch.where(psi < 0, unsat, torch.full_like(psi, mat.theta_s))


def K(psi: torch.Tensor, mat: Material) -> torch.Tensor:
    """Unsaturated hydraulic conductivity [m/s], Mualem."""
    Theta = _effective_saturation(psi, mat)
    inner = 1.0 - (1.0 - Theta ** (1.0 / mat.vg_m)) ** mat.vg_m
    unsat = mat.K_s * Theta ** 0.5 * inner ** 2
    return torch.where(psi < 0, unsat, torch.full_like(psi, mat.K_s))



def C(psi: torch.Tensor, mat: Material) -> torch.Tensor:
 """Dimensionless specific moisture capacity, C* = (dtheta/dpsi) * H_ref / dtheta_ref.
    dtheta/dpsi = dtheta * alpha * m * n * S**(n-1) * (1 + S**n)**(-m-1),  S = alpha*|psi|
    Positive by construction: the minus from dTheta/dS cancels the minus from
    dS/dpsi = -alpha (valid for psi < 0, where |psi| = -psi). Zero once saturated."""

 S = mat.vg_alpha * (psi.abs() + _EPS)
 n, m = mat.vg_n, mat.vg_m
 dtheta_dpsi = (
        mat.dtheta * mat.vg_alpha * m * n
        * S ** (n - 1.0)
        * (1.0 + S ** n) ** (-m - 1.0)
    )
   
   
 unsat = dtheta_dpsi * SCALES.H_ref / SCALES.dtheta_ref
 return torch.where(psi < 0, unsat, torch.zeros_like(psi))