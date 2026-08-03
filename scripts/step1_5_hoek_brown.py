"""
Step 1.5 - Hoek-Brown rock-mass parameters (m_b, s, a) and Mohr-Coulomb
equivalents (c, phi) for the Isikdere pit lithologies.

Source of inputs: Ulusay et al. (2014), Eng. Geol. 181, Table 3b.
Reduction equations: Hoek, Carranza-Torres & Corkum (2002).
MC-equivalent fit: Hoek et al. (2002) closed-form over 0 <= sig3' <= sig3max.

All units: run with `python step1_5_hoek_brown.py`.
"""
import math

D = 1.0  # disturbance factor: production blasting, large open pit (all units)

# name, GSI, sigma_ci [Pa], m_i
UNITS = [
    ("Marl (Sekkoy Fm.)",          50, 17.9e6, 4),
    ("Weak zone (marl, N slope)",  45,  4.29e6, 3),
    ("Coal seam",                  40,  7.80e6, 18),
    ("Tuffite (Yatagan Fm.)",      26,  3.59e6, 15),
]


def hoek_brown(GSI, m_i, D=1.0):
    """Return reduced m_b, s, a (Hoek et al. 2002)."""
    m_b = m_i * math.exp((GSI - 100) / (28 - 14 * D))
    s   = math.exp((GSI - 100) / (9 - 3 * D))
    a   = 0.5 + (1.0 / 6.0) * (math.exp(-GSI / 15) - math.exp(-20.0 / 3.0))
    return m_b, s, a


def mc_equivalent(GSI, m_i, sigma_ci, sig3max, D=1.0):
    """Return equivalent (c [Pa], phi [deg]) via Hoek 2002 closed-form fit
    over the range 0 <= sig3' <= sig3max."""
    m_b, s, a = hoek_brown(GSI, m_i, D)
    s3n = sig3max / sigma_ci
    term = (s + m_b * s3n) ** (a - 1)
    phi = math.asin((6 * a * m_b * term) /
                    (2 * (1 + a) * (2 + a) + 6 * a * m_b * term))
    coh = (sigma_ci * ((1 + 2 * a) * s + (1 - a) * m_b * s3n) * term) / \
          ((1 + a) * (2 + a) *
           math.sqrt(1 + (6 * a * m_b * term) / ((1 + a) * (2 + a))))
    return coh, math.degrees(phi)


if __name__ == "__main__":
    print("=== Hoek-Brown reduced parameters (D = 1) ===")
    print(f"{'Unit':28} {'GSI':>4} {'sci(MPa)':>9} {'m_i':>4} "
          f"{'m_b':>9} {'s':>11} {'a':>8}")
    for name, GSI, sci, m_i in UNITS:
        m_b, s, a = hoek_brown(GSI, m_i, D)
        print(f"{name:28} {GSI:4d} {sci/1e6:9.2f} {m_i:4d} "
              f"{m_b:9.4f} {s:11.3e} {a:8.4f}")

    print("\n=== MC equivalents (c, phi) vs. confining range sig3max ===")
    for tag, s3 in [("Full slope  (0.25*gamma*H ~ 850 kPa)", 850e3),
                    ("Mid         (~400 kPa)",               400e3),
                    ("Low         (~100 kPa)",               100e3),
                    ("Very low    (~25 kPa, near face)",      25e3)]:
        print(f"\n-- {tag} --")
        for name, GSI, sci, m_i in UNITS:
            c, phi = mc_equivalent(GSI, m_i, sci, s3, D)
            print(f"   {name:28} c = {c/1e3:6.1f} kPa   phi = {phi:5.1f} deg")
