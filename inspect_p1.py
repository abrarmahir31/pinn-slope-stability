import json
D = json.load(open('Phase 1/isikdere_phase1_dataset.json'))
S = D['strata']
for k in ('marl_sekkoy', 'marl_weakzone'):
    s = S[k]
    print(f"=== {k} ===")
    for f in ('E_rm_Pa','E_Pa','E','nu','poisson','n0','porosity',
              'K_s_m_s','rho_dry','rho_sat','alpha_1_m','vg_alpha',
              'vg_n','theta_s','theta_r','sigma_ci_Pa','GSI','m_i'):
        if f in s: print(f"   {f:14s} {s[f]}")
    miss = [f for f in ('E_rm_Pa','nu','n0') if f not in s]
    if miss: print("   keys not found under those names; full list:", list(s.keys()))
print("\n=== conflicts_resolved ===")
print(json.dumps(D['conflicts_resolved'], indent=1)[:2000])
print("\n=== open_items ===")
print(json.dumps(D['open_items'], indent=1)[:1200])
