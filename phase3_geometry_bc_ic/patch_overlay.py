import re, pathlib
p = pathlib.Path("14_overlay.py")
s = p.read_text()

new = '''def marl_band_diagnostic():
    """Locate where the marl band thickness goes negative.

    Top-of-Tm is traced in TWO pieces: d_mk_tm.csv (Mk on Tm, x 3.8-90.2) and
    d_mkd_base.csv (Mk_d on Tm, x 95.1-266.1). There is a 4.9 m gap between
    them and X_MK_DIVIDE = 90.7 sits inside it.
    """
    print("\\n--- marl band thickness scan ---")
    pat, hits = find_csvs()
    names = {os.path.basename(h).lower(): h for h in hits}
    left   = next((v for k, v in names.items() if "mk_tm"    in k), None)
    right  = next((v for k, v in names.items() if "mkd_base" in k), None)
    ground = next((v for k, v in names.items() if "ground"   in k), None)
    if not (left and right and ground):
        print("  contacts not identified; skipping.")
        return None
    L, R = load_contact(left), load_contact(right)
    a = np.vstack([L, R])
    a = a[np.argsort(a[:, 0])]
    print(f"  joined top-of-Tm: {len(a)} pts, "
          f"gap {L[:, 0].max():.1f} -> {R[:, 0].min():.1f} m "
          f"(X_MK_DIVIDE = {G.X_MK_DIVIDE})")
    xs = np.linspace(a[:, 0].min(), a[:, 0].max(), 2000)
    ztm = np.interp(xs, a[:, 0], a[:, 1])
    b = load_contact(ground)
    zg = np.interp(xs, b[:, 0], b[:, 1])
    t = zg - ztm
    i = int(np.argmin(t))
    print(f"  min thickness {t.min():+.3f} m at x = {xs[i]:.1f} m")
    if t.min() < 0:
        bad = t < 0
        print(f"  NEGATIVE from x = {xs[bad].min():.1f} to {xs[bad].max():.1f} m")
        inside = L[:, 0].max() <= xs[i] <= R[:, 0].min()
        print("  -> IN THE TRACE GAP" if inside else "  -> outside the gap: real")
        for cand in (90.7, 92.65, 95.1):
            zc = np.interp(cand, a[:, 0], a[:, 1])
            print(f"     X_MK_DIVIDE = {cand:6.2f} -> top-of-Tm z = {zc:.2f} m")
        return float(xs[i])
    return None
'''

m = re.search(r"def marl_band_diagnostic\(\):.*?(?=\n# ---|\ndef )", s, re.S)
if not m:
    raise SystemExit("anchor not found -- is this the right 14_overlay.py?")
p.write_text(s[:m.start()] + new + s[m.end():])
print("patched marl_band_diagnostic()")
