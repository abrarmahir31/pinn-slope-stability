"""The log is the provenance record for every downstream number.

A `clipped` field was silently dropped once and nearly went unnoticed.
This pins the key set so it cannot happen to the frozen baseline.

log_c_raw and grad_norms carry only the currently-active terms, so their
membership legitimately shifts when a term freezes or un-freezes. Their
presence is asserted, their contents are not.
"""
import json
import pathlib

from scripts.train import main

GOLDEN = pathlib.Path(__file__).parent / "golden_log_keys.json"
VOLATILE = ("log_c_raw.", "grad_norms.")


def keyset(d, prefix=""):
    out = set()
    for k, v in d.items():
        out.add(prefix + k)
        if isinstance(v, dict):
            out |= keyset(v, prefix + k + ".")
    return out


def test_log_schema(tmp_path):
    main(["--epochs", "2", "--log-every", "1",
          "--n-pde", "50", "--n-bc", "50", "--n-iface", "50",
          "--out", str(tmp_path)])
    rec = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[0])

    got = {k for k in keyset(rec) if not k.startswith(VOLATILE)}
    want = {k for k in json.loads(GOLDEN.read_text())
            if not k.startswith(VOLATILE)}
    assert got == want, (
        f"missing: {sorted(want - got)}\nunexpected: {sorted(got - want)}"
    )
    assert "log_c_raw" in rec
    assert "grad_norms" in rec