"""Is float64 necessary, or does the Richards residual survive float32?

The concern is not the hydrostatic IC -- that is smooth and well-scaled
and passes trivially. It is the residual, which differentiates (psi + z),
a difference of quantities of order 100 m. float32 carries ~7 significant
digits and that cancellation eats them before the derivative is taken.

Measures the per-point residual R, not the scalar loss: mean(w*R^2) over
10k points averages the cancellation noise away and would read clean even
when the field is not.
"""
import copy
import torch
from src.loss import _psi_field, _kc_factor
from src.coupling import KC_DEFAULT
from src.materials import load_materials
from src.nondim import SCALES
from grad_norm_table import build          # gives net, loss_fn, coll

net, loss_fn, coll = build(N=10000)
mats = load_materials()

def residual_for(dtype):
    n = copy.deepcopy(net).to(dtype)
    c_x = coll.x.to(dtype); c_z = coll.z.to(dtype); c_t = coll.t.to(dtype)
    fields = _psi_field(n)
    out = {}
    for tag in sorted(set(coll.tag.tolist())):
        m = torch.as_tensor(coll.tag == tag, device=coll.x.device)
        xs, zs, ts = c_x[m], c_z[m], c_t[m]
        print(f"  {tag}: xs={xs.dtype} p0={next(n.parameters()).dtype}")
        from src.loss import richards_residual      # adjust if elsewhere
        R = richards_residual(fields, xs, zs, ts, mats[tag],
                              s=SCALES, normalise="none",
                              k_factor=_kc_factor(n, xs, zs, ts, mats[tag],
                                                  KC_DEFAULT, tag, SCALES))
        out[tag] = R.detach().double()
    return out

r64 = residual_for(torch.float64)
r32 = residual_for(torch.float32)

for tag in r64:
    a, b = r64[tag], r32[tag]
    print(f"{tag:>5}  dtype32={b.dtype}  n={a.numel()}")   # sanity: must be from float32 path
    rel = (a - b).abs() / (a.abs() + 1e-300)
    print(f"        median rel err {rel.median():.3e}   max {rel.max():.3e}")
    print(f"        |R| median 64 {a.abs().median():.3e}   32 {b.abs().median():.3e}")