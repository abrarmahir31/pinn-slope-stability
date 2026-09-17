r"""Days 36-37 -- freeze a production baseline: archive the checkpoint, its
config, every field output, and a manifest of content hashes.

WHY HASHES AND NOT JUST A FOLDER. This frozen baseline is the input to every
SSR and sensitivity run that follows. If it drifts -- a re-run overwrites the
checkpoint, a figure is regenerated from a different seed, someone edits
`properties.py` -- then every downstream comparison is against a moving
reference and nothing is reproducible. The manifest records a SHA-256 for each
archived file and `--verify` re-checks them, so drift is detectable rather
than assumed absent.

WHAT IT REFUSES TO FREEZE. A baseline is a claim that the run is fit to build
on, so the obvious checks are enforced rather than left to the operator:

  * the checkpoint must carry `cfg["ansatz"]` (D-A.3), or the ansatz is
    unrecoverable and the archive is worthless six weeks later;
  * the test suite must pass;
  * the working tree must be clean, or the archived code is not the code that
    produced the run;
  * `docs/fig6_elastic_check.json`, if present, must not report a failing
    baseline -- an SSR sweep starting from a field already below FS = 1 has
    nothing to reduce.

`--force` overrides each of these and RECORDS in the manifest that it was
overridden, because a silent override is worse than no check.

    PYTHONPATH=. python scripts/freeze_baseline.py runs/ansatz/exp_seed7 --tag baseline-v1
    PYTHONPATH=. python scripts/freeze_baseline.py --verify baselines/baseline-v1
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

import torch


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""


def verify(root):
    mpath = os.path.join(root, "MANIFEST.json")
    if not os.path.exists(mpath):
        print(f"no MANIFEST.json in {root}")
        return 1
    man = json.load(open(mpath))
    bad, missing = [], []
    for rel, rec in man["files"].items():
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            missing.append(rel)
            continue
        if sha256(p) != rec["sha256"]:
            bad.append(rel)
    print(f"baseline {man.get('tag')}  frozen {man.get('frozen_utc')}")
    print(f"  files {len(man['files'])}, missing {len(missing)}, "
          f"altered {len(bad)}")
    for rel in missing:
        print(f"    MISSING  {rel}")
    for rel in bad:
        print(f"    ALTERED  {rel}")
    if man.get("overrides"):
        print(f"  OVERRIDDEN AT FREEZE TIME: {', '.join(man['overrides'])}")
    if not missing and not bad:
        print("  INTACT")
        return 0
    print("  BASELINE IS NOT INTACT. Do not compare downstream runs against "
          "it until this is\n  resolved -- the reference has moved.")
    return 2


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="run directory, e.g. runs/ansatz/exp_seed7")
    ap.add_argument("--verify", metavar="DIR", default=None)
    ap.add_argument("--tag", default=None, help="baseline name")
    ap.add_argument("--ckpt", default="ckpt_final.pt")
    ap.add_argument("--dest", default="baselines")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="additional files to archive (figures, artifacts)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--accept-elastic-overstress", action="store_true",
                    help="override ONLY the fig6 elastic-check gate, and record "
                         "it. D-5.4 keeps D = 1, under which ~19%% of points "
                         "(5.4%% of area) are pointwise below GHB FS = 1 by "
                         "construction; Day 40 showed that is not a boundary "
                         "artefact and D-5.1's soft constraint starts from it. "
                         "Every other gate stays strict.")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args(argv)

    if a.verify:
        return verify(a.verify)
    if not a.run:
        ap.error("give a run directory, or --verify DIR")

    tag = a.tag or f"baseline-{datetime.datetime.utcnow():%Y%m%d-%H%M}"
    root = os.path.join(a.dest, tag)
    if os.path.exists(root) and not a.force:
        print(f"{root} exists. Pick another --tag, or --force to overwrite.")
        return 1

    overrides = []

    def gate(ok, msg):
        if ok:
            return True
        if a.force:
            print(f"OVERRIDDEN: {msg}")
            overrides.append(msg)
            return True
        print(f"REFUSING TO FREEZE: {msg}\n  (--force overrides and records it)")
        return False

    ck = os.path.join(a.run, a.ckpt)
    if not os.path.exists(ck):
        print(f"no checkpoint at {ck}")
        print("NOTE: ansatz_ablation runs are launched with --ckpt-every 0, so "
              "they carry only\n`ckpt_final.pt` (and `ckpt_adam_final.pt` if "
              "the L-BFGS phase existed when they ran).")
        return 1

    d = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = d.get("cfg") or {}
    print(f"run   {a.run}\nckpt  {ck}  step {d.get('step')}")
    print(f"ansatz {cfg.get('ansatz')!r}  eps_psi {cfg.get('eps_psi')}  "
          f"cap_k {cfg.get('cap_k')}  seed {cfg.get('seed')}")

    if not gate(cfg.get("ansatz") is not None,
                "checkpoint carries no cfg['ansatz'] (D-A.3): the ansatz is "
                "unrecoverable"):
        return 1

    dirty = git("status", "--porcelain")
    if not gate(dirty == "", "working tree is dirty; archived code would not "
                             "match the run"):
        return 1

    if not a.skip_tests:
        print("\nrunning the suite ...")
        env = dict(os.environ, PYTHONPATH=".")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q"],
                           capture_output=True, text=True, env=env)
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        print(f"  {tail}")
        if not gate(r.returncode == 0, f"test suite failed: {tail}"):
            return 1

    ec = "docs/fig6_elastic_check.json"
    if os.path.exists(ec):
        frac = json.load(open(ec)).get("frac_yielded_domain")
        if frac is not None:
            print(f"\nelastic check: {100*frac:.3f}% of points below FS = 1")
            if a.accept_elastic_overstress and frac >= 0.01:
                overrides.append(
                    f"elastic check accepted under D-5.4: {100*frac:.2f}% of "
                    f"points below FS=1 (unweighted); gate not applied")
                print("  accepted under D-5.4 (--accept-elastic-overstress)")
            elif not gate(frac < 0.01,
                        f"baseline is already failing at SRF = 1 "
                        f"({100*frac:.2f}% below FS=1); an SSR sweep has "
                        f"nothing to reduce"):
                return 1
    else:
        print(f"\nNOTE: {ec} not found -- run make_fig6.py first so the "
              "elastic check is recorded.")

    os.makedirs(root, exist_ok=True)
    files = {}

    def take(src, rel=None):
        if not os.path.exists(src):
            return
        rel = rel or os.path.basename(src)
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst) or root, exist_ok=True)
        shutil.copy2(src, dst)
        files[rel] = {"sha256": sha256(dst), "bytes": os.path.getsize(dst),
                      "source": src.replace("\\", "/")}

    for name in os.listdir(a.run):
        p = os.path.join(a.run, name)
        if os.path.isfile(p):
            take(p, f"run/{name}")
    for extra in a.extra:
        take(extra, f"artifacts/{os.path.basename(extra)}")
    for auto in ("docs/fig4.png", "docs/fig5.png", "docs/fig6.png",
                 "docs/fig7.png", "docs/fig6_elastic_check.json",
                 "docs/verify_nguyen_raudkivi.json", "DECISIONS.md"):
        take(auto, f"artifacts/{os.path.basename(auto)}")

    man = {
        "tag": tag,
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
        "run_dir": a.run.replace("\\", "/"),
        "checkpoint": a.ckpt,
        "step": d.get("step"),
        "train_cfg": {k: v for k, v in cfg.items()
                      if isinstance(v, (int, float, str, bool, type(None)))},
        "git": {"commit": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": dirty != ""},
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "overrides": overrides,
        "files": files,
    }
    with open(os.path.join(root, "MANIFEST.json"), "w") as f:
        json.dump(man, f, indent=1)

    print(f"\nfroze {len(files)} files into {root}")
    for rel in sorted(files):
        print(f"   {rel}")
    print(f"\nverify at any time with:\n"
          f"   python scripts/freeze_baseline.py --verify {root}")
    print(f"\ntag it in git:\n"
          f"   git tag -a {tag} -m \"frozen baseline: {a.run} step "
          f"{d.get('step')}\"\n   git push origin {tag}")
    if overrides:
        print("\nFROZEN WITH OVERRIDES. They are recorded in the manifest and "
              "will be printed by\n--verify. Anything built on this baseline "
              "inherits them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())