r"""Every remaining GPU run, unattended, then the result data. ONE COMMAND.

HISTORICAL (27 Sep): this driver made the runs of 25-26 Sep with the D-5.13
three-condition criterion and each run's own loss reference. D-5.18 dropped
the displacement condition and D-5.19 fixed the reference for physics arms;
the reportable numbers come from scripts/final_results.py. COMMON is left as
it was so the record of how those runs were made stays true.

    scripts\run_remaining.bat                 (Miniforge Prompt, env active)
    python scripts/run_remaining.py --dry-run  print the plan, run nothing

What it does, in order:

 1. WAIT until the GPU is free (the scaled N_PDE chain started by hand on
    24 Sep). It polls nvidia-smi every 5 min and moves on after three idle
    polls in a row, so the gap between the chained 5k and 20k runs cannot
    fool it. --no-wait skips this.
 2. pytest -q. A failure is logged and flagged, not fatal.
 3. make_arms.py with the D-6.1 settings (Mk_d K_s x0.1/x10, Mk_d GSI 35/55,
    coupling off/all; rainfall arm dropped, O-10).
 4. The sweeps, most decision-free first:
      npde      ssr_ghb_n5000_wscaled, ssr_ghb_n20000_wscaled (D-5.17 scaled
                w_yield). Skipped when done; RESUMED if the hand-started chain
                died part-way (same flags, so the fingerprint matches).
      coupling  KC off (one-way) and KC everywhere            -> Fig 11
      ks        Mk_d K_s x0.1 and x10                          -> Fig 12
      gsi       Mk_d GSI 35 and 55 (sigma_0 NOT re-solved, O-11) -> Fig 12
      draws     collocation-draw replicates, seeds 11-14 (O-12)
    A finished run (result.json present) is skipped. A half-finished one is
    resumed by ssr_sweep.py itself. A failed run is logged and the queue
    carries on, so one bad arm does not cost the night.
 5. collect_results.py (fos_table), recompute_criterion.py (O-22 table),
    copies every run's result.json + sweep.jsonl into docs/results (runs/ is
    gitignored), and OVERNIGHT_SUMMARY.md. No figures are drawn.

Stopping and restarting is safe: rerun the same command and it picks up where
it left off. Every run uses the calibrated criterion of D-5.13 exactly as the
Phase 5 sweeps did, on baseline-v2, and the script refuses to start if the
baseline checkpoint's hash is not baseline-v2's.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join("runs", "ansatz_epsuv1e-2", "ckpt_final.pt")
BASELINE_SHA_PREFIX = "3345c50c"                 # baseline-v2 (D-5.13 note)
N_REF = 10000                                    # D-5.17: w_yield x N_REF / N_PDE

# Exactly the Phase 5 / run_phase6.bat RUN flags. Changing any of these makes
# the new runs incomparable with ssr_ghb_w1, and breaks resume of the N_PDE
# chain (its fingerprint was written with these values).
COMMON = ["--criterion", "GHB", "--sig3-lo", "0", "--sig3-hi", "179000",
          "--baseline", BASELINE, "--yield-norm", "raw",
          "--admissible-metric", "min_tag", "--admissible-mode", "floor",
          "--admissible-floor", "0.50", "--plateau-factor", "5",
          "--disp-factor", "5", "--srf-step", "0.25", "--bisect-tol", "0.01",
          "--srf-max", "4.0"]
W_REPORT = "1"                                   # D-5.8

# D-6.1 (proposed 24 Sep, supervisor to ratify): both arms on Mk_d.
ARMS_DIR = os.path.join("configs", "arms")
MAKE_ARMS = ["--ks-stratum", "Mk_d", "--gsi-stratum", "Mk_d",
             "--gsi-levels", "35,45,55", "--ks-factors", "0.1,10",
             "--out", ARMS_DIR]
DRAW_SEEDS = (11, 12, 13, 14)
GROUPS = ("npde", "coupling", "ks", "gsi", "draws")
HOURS = {"npde5000": 1.3, "npde20000": 4.5, "default": 2.2}

POLL_S = 300
IDLE_POLLS = 3


# --------------------------------------------------------------------------- plan
def plan() -> list[dict]:
    """The queue, in run order. Each job: name, group, extra args, hours."""
    jobs = []
    for n in (5000, 20000):
        w = N_REF / n
        jobs.append({"name": f"ssr_ghb_n{n}_wscaled", "group": "npde",
                     "args": ["--n-pde", str(n), "--w-yield", f"{w:g}"],
                     "hours": HOURS[f"npde{n}"]})
    for group, stems in (("coupling", ("coupling_low", "coupling_high")),
                         ("ks", ("Mk_d_K_s_low", "Mk_d_K_s_high")),
                         ("gsi", ("Mk_d_GSI_low", "Mk_d_GSI_high"))):
        for stem in stems:
            jobs.append({"name": f"ssr_arm_{stem}", "group": group,
                         "args": ["--w-yield", W_REPORT, "--overrides",
                                  os.path.join(ARMS_DIR, stem + ".json")],
                         "hours": HOURS["default"]})
    for s in DRAW_SEEDS:
        jobs.append({"name": f"ssr_ghb_draw{s}", "group": "draws",
                     "args": ["--w-yield", W_REPORT, "--sampling-seed", str(s)],
                     "hours": HOURS["default"]})
    return jobs


def command(job: dict) -> list[str]:
    return ([sys.executable, "-u", os.path.join("scripts", "ssr_sweep.py")]
            + COMMON + job["args"] + ["--out", os.path.join("runs", job["name"])])


def result_path(job: dict) -> str:
    return os.path.join("runs", job["name"], "result.json")


def read_result(job: dict) -> dict | None:
    p = result_path(job)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- log
class Log:
    def __init__(self, path: str | None):
        self.f = None
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str = "", stamp: bool = True):
        line = (f"[{dt.datetime.now():%Y-%m-%d %H:%M}] {msg}" if stamp and msg
                else msg)
        print(line, flush=True)
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()


# --------------------------------------------------------------------------- GPU
def parse_compute_apps(text: str) -> list[str]:
    """nvidia-smi --query-compute-apps=pid,process_name output -> python rows."""
    rows = [r.strip() for r in text.splitlines() if r.strip()]
    return [r for r in rows if "python" in r.lower()]


def gpu_busy() -> bool | None:
    """True/False from nvidia-smi; None if nvidia-smi cannot answer."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return bool(parse_compute_apps(out.stdout))


def recently_written(paths, minutes: float) -> bool:
    now = time.time()
    return any(os.path.exists(p) and now - os.path.getmtime(p) < minutes * 60
               for p in paths)


def wait_for_gpu(log: Log, jobs: list[dict], max_hours: float) -> bool:
    """Block until the GPU has been idle for IDLE_POLLS polls in a row.
    Returns False if max_hours ran out first."""
    npde = [j for j in jobs if j["group"] == "npde"]
    watch = [os.path.join("runs", j["name"], "sweep.jsonl") for j in npde]
    t_end = time.time() + max_hours * 3600
    idle = 0
    log(f"waiting for the GPU to be free (poll {POLL_S // 60} min, "
        f"{IDLE_POLLS} idle polls in a row, give up after {max_hours:g} h)")
    while time.time() < t_end:
        busy = gpu_busy()
        if busy is None:                       # nvidia-smi unavailable
            busy = recently_written(watch, 90)
        idle = 0 if busy else idle + 1
        prog = ", ".join(
            f"{j['name']}: " + ("done" if read_result(j) else
                               f"{_n_lines(os.path.join('runs', j['name'], 'sweep.jsonl'))} SRF pts")
            for j in npde)
        log(f"GPU {'busy' if busy else f'idle ({idle}/{IDLE_POLLS})'}  |  {prog}")
        if idle >= IDLE_POLLS:
            return True
        time.sleep(POLL_S)
    return False


def _n_lines(p: str) -> int:
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


# --------------------------------------------------------------------------- run
def run_streamed(cmd: list[str], log: Log, console_log: str | None = None,
                 timeout_h: float | None = None) -> int:
    """Run `cmd`, echoing every line to the screen, the main log and
    `console_log`. Returns the exit code."""
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONUNBUFFERED="1",
               PYTHONIOENCODING="utf-8")
    cf = None
    if console_log:
        os.makedirs(os.path.dirname(console_log), exist_ok=True)
        cf = open(console_log, "a", encoding="utf-8")
        cf.write(f"\n===== {dt.datetime.now():%Y-%m-%d %H:%M} "
                 f"{' '.join(cmd)}\n")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace",
                         bufsize=1, env=env, cwd=ROOT)
    t0 = time.time()
    try:
        for line in p.stdout:
            line = line.rstrip("\n")
            log(line, stamp=False)
            if cf:
                cf.write(line + "\n")
                cf.flush()
            if timeout_h and time.time() - t0 > timeout_h * 3600:
                p.kill()
                log(f"TIMEOUT after {timeout_h:g} h -- killed")
                break
        return p.wait()
    except KeyboardInterrupt:
        p.terminate()
        p.wait()
        raise
    finally:
        if cf:
            cf.close()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def keep_awake(on: bool):
    """Stop Windows sleeping while this runs. No-op elsewhere."""
    if os.name != "nt":
        return
    try:
        import ctypes
        flags = 0x80000000 | (0x00000001 if on else 0)   # CONTINUOUS|SYSTEM
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


# --------------------------------------------------------------------------- post
def copy_results(log: Log, dest: str = os.path.join("docs", "results")) -> int:
    """runs/ssr*/{result.json,sweep.jsonl} -> docs/results/<run>_*. runs/ is
    gitignored, so this is how the numbers reach git."""
    n = 0
    os.makedirs(dest, exist_ok=True)
    for d in sorted(glob.glob(os.path.join("runs", "ssr*"))):
        name = os.path.basename(d)
        rj = os.path.join(d, "result.json")
        if "ABORTED" in name or not os.path.exists(rj):
            continue
        try:
            with open(rj, encoding="utf-8") as f:
                if json.load(f).get("smoke"):
                    continue
        except (OSError, json.JSONDecodeError):
            continue
        for src, suffix in ((rj, "_result.json"),
                            (os.path.join(d, "sweep.jsonl"), "_sweep.jsonl")):
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(dest, name + suffix))
                n += 1
    log(f"copied {n} files into {dest}")
    return n


def summary_md(rows: list[dict], tests: str, started: dt.datetime) -> str:
    L = ["# Overnight run summary (scripts/run_remaining.py)", "",
         f"Started {started:%Y-%m-%d %H:%M}, finished "
         f"{dt.datetime.now():%Y-%m-%d %H:%M}. Baseline-v2 "
         f"(`{BASELINE_SHA_PREFIX}…`), calibrated criterion D-5.13, "
         f"w_yield {W_REPORT} (N_PDE runs scaled per D-5.17).", "",
         f"pytest: {tests}", "",
         "| run | group | status | FOS | bracket | hours | note |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        fos = "—" if r.get("fos") is None else f"{r['fos']:.3f}"
        L.append(f"| {r['name']} | {r['group']} | {r['status']} | {fos} | "
                 f"{r.get('bracket') or '—'} | {r.get('hours', '—')} | "
                 f"{r.get('note', '')} |")
    L += ["", "Tables: `fos_table.md` (plateau, tornado, coupling, replicates, "
          "N_PDE at matched effective w_yield) and `criterion_o22.md` (every "
          "run under both failure criteria).", "",
          "GSI arms ran against the baseline sigma_0 (O-11): state it as a "
          "limitation, or re-solve sigma_0 and rerun those two arms.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the commands, run nothing")
    ap.add_argument("--no-wait", action="store_true",
                    help="do not wait for the GPU to be free first")
    ap.add_argument("--max-wait-h", type=float, default=12.0)
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--groups", default=",".join(GROUPS),
                    help=f"subset of {','.join(GROUPS)}")
    ap.add_argument("--job-timeout-h", type=float, default=10.0,
                    help="kill a single sweep that runs longer than this")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    bad = set(groups) - set(GROUPS)
    if bad:
        ap.error(f"unknown group(s) {sorted(bad)}; choose from {GROUPS}")
    jobs = [j for j in plan() if j["group"] in groups]

    if a.dry_run:
        print(f"repo: {ROOT}\n")
        print("make_arms: python scripts/make_arms.py " + " ".join(MAKE_ARMS))
        todo = 0.0
        for j in jobs:
            done = read_result(j) is not None
            todo += 0 if done else j["hours"]
            state = "DONE, skipped" if done else f"~{j['hours']:g} h"
            print(f"\n[{j['group']}] {j['name']}  {state}\n  "
                  + " ".join(command(j)))
        print(f"\nremaining GPU time ~{todo:.0f} h")
        return 0

    started = dt.datetime.now()
    log = Log(os.path.join("runs", "_overnight",
                           f"overnight_{started:%Y%m%d_%H%M}.log"))
    log(f"run_remaining.py in {ROOT}; groups {groups}")

    if not os.path.exists(BASELINE):
        log(f"STOP: {BASELINE} not found")
        return 2
    sha = sha256(BASELINE)
    if not sha.startswith(BASELINE_SHA_PREFIX):
        log(f"STOP: {BASELINE} sha256 {sha[:12]}… is not baseline-v2 "
            f"({BASELINE_SHA_PREFIX}…). Nothing run.")
        return 2
    log(f"baseline-v2 verified ({sha[:12]}…)")

    keep_awake(True)
    rows, tests = [], "skipped"
    try:
        waited_ok = True
        if not a.no_wait:
            waited_ok = wait_for_gpu(log, jobs, a.max_wait_h)
            if not waited_ok:
                log("GPU never became free within --max-wait-h. The N_PDE "
                    "runs are skipped (another process may still be writing "
                    "them); everything else goes ahead.")

        if not a.skip_tests:
            log("pytest -q ...")
            rc = run_streamed([sys.executable, "-m", "pytest", "-q",
                               "-p", "no:cacheprovider"], log,
                              os.path.join("runs", "_overnight", "pytest.log"),
                              timeout_h=1.0)
            tests = "passed" if rc == 0 else f"FAILED (exit {rc}) -- see runs/_overnight/pytest.log"
            log(f"pytest {tests}")

        if any(j["group"] in ("coupling", "ks", "gsi") for j in jobs):
            rc = run_streamed([sys.executable, os.path.join("scripts", "make_arms.py")]
                              + MAKE_ARMS, log)
            if rc != 0:
                log("make_arms.py FAILED -- arm runs will fail too")

        todo = [j for j in jobs if read_result(j) is None]
        log(f"{len(jobs) - len(todo)} of {len(jobs)} runs already done; "
            f"~{sum(j['hours'] for j in todo):.0f} GPU-hours to go")
        for i, j in enumerate(jobs, 1):
            row = {"name": j["name"], "group": j["group"]}
            res = read_result(j)
            if res is not None:
                row.update(status="done earlier", fos=res.get("fos"),
                           bracket=res.get("bracket"))
                rows.append(row)
                continue
            if j["group"] == "npde" and not waited_ok:
                row.update(status="SKIPPED", note="GPU never free; rerun later")
                rows.append(row)
                continue
            ov = next((j["args"][k + 1] for k, v in enumerate(j["args"])
                       if v == "--overrides"), None)
            if ov and not os.path.exists(ov):
                row.update(status="FAILED", note=f"{ov} missing")
                rows.append(row)
                log(f"skip {j['name']}: {ov} missing")
                continue
            log(f"=== [{i}/{len(jobs)}] {j['name']} ({j['group']}, "
                f"~{j['hours']:g} h) ===")
            t0 = time.time()
            rc = run_streamed(command(j), log,
                              os.path.join("runs", j["name"], "console.log"),
                              timeout_h=a.job_timeout_h)
            hours = round((time.time() - t0) / 3600, 2)
            res = read_result(j)
            if rc == 0 and res is not None:
                row.update(status=res.get("status", "?"), fos=res.get("fos"),
                           bracket=res.get("bracket"), hours=hours)
                log(f"--- {j['name']}: {res.get('status')} FOS "
                    f"{res.get('fos')} in {hours} h")
            else:
                row.update(status="FAILED", hours=hours,
                           note=f"exit {rc}; see runs/{j['name']}/console.log")
                log(f"--- {j['name']}: FAILED (exit {rc}) after {hours} h; "
                    f"carrying on")
            rows.append(row)
    except KeyboardInterrupt:
        log("interrupted -- rerun the same command to resume")
    finally:
        keep_awake(False)

    log("collecting results ...")
    run_streamed([sys.executable, os.path.join("scripts", "collect_results.py"),
                  "--runs", os.path.join("runs", "ssr*"),
                  "--out", os.path.join("docs", "results")], log)
    run_streamed([sys.executable, os.path.join("scripts", "recompute_criterion.py"),
                  "--runs", os.path.join("runs", "ssr*"),
                  "--out", os.path.join("docs", "results")], log)
    copy_results(log)
    out = os.path.join("docs", "results", "OVERNIGHT_SUMMARY.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(summary_md(rows, tests, started))
    log(f"wrote {out}")
    log("")
    with open(out, encoding="utf-8") as f:
        log(f.read(), stamp=False)
    log("Next: git add docs\\results configs\\arms && git commit -m "
        "\"Phase 6 runs: results\" && git push origin step-5-strength")
    return 0 if all(r["status"] not in ("FAILED", "SKIPPED") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
