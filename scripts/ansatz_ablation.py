r"""Matched ablation over `NearPhysical.mode`. The experiment D-A.1 needs.

WHY A TRAINING RUN AND NOT ANOTHER GRADIENT TABLE. Day 27 measured six ansatz
variants by `w^` spread at t = 0 and produced a monotone trade-off with no
winner: the spread falls as the cap softens, and softening is exactly what
distorts the IC in the bottom few metres, which is the only band where the
Richards physics is live. A t = 0 gradient norm cannot separate "this ansatz
makes the seed measurable" from "this ansatz has moved the field somewhere the
solution is not". Only a trajectory can, and it has to be matched: same N, same
sampling and network seeds, same epochs, same balancer, same schedule, one
variable.

WHAT TO READ, in order of what actually decides it:

  psi_sat_frac   fraction of interior points at psi >= 0. Section 5 is
                 unsaturated everywhere (Z_WT = 197 m below Z_BASE = 200 m), so
                 this should be 0. Under `unbounded` at 8x64 it is 0.118 at
                 step 0 and RISES to 0.343 by step 20 before training pulls it
                 back. A bounded arm holds it at 0 by construction; the
                 question is what that costs.

  ic_head        the IC violation. The Day 26 note recorded ic_head ending
                 ABOVE 1.0 relative to L0 in every seed arm, and prod01 off the
                 analytic IC by 84 m of head at step 500. If a bounded ansatz
                 fixes psi_sat_frac but ic_head gets worse, it has bought
                 measurability by pinning the field somewhere wrong.

  pde_richards   the term whose seed was unmeasurable. Falling here is only
                 meaningful alongside ic_head -- the Day 25 disaster was
                 pde_richards at 5e-08 because the equation was structurally
                 satisfied, not solved.

  spread,        balancer health. A term on the clip has a weight no longer
  clipped        set by measurement.

WHAT THIS DOES NOT SETTLE. Two network seeds and a few hundred epochs on CPU
is not production. It is enough to reject an arm, not to adopt one. Run it at
production N on the lab PC before D-A.1 is closed.

    PYTHONPATH=. python scripts/ansatz_ablation.py --epochs 500 --seeds 20250812 7
    PYTHONPATH=. python scripts/ansatz_ablation.py --arms unbounded cap \
        --epochs 2000 --n-pde 10000 --device cuda --out runs/ansatz
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ARMS = {
    "unbounded": ["--ansatz", "unbounded"],
    "cap":       ["--ansatz", "cap", "--cap-k", "100"],
    "exp":       ["--ansatz", "exp"],
}

READ = ("bc_mech", "pde_mech", "pde_richards", "bc", "ic_head", "ic_disp")


# A log that has been touched within this many seconds is assumed to belong to
# a LIVE driver, not to a dead one. train.py flushes every --log-every steps, so
# at 0.57 s/epoch and log-every 25 that is ~14 s between writes; 300 s is a wide
# margin that still lets a genuinely crashed run be reclaimed in five minutes.
def _kill_with_parent():
    """Popen kwargs that tie the child's lifetime to this process at OS level.

    Needed because detaching the child from the console Ctrl-C group (so the
    parent can manage it deterministically) also means nothing kills the child
    if the parent dies without running its `finally`. A hard kill of the driver
    would otherwise leave a CUDA process training forever.

    Windows: a Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. The job
    handle is held by this process; when it exits, for any reason, the kernel
    closes the handle and kills everything in the job.

    Linux: prctl(PR_SET_PDEATHSIG, SIGKILL) in the child, which asks the kernel
    to signal it when its parent dies.

    Falls back to plain detachment if either mechanism is unavailable -- better
    than not detaching, since the common case is still a catchable Ctrl-C.
    """
    if sys.platform == "win32":
        kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
                               | 0x01000000}        # CREATE_BREAKAWAY_FROM_JOB
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)

            class _LIMITS(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                            ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", wintypes.DWORD),
                            ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t),
                            ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                            ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class _EXT(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", _LIMITS),
                            ("IoInfo", ctypes.c_byte * 48),
                            ("ProcessMemoryLimit", ctypes.c_size_t),
                            ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t),
                            ("PeakJobMemoryUsed", ctypes.c_size_t)]

            job = k32.CreateJobObjectW(None, None)
            info = _EXT()
            info.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_CLOSE
            k32.SetInformationJobObject(job, 9, ctypes.byref(info),
                                        ctypes.sizeof(info))
            _JOBS.append(job)                 # keep the handle alive
            kw["_job"] = job
        except Exception:
            pass                              # no job; detachment still applies
        return kw

    def _pdeathsig():
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").prctl(1, 9)   # PR_SET_PDEATHSIG, SIGKILL
        except Exception:
            pass
    return {"start_new_session": True, "preexec_fn": _pdeathsig}


_JOBS = []          # module-level so Job handles outlive run_one


LOCK_STALE_S = 300.0


class RunLocked(Exception):
    """Another driver owns this run directory."""


def _claim(out, log):
    """Take exclusive ownership of a run directory, or raise RunLocked.

    WHY. `run_one` deletes a partial log before restarting, because train.py
    APPENDS -- leaving a partial in place would interleave two trajectories into
    one file that parses cleanly and is silently wrong. That delete is safe only
    if no other process is writing. On 13 Sep two drivers were started against
    the same runs/ansatz and the second unlinked a log the first's child was
    still appending to.

    Two guards, because either alone is defeatable: an O_EXCL lockfile (catches
    a concurrent driver) and an mtime check on the log itself (catches a driver
    that died without releasing, and an ORPHANED CHILD still training after its
    parent was killed -- which is the case the lockfile cannot see, because the
    lock belongs to the dead parent while the writer is the surviving child).
    """
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".lock"

    if log.exists():
        age = time.time() - log.stat().st_mtime
        if age < LOCK_STALE_S:
            raise RunLocked(
                f"{log} was written {age:.0f}s ago -- another driver or an "
                f"orphaned child is still running this. Check `nvidia-smi` for "
                f"a C-type python.exe, or wait {LOCK_STALE_S - age:.0f}s.")

    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        age = time.time() - lock.stat().st_mtime
        if age < LOCK_STALE_S:
            raise RunLocked(f"{lock} held by pid in file, {age:.0f}s old")
        lock.unlink()                 # stale: previous driver died holding it
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    return lock


def run_one(arm, seed, a, root):
    out = root / f"{arm}_seed{seed}"
    log = out / "log.jsonl"
    child_out = out / "child.log"

    # Resume. A full ablation is arms x seeds x epochs and at 0.57 s/epoch that
    # is hours; it WILL be interrupted. A run is complete when its log's last
    # record reaches --epochs, and completed runs are reused rather than
    # repeated, so the driver can be re-invoked until the table fills in.
    if log.exists():
        try:
            done = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
            if done and done[-1]["step"] >= a.epochs - a.log_every:
                print(f"  {arm:<10} seed {seed:<9} {len(done):>4} records  "
                      f"(reused)")
                return done
        except (json.JSONDecodeError, KeyError, OSError):
            pass                      # truncated by an interrupt; rerun it

    try:
        lock = _claim(out, log)
    except RunLocked as e:
        print(f"  {arm:<10} seed {seed:<9} SKIPPED -- {e}")
        return None

    if log.exists():
        log.unlink()                  # train.py appends; clear the partial

    cmd = [sys.executable, "scripts/train.py",
           "--layers", str(a.layers), "--width", str(a.width),
           "--seed", str(seed), "--device", a.device,
           "--n-pde", str(a.n_pde), "--n-bc", str(a.n_bc),
           "--n-iface", str(a.n_iface),
           "--epochs", str(a.epochs), "--eps-psi", str(a.eps_psi),
           "--log-every", str(a.log_every), "--ckpt-every", "0",
           "--out", str(out)] + ARMS[arm]

    # The child's stdout goes to a FILE, not to a pipe. capture_output=True
    # holds it in memory and discards it when the child dies without writing,
    # which is how `unbounded/7: FAILED` reported nothing at all. A file also
    # means the run is tailable from another terminal while it is in flight.
    #
    # Detaching the child from the console's Ctrl-C group makes the PARENT --
    # not the terminal -- own its lifetime, so the `finally` below can
    # guarantee cleanup. That alone is not enough: it also guarantees the child
    # is ORPHANED if the parent dies uncatchably (SIGKILL, taskkill /F, a closed
    # console). Measured: SIGKILL the driver and the child reparents to init and
    # keeps training, holding VRAM and appending to a log the next driver will
    # try to delete. So `_kill_with_parent` installs an OS-level guarantee on
    # top -- a Job Object on Windows, PR_SET_PDEATHSIG on Linux -- that fires
    # whether or not any Python code gets to run.
    kw = dict(_kill_with_parent())

    t = time.time()
    proc = None
    try:
        with open(child_out, "w") as fh:
            job = kw.pop("_job", None)
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    text=True, **kw)
            if job is not None:
                import ctypes
                h = ctypes.WinDLL("kernel32").OpenProcess(
                    0x1F0FFF, False, proc.pid)     # PROCESS_ALL_ACCESS
                ctypes.WinDLL("kernel32").AssignProcessToJobObject(job, h)
                ctypes.WinDLL("kernel32").CloseHandle(h)
            rc = proc.wait()
    except KeyboardInterrupt:
        print(f"\n  {arm}/{seed}: interrupted, terminating child "
              f"pid {proc.pid if proc else '?'} ...")
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise
    finally:
        if proc and proc.poll() is None:      # any other exception path
            proc.kill()
            proc.wait()
        lock.unlink(missing_ok=True)

    if rc != 0:
        tail = ""
        try:
            tail = "".join(child_out.read_text(errors="replace")
                           .splitlines(keepends=True)[-25:])
        except OSError:
            pass
        print(f"  {arm}/{seed}: FAILED rc={rc}  (full output in {child_out})\n"
              f"{tail or '    child produced no output at all'}")
        return None

    try:
        recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    except OSError as e:
        print(f"  {arm}/{seed}: exited 0 but no readable log ({e})")
        return None
    print(f"  {arm:<10} seed {seed:<9} {len(recs):>4} records  "
          f"{time.time() - t:>6.1f}s")
    return recs


def summarise(recs):
    """Everything the decision turns on, from first and last record."""
    first, last = recs[0], recs[-1]
    L0 = last["L0"]
    return {
        "psi_sat_frac_0": first["psi_sat_frac"],
        "psi_sat_frac_max": max(r["psi_sat_frac"] for r in recs),
        "psi_sat_frac_end": last["psi_sat_frac"],
        "psi_max_m_end": last["psi_max_m"],
        "ratios": {k: last["L"][k] / L0[k] for k in READ if L0.get(k)},
        "spread_end": last.get("spread"),
        "clipped_end": last.get("clipped", []),
        "clipped_ever": sorted({c for r in recs for c in r.get("clipped", [])}),
        "steps": last["step"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    choices=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[20250812, 7])
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--n-pde", type=int, default=1200)
    ap.add_argument("--n-bc", type=int, default=600)
    ap.add_argument("--n-iface", type=int, default=400)
    ap.add_argument("--eps-psi", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--out", default="runs/ansatz")
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args(argv)

    root = pathlib.Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    print(f"{a.layers}x{a.width}, N_PDE {a.n_pde}, {a.epochs} epochs, "
          f"eps_psi {a.eps_psi}, seeds {a.seeds}, arms {a.arms}\n")

    res = {}
    for arm in a.arms:
        for seed in a.seeds:
            recs = run_one(arm, seed, a, root)
            if recs:
                res[f"{arm}|{seed}"] = summarise(recs)

    out = {"config": vars(a), "runs": res}

    print(f"\n{'arm':<11}{'seed':>10}{'sat@0':>8}{'sat max':>9}{'sat end':>9}"
          f"{'psi max':>10}")
    print("-" * 57)
    for k, v in res.items():
        arm, seed = k.split("|")
        print(f"{arm:<11}{seed:>10}{v['psi_sat_frac_0']:>8.3f}"
              f"{v['psi_sat_frac_max']:>9.3f}{v['psi_sat_frac_end']:>9.3f}"
              f"{v['psi_max_m_end']:>10.2f}")

    print(f"\nL_i / L0_i at step {a.epochs} (lower is better; ic_head above 1 "
          "means the IC got WORSE)")
    hdr = f"{'arm':<11}{'seed':>10}" + "".join(f"{k:>14}" for k in READ)
    print(hdr)
    print("-" * len(hdr))
    for k, v in res.items():
        arm, seed = k.split("|")
        print(f"{arm:<11}{seed:>10}" + "".join(
            f"{v['ratios'].get(t, float('nan')):>14.3e}" for t in READ))

    print(f"\n{'arm':<11}{'seed':>10}{'spread':>12}   clipped (ever)")
    for k, v in res.items():
        arm, seed = k.split("|")
        sp = v["spread_end"]
        print(f"{arm:<11}{seed:>10}"
              f"{(sp if sp else float('nan')):>12.3g}   "
              f"{','.join(v['clipped_ever']) or '-'}")

    print("\nREAD psi_sat_frac AND ic_head TOGETHER. An arm that zeroes the "
          "first while\nraising the second has not fixed the ansatz, it has "
          "moved the field off its own\ninitial condition to avoid the "
          "singularity. Neither is sufficient alone.")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())