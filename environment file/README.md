# rosetta-ml — environment setup

Reproducible Python 3.10 environment for protein modeling + machine learning:
**PyTorch (CPU)**, **NumPy**, **SciPy**, **Matplotlib**, and **PyRosetta** (Rosetta 3).

Everything is driven by `environment.yml` so the lab PC gets an identical stack.

---

## 0. Prerequisites

- A conda installer. Use **Miniforge** (free, defaults to the `conda-forge` channel).
  Anaconda's `defaults` channel now requires a paid license for many
  organizations — Miniforge sidesteps that entirely.
  Download: https://github.com/conda-forge/miniforge#install
- Git.
- A **PyRosetta license** (free for academic / non-commercial use). Request it at
  https://els2.comotion.uw.edu/product/pyrosetta — you'll receive credentials by
  email. (A plain *Rosetta* license is a **different** product; make sure you get
  the *PyRosetta* one.)

> Windows note: PyRosetta targets Linux/macOS. On Windows, do all of this inside
> **WSL2 (Ubuntu)** for the smoothest path. Native-Windows PyRosetta builds exist
> but WSL is far less trouble.

---

## 1. Install conda (Miniforge)

Linux / macOS / WSL:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
# accept the license, let it run `conda init`, then restart your shell
```

Confirm:

```bash
conda --version
```

---

## 2. Get the project files

Either clone this repo (once it exists on GitHub/GitLab) or create the folder and
drop in `environment.yml`, `.gitignore`, and `verify_env.py`:

```bash
mkdir rosetta-ml && cd rosetta-ml
# ...place environment.yml / .gitignore / verify_env.py here...
```

---

## 3. Create the environment from environment.yml

```bash
conda env create -f environment.yml
conda activate rosetta-ml
```

This installs Python 3.10, the CPU build of PyTorch, NumPy, SciPy, Matplotlib,
and pip — all from `conda-forge`.

Check the core stack:

```bash
python verify_env.py
```

You should see versions for numpy/scipy/matplotlib/torch, `PyTorch CUDA
available: False`, and pyrosetta reported as *not installed* (that's next).

---

## 4. (Alternative) venv instead of conda

`environment.yml` is a conda format, so conda is recommended. If you must use a
plain virtual environment, PyTorch CPU comes from the PyTorch index:

```bash
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy matplotlib
```

Then continue to Step 5 for PyRosetta. (You lose the single-file `environment.yml`
reproducibility with this route — prefer conda if you can.)

---

## 5. Install PyRosetta (do this after the env is active)

**Recommended — pip installer (no credentials in any file):**

```bash
pip install pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
```

**Alternative — conda channel (needs your credentials):**
Put your PyRosetta username/password into `~/.condarc` (this file is gitignored),
then install:

```yaml
# ~/.condarc
channels:
  - https://USERNAME:PASSWORD@conda.graylab.jhu.edu
  - conda-forge
```

```bash
conda install pyrosetta
```

Verify the whole stack now, including PyRosetta:

```bash
python verify_env.py
```

A line like `[ok]  pyrosetta  <version>` means you're done.

---

## 6. Initialise Git and push to GitHub / GitLab

```bash
git init
git add environment.yml .gitignore verify_env.py README.md
git commit -m "Initial environment: Python 3.10, PyTorch CPU, SciPy stack, PyRosetta setup"
git branch -M main
```

Create an **empty** remote repo on GitHub or GitLab (no README/licence — you have
them), copy its URL, then:

```bash
# GitHub
git remote add origin https://github.com/<you>/rosetta-ml.git
# or GitLab
# git remote add origin https://gitlab.com/<you>/rosetta-ml.git

git push -u origin main
```

If prompted for a password, use a **personal access token**, not your account
password (GitHub → Settings → Developer settings → Tokens; GitLab → Preferences →
Access Tokens).

---

## 7. Replicate on the lab PC

On the lab machine, install Miniforge (Step 1), then:

```bash
git clone https://github.com/<you>/rosetta-ml.git
cd rosetta-ml
conda env create -f environment.yml
conda activate rosetta-ml
# then repeat Step 5 to add PyRosetta (license credentials are per-machine)
python verify_env.py
```

Because PyRosetta lives outside `environment.yml`, the lab PC gets a byte-identical
conda stack and adds PyRosetta with the same two commands — no secrets travel
through Git.

### Pinning an exact PyRosetta build (optional, for strict reproducibility)

After installing PyRosetta on the first machine, capture its exact version:

```bash
pip freeze | grep -i pyrosetta   # e.g. pyrosetta==2026.xx+release.<hash>
```

Record that string in this README (or a `pyrosetta-version.txt`) so the lab PC can
install the same build via the pip installer's version argument or the matching
wheel.

---

## Keeping the environment in sync

When you add a package, edit `environment.yml`, then on every machine:

```bash
conda env update -f environment.yml --prune
git add environment.yml && git commit -m "env: add <package>" && git push
```
