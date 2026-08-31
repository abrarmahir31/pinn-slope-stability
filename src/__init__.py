# Windows: torch must load before numpy/MKL or shm.dll fails to resolve.
# Harmless on Linux. See lab notebook 2026-08-25.
import torch as _torch  # noqa: F401