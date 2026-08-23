"""Must set these before faiss or torch get imported anywhere in the test
process — both link their own OpenMP runtime (libomp on macOS,
libiomp5md.dll on Windows), and loading both
without this flag set reliably segfaults (observed directly: the full
suite crashed mid-run once `correlation/closed_loop.py` (faiss) and
`detector/model.py` (torch) were both exercised in the same pytest
process). conftest.py is imported by pytest before any test module, which
is early enough for this to take effect.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
