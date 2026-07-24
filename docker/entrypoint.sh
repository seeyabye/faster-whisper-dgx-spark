#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Python:" "$(python3 --version || true)"
echo "[entrypoint] pip packages (nvidia-*):"
python3 - <<'PY' || true
import pkgutil
names = sorted({m.name for m in pkgutil.iter_modules() if m.name.startswith("nvidia")})
print("\n".join(names) if names else "(none)")
PY

NVPATHS="$(python3 - <<'PY'
import os, sys

def try_mod(modname):
    try:
        m = __import__(modname, fromlist=["*"])
        f = getattr(m, "__file__", None)
        print(f"[entrypoint] {modname}.__file__ = {f}")
        if not f:
            return None
        return os.path.dirname(f)
    except Exception as e:
        print(f"[entrypoint] import failed: {modname}: {e}", file=sys.stderr)
        return None

paths = []
for mod in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
    d = try_mod(mod)
    if d:
        paths.append(d)

print(":".join(paths))
PY
)"

BASE="/usr/local/cuda/lib64:/usr/local/lib"
if [[ -n "${NVPATHS}" ]]; then
  export LD_LIBRARY_PATH="${BASE}:${NVPATHS}:${LD_LIBRARY_PATH:-}"
  echo "[entrypoint] LD_LIBRARY_PATH set with NVPATHS"
else
  export LD_LIBRARY_PATH="${BASE}:${LD_LIBRARY_PATH:-}"
  echo "[entrypoint] NVPATHS empty; using CUDA base paths only"
fi

echo "[entrypoint] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

exec "$@"
