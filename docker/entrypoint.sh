#!/usr/bin/env bash
set -euo pipefail

NVPATHS="$(python3 - <<'PY' || true
import os
paths = []
for mod in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
    try:
        m = __import__(mod, fromlist=["*"])
        d = os.path.dirname(m.__file__)
        if d:
            paths.append(d)
    except:
        pass
print(":".join(paths))
PY
)"

export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib:${NVPATHS}"

exec "$@"