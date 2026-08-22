#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${SUBMISSION_IMAGE:-supermarket_sorting:submission}"
EXPECTED_SHA="5763801f2875491ab8b00c61fd1ed539de221ae3d2712cf93ad2464529365daf"

docker build \
  --build-arg "MODEL_SHA256=${EXPECTED_SHA}" \
  -t "${IMAGE}" \
  "${ROOT}"

# Intentionally no -v/--mount: prove the image contains source, scripts and
# the nine-class checkpoint.
docker run --rm "${IMAGE}" bash -lc '
  set -euo pipefail
  cd /workspace/baseline
  (cd examples/supermarket_sorting/perception/checkpoints && sha256sum -c supermarket_multiclass.pt.sha256)
  test -x scripts/run_v2_decision_client.sh
  python3 -m unittest discover -s examples/supermarket_sorting/tests -p "test_*.py"
  python3 - <<"PY"
from examples.supermarket_sorting.perception.backends import YoloBackend

path = "/workspace/baseline/examples/supermarket_sorting/perception/checkpoints/supermarket_multiclass.pt"
backend = YoloBackend(path)
if not backend.is_official_multiclass:
    raise SystemExit("checkpoint does not expose exactly the nine official classes")
print("submission image verification: source + tests + official model OK")
PY
'
