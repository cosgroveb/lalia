#!/bin/zsh
set -euo pipefail
root=${0:A:h}
cd "$root"
test ! -e frontend-preflight.json
sha256sum --check source.sha256
image=$(cat image.id)
test "$(cat image-parent-verified)" = "$image"
name=lalia-moss-frontend-preflight-20261027
owner=$(python3 -c 'import uuid; print(uuid.uuid4().hex)')
test -z "$(docker ps --all --filter name=^/$name\$ --format '{{.ID}}')"
cleanup() {
    cid=$(docker ps --all --no-trunc --filter name=^/$name\$ \
        --filter label=lalia.moss.preflight.owner=$owner --format '{{.ID}}')
    if [[ -n "$cid" ]]; then
        docker stop --time 5 "$cid"
    fi
}
trap cleanup EXIT INT TERM HUP
timeout --signal=TERM --kill-after=5s 120s docker run --init --rm --name "$name" \
    --cidfile "$root/frontend-preflight.cid" --label lalia.moss.preflight.owner=$owner \
    --network none --read-only --memory 6g --memory-swap 6g --cpus 2 \
    --tmpfs /tmp:rw,nosuid,size=1g --env HF_HOME=/tmp/hf --env NUMBA_CACHE_DIR=/tmp/numba \
    --env FRONTEND_IMAGE_ID="$image" \
    --volume "$root/probe:/probe:ro" --volume "$root/inputs:/inputs:ro" \
    --entrypoint timeout "$image" --signal=TERM --kill-after=5s 110s \
    python3 /probe/official_preflight.py >frontend-preflight.tmp 2>frontend-preflight.log
python3 - <<'PY'
import json
from pathlib import Path
value = json.loads(Path("frontend-preflight.tmp").read_text())
if value["status"] != "ready":
    raise ValueError("Official WAV preflight failed")
Path("frontend-preflight.tmp").rename("frontend-preflight.json")
PY
