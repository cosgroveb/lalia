#!/bin/zsh
set -euo pipefail
cd ${0:A:h}
test ! -e image.id
sha256sum --check source.sha256
base=sha256:ae2d9c549e3fdcaffed02030a86ff1f0ac7a781ae4b79e1c29a867030b4fab48
test "$(docker image inspect lalia-moss-cuda-probe:20261027-r4 --format '{{.Id}}')" = "$base"
docker image inspect "$base" >parent-image.json
timeout --signal=TERM --kill-after=10s 900s docker build --pull=false \
    --tag lalia-moss-official-e2e:20261027 --iidfile image.id . >build.log 2>&1
test "$(docker image inspect lalia-moss-cuda-probe:20261027-r4 --format '{{.Id}}')" = "$base"
docker image inspect "$(cat image.id)" >frontend-image.json
python3 - <<'PY'
import json
from pathlib import Path
parent = json.loads(Path("parent-image.json").read_text())[0]
derived = json.loads(Path("frontend-image.json").read_text())[0]
layers = parent["RootFS"]["Layers"]
if derived["RootFS"]["Layers"][:len(layers)] != layers:
    raise ValueError("Derived image does not preserve the pinned parent layers")
Path("image-parent-verified").write_text(derived["Id"] + "\n")
print(derived["Id"])
PY
