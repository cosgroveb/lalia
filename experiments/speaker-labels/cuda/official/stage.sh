#!/bin/zsh
set -euo pipefail
root=${0:A:h}
remote=${1:?Usage: stage.sh /tmp/fresh-run existing-remote-weight-shard}
weights=${2:?Existing remote weight shard required}
[[ "$remote" =~ '^/tmp/lalia-moss-official-e2e-[a-zA-Z0-9._-]+$' ]]
[[ "$weights" =~ '^/[a-zA-Z0-9/._-]+$' ]]
cd "$root"
shasum --algorithm 256 --check local-source.sha256
ssh_options=(-o BatchMode=yes -o ConnectTimeout=5)
ssh $ssh_options spark.local "zsh -lic 'set -e; umask 077; mkdir -m 700 $remote; mkdir $remote/probe $remote/inputs'"
scp "$root"/{Dockerfile,.dockerignore,requirements.lock,requirements.in,constraints.txt,verify_environment.py,build-image.sh,preflight.sh,source.sha256,provenance.json} spark.local:$remote/
scp "$root"/probe/*.py spark.local:$remote/probe/
scp -r "$root"/inputs/. spark.local:$remote/inputs/
ssh $ssh_options spark.local "zsh -lic 'set -e; ln $weights $remote/inputs/model/model-00000-of-00001.safetensors; cd $remote; sha256sum --check source.sha256'"
