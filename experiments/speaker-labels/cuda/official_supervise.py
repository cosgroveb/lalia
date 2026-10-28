#!/usr/bin/env python3
"""Supervise one three-hour official CUDA run and restore the main vLLM service."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import supervise as base
from official_transcribe import GIB, available_bytes, finish_evidence, sha256, write_json

PARENT_IMAGE = "sha256:ae2d9c549e3fdcaffed02030a86ff1f0ac7a781ae4b79e1c29a867030b4fab48"
PRODUCTION_IMAGE = "sha256:f036a1380f525fe4c83639fcdc21813116e3223933751046a57e45b594ed11d5"
WORKER_SECONDS = 10800
ATTACH_SECONDS = WORKER_SECONDS + 20
WATCHDOG_SECONDS = WORKER_SECONDS + 900
WATCHDOG_JOIN_SECONDS = 600
CACHE_ENV = ("HF_HOME=/tmp/hf", "NUMBA_CACHE_DIR=/tmp/numba",
             "TRITON_CACHE_DIR=/tmp/triton", "PYTORCH_KERNEL_CACHE_PATH=/tmp/torch-kernels")
CACHE_ARGS = [argument for value in CACHE_ENV for argument in ("--env", value)]


def pid_identity(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[19]
    except FileNotFoundError:
        return None


class Supervisor(base.Supervisor):
    def __init__(self, root, call=base.command, popen=subprocess.Popen, sleep=time.sleep, *, kernel=False):
        super().__init__(root, "score", call, popen, sleep)
        self.kernel = kernel
        self.mode = "kernel" if kernel else "e2e"
        self.work = self.root / "results" / self.mode
        self.cidfile = self.work / "container.cid"
        self.name = f"lalia-moss-cuda-{self.mode}-20261027"
        self.watchdog = None
        self.watchdog_log = None

    def production_image(self):
        image = self.call(["docker", "inspect", "vllm-vllm", "--format", "{{.Image}}"], 15)
        if image != PRODUCTION_IMAGE:
            raise ValueError("Production vLLM image differs from the approved pin")
        self.events.append("production_image_verified")

    def prepare(self):
        if self.work.exists() or self.containers():
            raise ValueError("Run directory or container collision. No automatic rerun")
        for line in (self.root / "source.sha256").read_text().splitlines():
            digest, name = line.split("  ", 1)
            if sha256(self.root / name) != digest:
                raise ValueError(f"Frozen source changed: {name}")
        image = (self.root / "image.id").read_text().strip()
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", image) or image == PARENT_IMAGE:
            raise ValueError("Expected the separately built frontend image digest")
        if self.call(["docker", "image", "inspect", image, "--format", "{{.Id}}"], 15) != image:
            raise ValueError("Frontend image identity differs")
        if not self.kernel:
            receipt = json.loads((self.root / "frontend-preflight.json").read_text())
            if receipt["status"] != "ready" or receipt["image"] != image:
                raise ValueError("Official WAV preflight missing or from another image")
            if receipt["manifest_sha256"] != sha256(self.root / "inputs/manifest.json"):
                raise ValueError("Preflight input manifest differs")
            self.verify_kernel_receipt(json.loads((self.root / "kernel-preflight.json").read_text()), image)
            lifecycle = json.loads((self.root / "results/kernel/lifecycle.json").read_text())
            if lifecycle["original_error"] or lifecycle["cleanup_errors"] or not {
                    "kernel_preflight_verified", "owned_container_absent", "docker_attach_reaped",
                    "service_health_ok"}.issubset(lifecycle["events"]):
                raise ValueError("Kernel preflight or cleanup did not complete")
        if self.call(["docker", "ps", "--filter", "name=lalia-moss-cuda-", "--format", "{{.ID}}"], 15):
            raise ValueError("Another task-owned CUDA run is active")
        self.call(["systemctl", "is-active", "--quiet", base.SERVICE], 10)
        self.health(5)
        self.production_image()
        self.work.mkdir(parents=True, mode=0o700)
        self.created_work = True
        write_json(self.work / "image.json", {"image": image, "parent": PARENT_IMAGE,
                   "source_inventory_sha256": sha256(self.root / "source.sha256")})
        memory, seconds, script = ("4g", "60s", "official_kernel_preflight.py") if self.kernel else (
            "64g", f"{WORKER_SECONDS}s", "official_transcribe.py")
        self.call(["docker", "create", "--init", "--rm", "--name", self.name,
                   "--user", f"{os.getuid()}:{os.getgid()}",
                   "--cidfile", str(self.cidfile), "--label", f"{base.LABEL}={self.owner}",
                   "--gpus", "all", "--network", "none", "--read-only",
                   "--memory", memory, "--memory-swap", memory, "--cpus", "4",
                   "--tmpfs", "/tmp:rw,nosuid,exec,size=2g",
                   *CACHE_ARGS, "--env", f"FRONTEND_IMAGE_ID={image}",
                   "--env", f"SOURCE_INVENTORY_SHA256={sha256(self.root / 'source.sha256')}",
                   "--volume", f"{self.root / 'probe'}:/probe:ro",
                   "--volume", f"{self.work}:/results",
                   *([] if self.kernel else ["--volume", f"{self.root / 'inputs'}:/inputs:ro"]),
                   "--entrypoint", "timeout", image, "--signal=TERM", "--kill-after=5s",
                   seconds, "sh", "-c", 'mkdir -p "$PYTORCH_KERNEL_CACHE_PATH" && exec python3 /probe/' + script], 30)
        self.cid = self.cidfile.read_text().strip()
        if not re.fullmatch(r"[a-f0-9]{64}", self.cid) or self.containers(owned=True) != [self.cid]:
            raise ValueError("Created container identity differs")
        self.events.append("owned_container_created")

    def verify_kernel_receipt(self, receipt, image):
        if receipt["status"] != "ready" or receipt["image"] != image:
            raise ValueError("Kernel preflight missing or from another image")
        if receipt["source_inventory_sha256"] != sha256(self.root / "source.sha256"):
            raise ValueError("Kernel preflight source inventory differs")
        for variable, location in (value.split("=", 1) for value in CACHE_ENV[2:]):
            if receipt["caches"][variable]["path"] != location or not receipt["caches"][variable]["files"]:
                raise ValueError("Kernel preflight cache evidence differs")
        if (not receipt["root_read_only"] or " - tmpfs " not in receipt["tmp_mount"]
                or "noexec" in receipt["tmp_mount"]):
            raise ValueError("Kernel preflight filesystem profile differs")

    def execute(self):
        self.prepare()
        if self.kernel:
            self.log = (self.work / "container.log").open("w")
            self.process = self.popen(["docker", "start", "--attach", self.cid],
                                      stdout=self.log, stderr=subprocess.STDOUT)
            code = self.process.wait(timeout=80)
            if code:
                raise RuntimeError(f"Kernel preflight attach exited {code}")
            self.log.flush()
            from smoke_numeric import reject_fatal_diagnostics
            reject_fatal_diagnostics(self.work / "container.log")
            receipt = json.loads((self.work / "container.log").read_text())
            self.verify_kernel_receipt(receipt, (self.root / "image.id").read_text().strip())
            write_json(self.root / "kernel-preflight.json", receipt)
            self.events.append("kernel_preflight_verified")
            return
        write_json(self.work / "watchdog-owner.json", {"owner": self.owner, "cid": self.cid,
                   "pid": os.getpid(), "pid_identity": pid_identity(os.getpid())})
        self.watchdog_log = (self.work / "watchdog.log").open("w")
        self.watchdog = self.popen([sys.executable, str(Path(__file__).resolve()), "watchdog",
                                   "--root", str(self.root)], start_new_session=True,
                                  stdout=self.watchdog_log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 10
        while not (self.work / "watchdog-ready").exists():
            if self.watchdog.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("Independent host watchdog did not arm")
            self.sleep(0.1)
        self.pause_requested = True
        (self.work / "pause-requested").touch()
        self.call(["sudo", "-n", "systemctl", "stop", base.SERVICE], 100)
        self.events.append("service_stop_completed")
        snapshot = self.call(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader"], 15)
        (self.work / "resident-gpu-processes.txt").write_text(snapshot + "\n")
        utilization = self.call(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.free",
                                 "--format=csv,noheader"], 15)
        (self.work / "gpu-before.txt").write_text(utilization + "\n")
        if available_bytes() < 72 * GIB:
            raise MemoryError("Need 72 GiB MemAvailable after the main-service pause")
        self.log = (self.work / "container.log").open("w")
        self.process = self.popen(["docker", "start", "--attach", self.cid],
                                  stdout=self.log, stderr=subprocess.STDOUT)
        code = self.process.wait(timeout=ATTACH_SECONDS)
        if code:
            raise RuntimeError(f"Container attach exited {code}")
        self.log.flush()
        from smoke_numeric import reject_fatal_diagnostics
        reject_fatal_diagnostics(self.work / "container.log")
        result = json.loads((self.work / "model/result.json").read_text())
        ids = json.loads((self.work / "model/generated-ids.json").read_text())
        evidence = finish_evidence(ids, 151645)
        if result["status"] != "generated" or any(result[k] != v for k, v in evidence.items()):
            raise ValueError("Native result disagrees with actual generated IDs")
        self.events.append("native_result_verified")

    def cleanup(self, errors):
        super().cleanup(errors)
        if self.pause_requested:
            try:
                self.production_image()
            except Exception as error:
                errors.append(f"restored image: {error}")
        if self.watchdog is not None:
            (self.work / "supervisor-cleanup-finished").touch()
            try:
                code = self.watchdog.wait(timeout=WATCHDOG_JOIN_SECONDS)
                if code:
                    errors.append(f"Host watchdog exited {code}")
                self.events.append("host_watchdog_reaped")
            except Exception as error:
                errors.append(f"Host watchdog reap: {error}")
            self.watchdog_log.close()


def watchdog(root):
    runner = Supervisor(root)
    info = json.loads((runner.work / "watchdog-owner.json").read_text())
    runner.owner, runner.cid = info["owner"], info["cid"]
    (runner.work / "watchdog-ready").write_text(str(os.getpid()))
    deadline = time.monotonic() + WATCHDOG_SECONDS
    while not (runner.work / "supervisor-cleanup-finished").exists():
        if pid_identity(info["pid"]) != info["pid_identity"] or time.monotonic() >= deadline:
            break
        time.sleep(1)
    else:
        return 0
    # The same ownership checks and absence-before-restore order apply after SIGKILL.
    (runner.work / "watchdog-recovering").touch()
    runner.pause_requested = (runner.work / "pause-requested").exists()
    errors = []
    runner.cleanup(errors)
    write_json(runner.work / "watchdog-recovery.json", {"errors": errors, "events": runner.events})
    return int(bool(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "watchdog", "kernel"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "watchdog":
        raise SystemExit(watchdog(args.root))
    for sig in base.SIGNALS:
        signal.signal(sig, base.interrupted)
    raise SystemExit(Supervisor(args.root, kernel=args.action == "kernel").run())


if __name__ == "__main__":
    main()
