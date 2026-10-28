#!/usr/bin/env python3
"""Bound and clean up one MOSS smoke or score container."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time
import uuid

SERVICE = "vllm-vllm.service"
IMAGE = "lalia-moss-cuda-probe:20261027-r4"
EXACT_PREFIX_IMAGE = "sha256:ae2d9c549e3fdcaffed02030a86ff1f0ac7a781ae4b79e1c29a867030b4fab48"
PRODUCTION_IMAGE = "sha256:f036a1380f525fe4c83639fcdc21813116e3223933751046a57e45b594ed11d5"
CACHE_ENV = ("HF_HOME=/tmp/hf", "NUMBA_CACHE_DIR=/tmp/numba",
             "TRITON_CACHE_DIR=/tmp/triton", "PYTORCH_KERNEL_CACHE_PATH=/tmp/torch-kernels")
LABEL = "lalia.moss.probe.owner"
SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class Interrupted(RuntimeError):
    pass


def interrupted(signum, frame):
    raise Interrupted(f"Received signal {signum}")


def command(argv, timeout):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{argv[0:3]} exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout.strip()


def idle_metrics(text):
    names = ("vllm:num_requests_running", "vllm:num_requests_waiting")
    values = {name: [] for name in names}
    label = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"(?:[^"\\]|\\[\\n"])*"\s*(?:,|$)')
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for name in names:
            if line == name or line.startswith((name + "{", name + " ", name + "\t")):
                match = re.fullmatch(re.escape(name) + r'(?:\{(.*)\})?\s+(\S+)(?:\s+\d+)?', line)
                if match is None:
                    raise ValueError(f"Malformed idle gauge: {name}")
                labels, offset, keys = (match[1] or "").strip(), 0, set()
                while offset < len(labels):
                    item = label.match(labels, offset)
                    if item is None or item[1] in keys:
                        raise ValueError(f"Malformed or duplicate idle label: {name}")
                    keys.add(item[1])
                    offset = item.end()
                value = float(match[2])
                if not math.isfinite(value) or value != 0:
                    raise ValueError(f"Service is not idle: {name}")
                values[name].append(value)
    if not all(values.values()):
        raise ValueError("Missing running or waiting gauge")
    return values


class Supervisor:
    def __init__(self, root, mode, call=command, popen=subprocess.Popen, sleep=time.sleep, *, fp32_rope=False, exact_prefix=False,
                 hidden_boundary=False, midpoint=None, shared_inputs=None):
        if fp32_rope and mode != "score":
            raise ValueError("--fp32-rope requires score mode")
        if exact_prefix and mode not in ("score", "hidden-kernel"):
            raise ValueError("--exact-prefix requires score mode")
        if mode == "hidden-kernel":
            hidden_boundary, exact_prefix = True, True
        if midpoint:
            if midpoint not in ("capture", "self", "cross") or mode != "score" or not exact_prefix or fp32_rope or hidden_boundary:
                raise ValueError("Midpoint requires only the unchanged exact-prefix score profile")
            hidden_boundary = True
        if shared_inputs is not None and not hidden_boundary:
            raise ValueError("Shared inputs require the reviewed boundary runtime")
        if hidden_boundary and (mode not in ("score", "hidden-kernel") or not exact_prefix or fp32_rope):
            raise ValueError("Hidden boundary requires the unchanged exact-prefix profile")
        self.hidden_boundary = hidden_boundary
        self.exact_prefix = exact_prefix
        self.fp32_rope = fp32_rope
        self.root, self.mode = root.resolve(), mode
        self.midpoint = midpoint
        self.inputs = shared_inputs.resolve() if shared_inputs is not None else self.root / "inputs"
        self.work = self.root / "results" / ("midpoint-" + midpoint if midpoint else mode)
        self.cidfile = self.work / "container.cid"
        self.name = f"lalia-moss-cuda-midpoint-{midpoint}-20261028" if midpoint else f"lalia-moss-cuda-{mode}-20261027"
        self.owner = uuid.uuid4().hex
        self.call, self.popen, self.sleep = call, popen, sleep
        self.cid = None
        self.process = None
        self.start_requested = False
        self.log = None
        self.created_work = False
        self.pause_requested = False
        self.events = []

    def source_hash(self):
        path = self.root / "source-hashes.json"
        inventory = json.loads(path.read_text())
        if set(inventory) != {item.name for item in (self.root / "probe").iterdir()}:
            raise ValueError("Frozen boundary source inventory differs")
        for name, digest in inventory.items():
            if Path(name).name != name or hashlib.sha256((self.root / "probe" / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Frozen boundary source differs")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def production_image(self):
        if self.call(["docker", "inspect", "vllm-vllm", "--format", "{{.Image}}"], 15) != PRODUCTION_IMAGE:
            raise ValueError("Production image differs")
        self.events.append("production_image_verified")

    def verify_hidden_kernel(self):
        directory = self.root / "results/hidden-kernel"
        receipt = json.loads((directory / "kernel.json").read_text())
        lifecycle = json.loads((directory / "lifecycle.json").read_text())
        required = {"hidden_kernel_verified", "owned_container_absent", "docker_attach_reaped", "service_health_ok"}
        if lifecycle["original_error"] or lifecycle["cleanup_errors"] or not required.issubset(lifecycle["events"]):
            raise ValueError("Hidden kernel preflight or cleanup failed")
        self.require_hidden_kernel(receipt, directory)

    def verify_boundary_inputs(self):
        from hidden_boundary import capture_receipt, sha256

        inputs = self.inputs
        if sha256(inputs / "manifest.json") != (self.root / "input-manifest.sha256").read_text().strip():
            raise ValueError("Boundary input manifest changed")
        manifest = json.loads((inputs / "manifest.json").read_text())
        if manifest.get("hidden_boundary") is not True or manifest.get("score_profile") != "exact-prefix":
            raise ValueError("Boundary input profile differs")
        for name, digest in manifest["files"].items():
            if Path(name).name != name or sha256(inputs / name) != digest:
                raise ValueError("Boundary input changed")
        capture_receipt(inputs, "mac")

    def verify_midpoint(self):
        from midpoint import validate_config

        path = self.root / f"midpoint-{self.midpoint}.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != path.with_suffix(".sha256").read_text().strip():
            raise ValueError("Midpoint pass configuration changed")
        config = json.loads(path.read_text())
        if config["phase"] != self.midpoint:
            raise ValueError("Midpoint pass differs")
        validate_config(self.root, config, "cuda")

    def require_hidden_kernel(self, receipt, directory):
        if (receipt["status"] != "ready" or receipt["image"] != EXACT_PREFIX_IMAGE
                or receipt["source_inventory_sha256"] != self.source_hash()
                or receipt["outer_product_max_error"] != 0 or receipt["cuda_peak_allocated_bytes"] >= 1024**3
                or receipt["root_read_only"] or "noexec" in receipt["tmp_mount"] or " - tmpfs " not in receipt["tmp_mount"]):
            raise ValueError("Hidden kernel runtime or cache receipt differs")
        for variable, expected in (value.split("=", 1) for value in CACHE_ENV[2:]):
            if receipt["caches"][variable]["path"] != expected or not receipt["caches"][variable]["files"]:
                raise ValueError("Hidden kernel cache was not exercised")
        expected = {"uid": os.getuid(), "gid": os.getgid(), "directory_mode": 0o700,
                    "file_uid": os.getuid(), "file_gid": os.getgid(), "file_mode": 0o600}
        if receipt["ownership"] != expected:
            raise ValueError("Hidden kernel output ownership differs")
        output = directory / "private-output/check.json"
        if json.loads(output.read_text()) != {"uid": os.getuid(), "gid": os.getgid()}:
            raise ValueError("Host cannot read expected private kernel output")
        if (output.stat().st_uid != os.getuid() or output.stat().st_gid != os.getgid()
                or output.stat().st_mode & 0o777 != 0o600 or output.parent.stat().st_uid != os.getuid()
                or output.parent.stat().st_gid != os.getgid() or output.parent.stat().st_mode & 0o777 != 0o700):
            raise ValueError("Host output ownership or private modes differ")

    def containers(self, owned=False):
        args = ["docker", "ps", "--all", "--no-trunc", "--filter", f"name=^/{self.name}$"]
        if owned:
            args += ["--filter", f"label={LABEL}={self.owner}"]
        result = self.call(args + ["--format", "{{.ID}}"], 15).split()
        if any(not re.fullmatch(r"[a-f0-9]{64}", cid) for cid in result):
            raise ValueError("Unexpected Docker container ID")
        if len(result) > 1 or (owned and self.cid and any(cid != self.cid for cid in result)):
            raise ValueError("Container ownership mismatch")
        if owned and result and self.cid is None:
            self.cid = result[0]
        return result

    def health(self, seconds):
        deadline = time.monotonic() + seconds
        last = None
        while True:
            try:
                self.call(["curl", "--fail", "--silent", "--show-error", "--max-time", "3",
                           "--output", "/dev/null", "http://127.0.0.1:8000/health"], 5)
                self.events.append("service_health_ok")
                return
            except Interrupted:
                raise
            except Exception as error:
                last = error
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Service health deadline: {last}")
            self.sleep(min(2, max(0, deadline - time.monotonic())))

    def prepare(self):
        if self.work.exists() or self.cidfile.exists():
            raise ValueError(f"Run directory/CID collision: {self.work}")
        if self.containers():
            raise ValueError(f"Container name collision: {self.name}")
        self.call(["systemctl", "is-active", "--quiet", SERVICE], 10)
        self.health(5)
        if self.hidden_boundary:
            self.source_hash()
            self.production_image()
            if self.call(["docker", "image", "inspect", EXACT_PREFIX_IMAGE, "--format", "{{.Id}}"], 15) != EXACT_PREFIX_IMAGE:
                raise ValueError("Archived decoder image is unavailable or differs")
            if self.mode == "score":
                self.verify_hidden_kernel()
                self.verify_boundary_inputs()
                if self.midpoint:
                    self.verify_midpoint()
        self.work.mkdir(parents=True, mode=0o700)
        self.created_work = True
        seconds, memory = {"smoke": (45, "2g"), "shapes": (120, "6g"), "decoder": (100, "4g"),
                           "score": (300, "24g"), "hidden-kernel": (60, "4g")}[self.mode]
        script = "/probe/decoder_smoke.py" if self.mode == "decoder" else "/probe/score_decoder.py"
        if self.mode == "hidden-kernel":
            script = "/probe/hidden_kernel_preflight.py"
        args = ["docker", "create", "--init", "--rm", "--name", self.name,
                "--cidfile", str(self.cidfile), "--label", f"{LABEL}={self.owner}",
                "--gpus", "all", "--network", "none", "--memory", memory,
                "--memory-swap", memory, "--cpus", "4",
                "--volume", f"{self.root / 'probe'}:/probe:ro",
                "--volume", f"{self.work}:/results",
                "--volume", f"{self.inputs}:/inputs:ro",
                "--entrypoint", "timeout", EXACT_PREFIX_IMAGE if self.exact_prefix else IMAGE, "--signal=TERM", "--kill-after=5s",
                f"{seconds}s", "python3", script]
        args += {"smoke": ["--cuda-smoke"], "shapes": ["--cuda-shapes"],
                 "decoder": [], "hidden-kernel": [], "score": ["--inputs", "/inputs", "--out", "/results/model"]}[self.mode]
        if self.exact_prefix and self.mode == "score":
            args.append("--exact-prefix")
        if self.fp32_rope:
            args.append("--fp32-rope")
        if self.hidden_boundary:
            position = args.index("--gpus")
            settings = ["--user", f"{os.getuid()}:{os.getgid()}", "--tmpfs", "/tmp:rw,nosuid,exec,size=2g"]
            for value in (*CACHE_ENV, f"FRONTEND_IMAGE_ID={EXACT_PREFIX_IMAGE}",
                          f"SOURCE_INVENTORY_SHA256={self.source_hash()}"):
                settings += ["--env", value]
            args[position:position] = settings
            if self.midpoint:
                args[position:position] = ["--volume", f"{self.root}:/trial:ro"]
            index = args.index("python3")
            worker = args[index:]
            if self.midpoint:
                worker += ["--midpoint-config", f"/trial/midpoint-{self.midpoint}.json"]
            elif self.mode == "score":
                worker.append("--hidden-boundary")
            # Arguments are fixed paths/flags, not shell-supplied text.
            args[index:] = ["sh", "-c", "mkdir -p /tmp/torch-kernels && exec " + shlex.join(worker)]
        self.call(args, 30)
        self.cid = self.cidfile.read_text().strip()
        if not re.fullmatch(r"[a-f0-9]{64}", self.cid) or self.containers(owned=True) != [self.cid]:
            raise ValueError("Created container identity differs")
        self.events.append("owned_container_created")

    def execute(self):
        self.prepare()
        if self.mode == "score":
            if self.hidden_boundary:
                metrics = self.call(["curl", "--fail", "--silent", "--show-error", "--max-time", "5",
                                     "http://127.0.0.1:8000/metrics"], 7)
                observed = idle_metrics(metrics)
                (self.work / "idle.json").write_text(json.dumps(observed, indent=2) + "\n")
                self.events.append("service_idle_verified")
            self.pause_requested = True
            self.call(["sudo", "-n", "systemctl", "stop", SERVICE], 100)
            self.events.append("service_stop_completed")
        self.log = (self.work / "container.log").open("w")
        self.start_requested = True
        self.process = self.popen(["docker", "start", "--attach", self.cid],
                                  stdout=self.log, stderr=subprocess.STDOUT)
        result = self.process.wait(timeout={"smoke": 60, "shapes": 140, "decoder": 120, "score": 320,
                                            "hidden-kernel": 80}[self.mode])
        if result:
            raise RuntimeError(f"Container attach exited {result}")
        from smoke_numeric import reject_fatal_diagnostics
        self.log.flush()
        reject_fatal_diagnostics(self.work / "container.log")
        if self.mode == "hidden-kernel":
            receipt = json.loads((self.work / "container.log").read_text())
            self.require_hidden_kernel(receipt, self.work)
            (self.work / "kernel.json").write_text(json.dumps(receipt, indent=2) + "\n")
            self.events.append("hidden_kernel_verified")
        elif self.midpoint:
            from midpoint import final_receipt, read_sequence

            receipt = json.loads((self.work / "model/result.json").read_text())["midpoint"]
            if receipt["final"] != final_receipt(self.work / "model", "cuda", self.midpoint):
                raise ValueError("Midpoint final result changed")
            if self.midpoint == "capture":
                read_sequence(self.work / "model/sequence", "cuda")
            self.verify_midpoint()
            self.events.append("midpoint_verified")
        elif self.hidden_boundary:
            from hidden_boundary import capture_receipt

            capture_receipt(self.work / "model", "cuda")
            receipt = json.loads((self.work / "model/result.json").read_text())
            if not receipt["boundary"]["identical_hidden"]:
                path = self.work / "model/head-projections.safetensors"
                if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["boundary"]["projections_sha256"]:
                    raise ValueError("CUDA head projections changed")
            self.events.append("hidden_boundary_verified")
        self.events.append("container_command_completed")

    def cleanup(self, errors):
        stopped = False
        try:
            for cid in self.containers(owned=True):
                if self.hidden_boundary and not self.start_requested:
                    self.call(["docker", "rm", cid], 15)
                    self.events.append("unstarted_owned_container_removed")
                    continue
                try:
                    self.call(["docker", "stop", "--time", "5", cid], 20)
                except Exception as error:
                    not_found = "No such container" in str(error) or "No such object" in str(error)
                    if not_found and not self.containers(owned=True):
                        self.events.append("auto_removed_before_stop")
                        continue
                    raise
                self.events.append("owned_container_stop_completed")
                try:
                    self.call(["docker", "wait", cid], 15)
                    self.events.append("owned_container_wait_completed")
                except Exception as error:
                    # --rm may remove it between stop and wait. Docker stop already
                    # waited for termination. Require independent absence evidence.
                    absent = not self.containers(owned=True)
                    if absent and ("No such container" in str(error) or "No such object" in str(error)):
                        self.events.append("auto_removed_after_stop")
                    else:
                        raise
                if self.containers(owned=True):
                    self.call(["docker", "rm", cid], 15)
            stopped = not self.containers(owned=True)
            if not stopped:
                raise RuntimeError("Owned container still exists")
            self.events.append("owned_container_absent")
        except Exception as error:
            errors.append(f"container cleanup: {error}")
        if self.process is not None:
            reaped = False
            try:
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=5)
                    errors.append("Docker attach required termination during cleanup")
                reaped = True
            except Exception as error:
                errors.append(f"Docker attach cleanup: {error}")
            if reaped:
                self.events.append("docker_attach_reaped")
        if self.log is not None:
            self.log.close()
        if self.pause_requested:
            if stopped:
                try:
                    self.call(["sudo", "-n", "systemctl", "start", SERVICE], 100)
                    self.events.append("service_start_completed")
                except Exception as error:
                    errors.append(f"service restoration: {error}")
            else:
                errors.append("Service restoration withheld: owned container termination unconfirmed")
        try:
            self.health(300 if self.pause_requested else 10)
            if self.hidden_boundary and self.pause_requested:
                self.production_image()
        except Exception as error:
            errors.append(f"final health: {error}")
        if self.hidden_boundary:
            try:
                self.source_hash()
            except Exception as error:
                errors.append(f"final source verification: {error}")

    def run(self):
        original = None
        errors = []
        try:
            self.execute()
        except BaseException as error:
            original = f"{type(error).__name__}: {error}"
        finally:
            previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in SIGNALS}
            try:
                self.cleanup(errors)
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
        report = {"mode": self.mode, "fp32_rope": self.fp32_rope, "exact_prefix": self.exact_prefix,
                  "hidden_boundary": self.hidden_boundary, "midpoint": self.midpoint, "name": self.name, "cid": self.cid,
                  "owner_label": self.owner, "original_error": original,
                  "cleanup_errors": errors, "events": self.events}
        if self.created_work:
            (self.work / "lifecycle.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        return int(original is not None or bool(errors))


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "shapes", "decoder", "score", "hidden-kernel"))
    parser.add_argument("--root", type=Path, default=Path("/tmp/lalia-moss-cuda-20261027-r4"))
    parser.add_argument("--fp32-rope", action="store_true")
    parser.add_argument("--exact-prefix", action="store_true")
    parser.add_argument("--hidden-boundary", action="store_true")
    parser.add_argument("--midpoint", choices=("capture", "self", "cross"))
    parser.add_argument("--shared-inputs", type=Path)
    args = parser.parse_args()
    for sig in SIGNALS:
        signal.signal(sig, interrupted)
    raise SystemExit(Supervisor(args.root, args.mode, fp32_rope=args.fp32_rope, exact_prefix=args.exact_prefix,
                               hidden_boundary=args.hidden_boundary, midpoint=args.midpoint,
                               shared_inputs=args.shared_inputs).run())


if __name__ == "__main__":
    main()
