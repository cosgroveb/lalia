#!/usr/bin/env python3
"""Bound and clean up one MOSS smoke or score container."""
import argparse
import json
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid

SERVICE = "vllm-vllm.service"
IMAGE = "lalia-moss-cuda-probe:20261027-r4"
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


class Supervisor:
    def __init__(self, root, mode, call=command, popen=subprocess.Popen, sleep=time.sleep, *, fp32_rope=False):
        if fp32_rope and mode != "score":
            raise ValueError("--fp32-rope requires score mode")
        self.fp32_rope = fp32_rope
        self.root, self.mode = root.resolve(), mode
        self.work = self.root / "results" / mode
        self.cidfile = self.work / "container.cid"
        self.name = f"lalia-moss-cuda-{mode}-20261027"
        self.owner = uuid.uuid4().hex
        self.call, self.popen, self.sleep = call, popen, sleep
        self.cid = None
        self.process = None
        self.log = None
        self.created_work = False
        self.pause_requested = False
        self.events = []

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
        self.work.mkdir(parents=True, mode=0o700)
        self.created_work = True
        seconds, memory = {"smoke": (45, "2g"), "shapes": (120, "6g"), "decoder": (100, "4g"),
                           "score": (300, "24g")}[self.mode]
        script = "/probe/decoder_smoke.py" if self.mode == "decoder" else "/probe/score_decoder.py"
        args = ["docker", "create", "--init", "--rm", "--name", self.name,
                "--cidfile", str(self.cidfile), "--label", f"{LABEL}={self.owner}",
                "--gpus", "all", "--network", "none", "--memory", memory,
                "--memory-swap", memory, "--cpus", "4",
                "--volume", f"{self.root / 'probe'}:/probe:ro",
                "--volume", f"{self.work}:/results",
                "--volume", f"{self.root / 'inputs'}:/inputs:ro",
                "--entrypoint", "timeout", IMAGE, "--signal=TERM", "--kill-after=5s",
                f"{seconds}s", "python3", script]
        args += {"smoke": ["--cuda-smoke"], "shapes": ["--cuda-shapes"],
                 "decoder": [], "score": ["--inputs", "/inputs", "--out", "/results/model"]}[self.mode]
        if self.fp32_rope:
            args.append("--fp32-rope")
        self.call(args, 30)
        self.cid = self.cidfile.read_text().strip()
        if not re.fullmatch(r"[a-f0-9]{64}", self.cid) or self.containers(owned=True) != [self.cid]:
            raise ValueError("Created container identity differs")
        self.events.append("owned_container_created")

    def execute(self):
        self.prepare()
        if self.mode == "score":
            self.pause_requested = True
            self.call(["sudo", "-n", "systemctl", "stop", SERVICE], 100)
            self.events.append("service_stop_completed")
        self.log = (self.work / "container.log").open("w")
        self.process = self.popen(["docker", "start", "--attach", self.cid],
                                  stdout=self.log, stderr=subprocess.STDOUT)
        result = self.process.wait(timeout={"smoke": 60, "shapes": 140, "decoder": 120, "score": 320}[self.mode])
        if result:
            raise RuntimeError(f"Container attach exited {result}")
        from smoke_numeric import reject_fatal_diagnostics
        self.log.flush()
        reject_fatal_diagnostics(self.work / "container.log")
        self.events.append("container_command_completed")

    def cleanup(self, errors):
        stopped = False
        try:
            for cid in self.containers(owned=True):
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
        except Exception as error:
            errors.append(f"final health: {error}")

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
        report = {"mode": self.mode, "fp32_rope": self.fp32_rope, "name": self.name, "cid": self.cid,
                  "owner_label": self.owner, "original_error": original,
                  "cleanup_errors": errors, "events": self.events}
        if self.created_work:
            (self.work / "lifecycle.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        return int(original is not None or bool(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "shapes", "decoder", "score"))
    parser.add_argument("--root", type=Path, default=Path("/tmp/lalia-moss-cuda-20261027-r4"))
    parser.add_argument("--fp32-rope", action="store_true")
    args = parser.parse_args()
    for sig in SIGNALS:
        signal.signal(sig, interrupted)
    raise SystemExit(Supervisor(args.root, args.mode, fp32_rope=args.fp32_rope).run())


if __name__ == "__main__":
    main()
