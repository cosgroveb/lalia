"""Keep this Mac awake until a one-shot report is queued in the existing Codex task."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def interrupt(*_):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--at", type=dt.datetime.fromisoformat, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.at.tzinfo is None:
        parser.error("--at requires a timezone offset")
    marker = args.state / "report-queued.json"
    if marker.exists():
        return
    signal.signal(signal.SIGTERM, interrupt)
    awake = subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])
    (args.state / "report-process.json").write_text(json.dumps({
        "pid": os.getpid(), "caffeinate_pid": awake.pid, "at": args.at.isoformat(),
    }, indent=2) + "\n")
    try:
        while time.time() < args.at.timestamp():
            time.sleep(max(0, min(30, args.at.timestamp() - time.time())))
        prompt = (
            "Morning report due for the authorized overnight Lalia speaker-ASR experiment. "
            f"Read aggregate results and handoff in {args.state}. "
            "Review public accuracy/speaker metrics, timing, physical footprint and failures. "
            "Private audio/transcripts must remain local; use aggregate diagnostics only. "
            "Complete any necessary bounded checks, give Brian the recommendation and measured limits, "
            "update GBrain, verify production config unchanged, and clean up task-owned experiment/report "
            "LaunchAgents and processes. Worktree ~/code/cosgroveb/lalia-speaker-eval. "
            "Do not mistake launch success for completed inference. Report actual results in this thread."
        )
        for attempt in range(5):
            try:
                result = subprocess.run(["codex", "queue", "--thread", args.thread, "--message", prompt],
                                        check=True, capture_output=True, text=True, timeout=60)
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                (args.state / "report-delivery-error.txt").write_text(f"Attempt {attempt + 1}: {error}\n")
                if attempt == 4:
                    raise
                time.sleep(30)
        marker.write_text(json.dumps({"queued_at": dt.datetime.now().astimezone().isoformat(),
                                     "receipt": result.stdout.strip()}, indent=2) + "\n")
        print(result.stdout.strip(), flush=True)
        # Leave time for the report turn to read artifacts even if the display is asleep.
        finish = time.time() + 1800
        while time.time() < finish:
            time.sleep(max(0, min(30, finish - time.time())))
    finally:
        awake.terminate()
        awake.wait(timeout=10)


if __name__ == "__main__":
    main()
