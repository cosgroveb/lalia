"""Reuse the tiny kernel check under the boundary trial's host UID and caches."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import stat

from official_kernel_preflight import main as kernel_check


def main():
    os.umask(0o077)
    directory = Path("/results/private-output")
    directory.mkdir(mode=0o700)
    output = directory / "check.json"
    output.write_text(json.dumps({"uid": os.getuid(), "gid": os.getgid()}) + "\n")
    ownership = {"uid": directory.stat().st_uid, "gid": directory.stat().st_gid,
                 "directory_mode": stat.S_IMODE(directory.stat().st_mode),
                 "file_uid": output.stat().st_uid, "file_gid": output.stat().st_gid,
                 "file_mode": stat.S_IMODE(output.stat().st_mode)}
    if ownership != {"uid": os.getuid(), "gid": os.getgid(), "directory_mode": 0o700,
                     "file_uid": os.getuid(), "file_gid": os.getgid(), "file_mode": 0o600}:
        raise ValueError("Private output ownership or modes differ")
    captured = io.StringIO()
    with redirect_stdout(captured):
        kernel_check()
    receipt = json.loads(captured.getvalue())
    receipt["ownership"] = ownership
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
