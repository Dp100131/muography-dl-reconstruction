#!/usr/bin/env python3
"""Runs a command and records its resource usage (Python >= 3.9, standard library only).

    python3 run_timed.py OUT.json -- COMMAND [ARGS...]

stdout and stderr are inherited (the caller redirects them to run.log). OUT.json (never
overwritten) gets wall time, user and system CPU time and the peak resident set size of the
command and its descendants. The exit status is the command's (128 + signal when killed).

Why not /usr/bin/time: it does not exist on the login node and sacct reported no TotalCPU for the
pilot jobs; getrusage(RUSAGE_CHILDREN) is measured per process, which is what the run table needs.
"""
import json
import os
import resource
import socket
import subprocess
import sys
import time


def main(argv):
    if len(argv) < 4 or argv[2] != "--":
        sys.stderr.write("usage: run_timed.py OUT.json -- COMMAND [ARGS...]\n")
        return 2
    out, cmd = argv[1], argv[3:]
    if os.path.exists(out):
        sys.stderr.write("refusing to overwrite %s\n" % out)
        return 2
    t_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.monotonic()
    try:
        rc = subprocess.call(cmd)
    except OSError as exc:
        sys.stderr.write("cannot run %s: %s\n" % (cmd[0], exc))
        rc = 127
    wall = time.monotonic() - t0
    if rc < 0:
        rc = 128 - rc
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = {
        "formato": "oe1-recursos/1",
        "orden": cmd,
        "codigo_de_salida": rc,
        "inicio_utc": t_utc,
        "fin_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": wall,
        "user_s": ru.ru_utime,
        "sys_s": ru.ru_stime,
        "maxrss_kb": ru.ru_maxrss,          # kilobytes on Linux
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    with open(out, "x") as fh:
        json.dump(record, fh, indent=1)
        fh.write("\n")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
