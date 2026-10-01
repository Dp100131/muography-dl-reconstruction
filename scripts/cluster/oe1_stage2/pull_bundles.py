#!/usr/bin/env python3
"""OE1 stage 2: download the files that corridas.sbatch exports through the /home buffer of SC3.

Runs on the students' machine. Every round lists the `<name>.listo` markers of the remote buffer, copies
`<name>` and `<name>.sha256` with scp, checks the SHA-256 of the local copy and only then removes the three remote
files. A file whose check fails is downloaded again in the next round (three attempts, then the script stops
without removing it). It ends when --expect files have been received or after --max-hours.

  python pull_bundles.py --host sc3-guane --remote oe1-trabajo/etapa2/salida --local DIR --expect 85
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time


def ssh(host, command, timeout=120):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], capture_output=True, text=True,
                          timeout=timeout)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ready(host, remote):
    r = ssh(host, 'cd %s && for f in *.listo; do [ -e "$f" ] && echo "${f%%.listo}"; done' % remote)
    if r.returncode != 0:
        return None
    return [x.strip() for x in r.stdout.splitlines() if x.strip()]


def fetch(host, remote, local, name):
    part = os.path.join(local, name + ".part")
    shaf = os.path.join(local, name + ".sha256")
    for src, dst in ((name, part), (name + ".sha256", shaf)):
        r = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "%s:%s/%s" % (host, remote, src), dst],
                           capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            return False, "scp %s: %s" % (src, r.stderr.strip()[:200])
    with open(shaf, "r", encoding="utf-8") as fh:
        want = fh.read().split()[0]
    got = sha256_file(part)
    if got != want:
        os.remove(part)
        return False, "sha256 distinto (%s != %s)" % (got, want)
    os.replace(part, os.path.join(local, name))
    r = ssh(host, "cd %s && rm -f -- %s %s.sha256 %s.listo" % (remote, name, name, name))
    if r.returncode != 0:
        return False, "copiado y verificado, pero no se pudo borrar en SC3: %s" % r.stderr.strip()[:200]
    return True, got


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="sc3-guane")
    p.add_argument("--remote", default="oe1-trabajo/etapa2/salida")
    p.add_argument("--local", required=True)
    p.add_argument("--expect", type=int, required=True)
    p.add_argument("--poll", type=int, default=20)
    p.add_argument("--max-hours", type=float, default=8.0)
    args = p.parse_args(argv)
    os.makedirs(args.local, exist_ok=True)
    log = os.path.join(args.local, "descargas.tsv")
    done = set()
    if os.path.isfile(log):
        with open(log, "r", encoding="utf-8") as fh:
            done = set(line.split("\t")[1] for line in fh if line.strip())
    failures = {}
    t_end = time.time() + args.max_hours * 3600
    print("%s inicio: %d ya descargados, esperados %d" % (time.strftime("%H:%M:%S"), len(done), args.expect), flush=True)
    while len(done) < args.expect and time.time() < t_end:
        try:
            names = ready(args.host, args.remote)
        except subprocess.TimeoutExpired:
            names = None
        for name in names or []:
            if name in done:
                continue
            try:
                ok, info = fetch(args.host, args.remote, args.local, name)
            except subprocess.TimeoutExpired:
                ok, info = False, "tiempo agotado"
            if ok:
                done.add(name)
                size = os.path.getsize(os.path.join(args.local, name))
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write("%s\t%s\t%d\t%s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), name, size, info))
                print("%s descargado %s (%.1f MB) [%d/%d]" % (time.strftime("%H:%M:%S"), name, size / 1e6, len(done),
                                                              args.expect), flush=True)
            else:
                failures[name] = failures.get(name, 0) + 1
                print("%s FALLO %s: %s (intento %d)" % (time.strftime("%H:%M:%S"), name, info, failures[name]),
                      flush=True)
                if failures[name] >= 3:
                    print("se detiene: %s falló tres veces y queda en SC3" % name, flush=True)
                    return 1
        if len(done) < args.expect:
            time.sleep(args.poll)
    print("%s fin: %d de %d" % (time.strftime("%H:%M:%S"), len(done), args.expect), flush=True)
    return 0 if len(done) >= args.expect else 1


if __name__ == "__main__":
    sys.exit(main())
