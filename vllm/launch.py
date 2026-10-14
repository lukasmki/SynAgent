"""Launch the multi-model vLLM servers described in config.yaml.

Usage: python launch.py <config.yaml> [server ...] [--dry-run]
Normally run through start.sh, which sets up the venv and cache paths.

Each server gets its own `vllm serve --config` file and log under $SYNAGENT_LOG_DIR
(default $PWD/logs/). Servers that share a GPU are started one at a time:
vLLM sizes its memory budget from what is free at startup, so two instances profiling the
same GPU at once can fail or over-allocate. Ctrl-C (or SIGTERM) stops all of them.
"""
import os
import signal
import subprocess
import sys
import time
import urllib.request

import yaml

STARTUP_TIMEOUT_S = 1800


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def tail(path, n=40):
    with open(path, errors="replace") as f:
        return "".join(f.readlines()[-n:])


def main():
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry_run = "--dry-run" in sys.argv
    cfg_path, wanted = args[0], args[1:]
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    defaults = cfg.get("defaults") or {}
    servers = cfg["servers"]
    unknown = set(wanted) - set(servers)
    if unknown:
        sys.exit(f"unknown server(s): {', '.join(sorted(unknown))}; have {', '.join(servers)}")
    names = wanted or list(servers)

    run_dir = os.environ.get("SYNAGENT_LOG_DIR", os.path.expandvars("$PWD/logs"))
    os.makedirs(run_dir, exist_ok=True)
    plan = {}
    for name in names:
        spec = servers[name]
        opts = {**defaults, **spec["vllm"]}
        conf = os.path.join(run_dir, f"{name}.yaml")
        with open(conf, "w") as f:
            f.write(f"# generated from {os.path.abspath(cfg_path)} [{name}]; edit that file instead\n")
            yaml.safe_dump(opts, f, sort_keys=False)
        plan[name] = dict(gpus=set(spec["gpus"]), port=opts["port"], model=opts["model"], conf=conf,
                          log=os.path.join(run_dir, f"{name}.log"))
        gpus = ",".join(map(str, spec["gpus"]))
        print(f"{name:12s} GPU {gpus:4s} :{opts['port']}  {opts['model']}  (config {conf})")
    if dry_run:
        return

    procs, starting = {}, {}

    def stop_all(code=1, *_):
        for p in procs.values():
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)
        deadline = time.time() + 60
        for p in procs.values():
            try:
                p.wait(timeout=max(1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        sys.exit(code)

    signal.signal(signal.SIGINT, lambda *_: stop_all(0))
    signal.signal(signal.SIGTERM, lambda *_: stop_all(0))

    pending = list(names)
    t0 = time.time()
    while pending or starting:
        for name in list(pending):
            s = plan[name]
            if any(s["gpus"] & plan[o]["gpus"] for o in starting):
                continue
            if healthy(s["port"]):
                print(f"port {s['port']} is already serving; stop that server first", file=sys.stderr)
                stop_all()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, sorted(s["gpus"]))))
            with open(s["log"], "w") as log:
                procs[name] = subprocess.Popen(["vllm", "serve", "--config", s["conf"]], env=env,
                                               stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
            starting[name] = time.time()
            pending.remove(name)
            print(f"[{time.time() - t0:5.0f}s] starting {name} (log {s['log']})", flush=True)
        for name in list(starting):
            s = plan[name]
            if procs[name].poll() is not None:
                print(f"\n{name} exited during startup (code {procs[name].returncode}); "
                      f"last lines of {s['log']}:\n{tail(s['log'])}", file=sys.stderr)
                stop_all()
            if healthy(s["port"]):
                print(f"[{time.time() - t0:5.0f}s] {name} ready on :{s['port']}", flush=True)
                del starting[name]
            elif time.time() - starting[name] > STARTUP_TIMEOUT_S:
                print(f"{name} not healthy after {STARTUP_TIMEOUT_S}s; see {s['log']}", file=sys.stderr)
                stop_all()
        time.sleep(2)

    host = os.uname().nodename
    print(f"\nAll servers up on {host}:")
    for name in names:
        print(f"  {name:12s} http://{host}:{plan[name]['port']}/v1   model={plan[name]['model']}")
    print("Ctrl-C to stop.", flush=True)
    while True:
        for name, p in procs.items():
            if p.poll() is not None:
                print(f"{name} exited (code {p.returncode}); stopping the rest. "
                      f"Last lines of {plan[name]['log']}:\n{tail(plan[name]['log'])}", file=sys.stderr)
                stop_all()
        time.sleep(5)


if __name__ == "__main__":
    main()
