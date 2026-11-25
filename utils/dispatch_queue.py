# -*- coding: utf-8 -*-
"""Run worker commands on GPUs from one shared queue: a worker takes the next job as soon as it is free.

Worker output goes straight to the console (lines of concurrent workers interleave).  Each worker gets
OMP / MKL threads = cpu_count // number of workers (env FORCE_OMP overrides), so the processes do not thrash the cores.
"""
import os
import queue
import shlex
import subprocess
import sys
import threading


def run_on_gpus(jobs, ngpu, base_gpu=0, jobs_per_gpu=1):
    """jobs = [(label, argv without --gpu)]; ngpu * jobs_per_gpu workers, worker i on GPU base_gpu + i % ngpu.
    Returns the labels whose command failed (their cells stay missing, so a re-run retries exactly those).
    One Ctrl+C stops the queue and kills the running children."""
    q = queue.Queue()
    for job in jobs:
        q.put(job)
    failed, live = [], set()
    lock = threading.Lock()
    stop = threading.Event()
    n_workers = max(1, int(ngpu)) * max(1, int(jobs_per_gpu))
    omp = os.environ.get("FORCE_OMP", str(max(1, (os.cpu_count() or n_workers) // n_workers)))
    env = {**os.environ, "OMP_NUM_THREADS": omp, "MKL_NUM_THREADS": omp}

    def worker(gpu):
        while not stop.is_set():
            try:
                label, cmd = q.get_nowait()
            except queue.Empty:
                return
            cmd = list(cmd) + ["--gpu", str(gpu)]
            with lock:
                print(f"\n>>> [gpu{gpu}] {label}\n  CMD: " + " ".join(shlex.quote(c) for c in cmd), flush=True)
            try:
                p = subprocess.Popen(cmd, env=env)
            except OSError as e:                        # the job is recorded, the worker keeps going
                with lock:
                    print(f"  [WARN] [gpu{gpu}] launch failed for {label}: {e!r}", flush=True)
                    failed.append(label)
                continue
            with lock:
                live.add(p)
            if stop.is_set():                           # Ctrl+C came while launching
                p.terminate()
            ret = p.wait()
            with lock:
                live.discard(p)
                if ret != 0 and not stop.is_set():
                    print(f"  [WARN] [gpu{gpu}] worker returned {ret} for: {label}", flush=True)
                    failed.append(label)

    threads = [threading.Thread(target=worker, args=(base_gpu + i % max(1, int(ngpu)),), daemon=True)
               for i in range(n_workers)]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):       # join with a timeout so Ctrl+C is caught at once
            for t in threads:
                t.join(timeout=0.2)
    except KeyboardInterrupt:
        stop.set()
        with lock:
            for p in live:
                p.terminate()
        print("\n[interrupted] running workers killed; finished runs are in the CSV - re-run to resume.",
              file=sys.stderr)
        raise
    return failed
