# -*- coding: utf-8 -*-
"""
Dynamic multi-GPU dispatch for the orchestrators (CF-JEPA-style queue,
minus torchrun/fcntl: one orchestrator process owns an in-memory queue and
runs one worker subprocess per GPU; a GPU grabs the next job the moment it
frees up, so a slow model on one GPU never idles the other).

Used by run_forecast.py / run_lr_search.py via `--ngpu N`:
  ngpu=1 -> identical to the old sequential behavior (same GPU, same order).
  ngpu=N -> jobs run on GPU indices base_gpu .. base_gpu+N-1.

Worker stdout/stderr stream directly to the console, so lines from
concurrent workers interleave — same as running two terminals by hand.
"""

import queue
import subprocess
import sys
import threading


def run_on_gpus(jobs, ngpu, base_gpu=0):
    """Run worker commands across GPUs with a dynamic queue.

    Args:
        jobs: list of (label, cmd) — cmd is the argv list WITHOUT --gpu;
              the assigned device index is appended here.
        ngpu: number of concurrent workers (one per GPU).
        base_gpu: first GPU index; workers use base_gpu .. base_gpu+ngpu-1.

    Returns:
        list of labels whose worker exited non-zero (they stay "missing"
        in the results CSV, so a re-run retries exactly those).
    """
    q = queue.Queue()
    for job in jobs:
        q.put(job)

    failed = []
    print_lock = threading.Lock()

    def gpu_worker(gpu_id):
        while True:
            try:
                label, cmd = q.get_nowait()
            except queue.Empty:
                return
            full_cmd = list(cmd) + ["--gpu", str(gpu_id)]
            with print_lock:
                print(f"\n>>> [gpu{gpu_id}] {label}", flush=True)
                print("  CMD:", " ".join(full_cmd), flush=True)
            ret = subprocess.run(full_cmd)
            if ret.returncode != 0:
                with print_lock:
                    print(f"  [WARN] [gpu{gpu_id}] worker returned "
                          f"{ret.returncode} for: {label}; continuing",
                          flush=True)
                    failed.append(label)
            q.task_done()

    ngpu = max(1, int(ngpu))
    threads = [
        threading.Thread(target=gpu_worker, args=(base_gpu + i,), daemon=True)
        for i in range(ngpu)
    ]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        # Ctrl+C: the workers are child processes and receive the signal
        # too; just surface the interruption to the caller.
        print("\n[interrupted] in-flight runs are lost; finished runs are "
              "already in the CSV — re-run to resume.", file=sys.stderr)
        raise
    return failed
