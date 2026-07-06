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

import os
import queue
import subprocess
import sys
import threading


def run_on_gpus(jobs, ngpu, base_gpu=0, jobs_per_gpu=1):
    """Run worker commands across GPUs with a dynamic queue.

    Args:
        jobs: list of (label, cmd) — cmd is the argv list WITHOUT --gpu;
              the assigned device index is appended here.
        ngpu: number of GPUs; worker i uses GPU base_gpu + (i % ngpu).
        base_gpu: first GPU index.
        jobs_per_gpu: concurrent workers PER GPU (oversubscription). The
            benchmark's models are tiny, so a single job starves the GPU
            between kernel launches (host-bound); stacking 2–4 jobs per
            GPU raises throughput. Total workers = ngpu * jobs_per_gpu.

    CPU-thread pinning: each worker subprocess gets OMP_NUM_THREADS /
    MKL_NUM_THREADS = max(1, cpu_count // total_workers), so N torch
    processes don't each spawn cpu_count intra-op threads and thrash the
    cores. (Pinning covers only THIS orchestrator's workers — if several
    orchestrator processes share the box, their pools don't see each
    other; budget accordingly.)

    Returns:
        list of labels whose worker exited non-zero (they stay "missing"
        in the results CSV, so a re-run retries exactly those).
    """
    q = queue.Queue()
    for job in jobs:
        q.put(job)

    failed = []
    print_lock = threading.Lock()

    ngpu = max(1, int(ngpu))
    jobs_per_gpu = max(1, int(jobs_per_gpu))
    n_workers = ngpu * jobs_per_gpu
    omp = max(1, (os.cpu_count() or n_workers) // n_workers)
    env = {**os.environ,
           "OMP_NUM_THREADS": str(omp), "MKL_NUM_THREADS": str(omp)}

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
            ret = subprocess.run(full_cmd, env=env)
            if ret.returncode != 0:
                with print_lock:
                    print(f"  [WARN] [gpu{gpu_id}] worker returned "
                          f"{ret.returncode} for: {label}; continuing",
                          flush=True)
                    failed.append(label)
            q.task_done()

    threads = [
        threading.Thread(target=gpu_worker,
                         args=(base_gpu + (i % ngpu),), daemon=True)
        for i in range(n_workers)
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
