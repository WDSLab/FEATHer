# -*- coding: utf-8 -*-
"""
Dynamic multi-GPU dispatch: one process owns an in-memory queue and runs
worker subprocesses; a GPU grabs the next job the moment it frees up, so a
slow model on one GPU never idles the other.

Used by run.py via `--ngpu N`:
  ngpu=1 -> sequential on one GPU, in job order.
  ngpu=N -> jobs run on GPU indices base_gpu .. base_gpu+N-1.

Worker stdout/stderr stream directly to the console, so lines from
concurrent workers interleave — same as running two terminals by hand.
"""

import os
import queue
import shlex
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
    cores. (Pinning covers only this process's workers; several runners
    on one box do not see each other.)

    Returns:
        list of labels whose worker exited non-zero (they stay "missing"
        in the results CSV, so a re-run retries exactly those).
    """
    q = queue.Queue()
    for job in jobs:
        q.put(job)

    failed = []
    print_lock = threading.Lock()

    # Single-Ctrl+C shutdown: `stop` tells worker threads to stop grabbing
    # new jobs; `live_procs` lets the main thread terminate in-flight children.
    # Without it, a Ctrl+C that kills a child (Windows sends it to the whole
    # console group) looks like a failed job and the loop launches the next one.
    stop = threading.Event()
    live_procs = set()
    proc_lock = threading.Lock()

    def _terminate(p):
        try:
            p.terminate()
        except Exception:
            pass

    ngpu = max(1, int(ngpu))
    jobs_per_gpu = max(1, int(jobs_per_gpu))
    n_workers = ngpu * jobs_per_gpu
    omp = int(os.environ.get("FORCE_OMP", max(1, (os.cpu_count() or n_workers) // n_workers)))
    env = {**os.environ,
           "OMP_NUM_THREADS": str(omp), "MKL_NUM_THREADS": str(omp)}

    def gpu_worker(gpu_id):
        while not stop.is_set():
            try:
                label, cmd = q.get_nowait()
            except queue.Empty:
                return
            if stop.is_set():
                q.task_done()
                return
            full_cmd = list(cmd) + ["--gpu", str(gpu_id)]
            with print_lock:
                print(f"\n>>> [gpu{gpu_id}] {label}", flush=True)
                print("  CMD:", " ".join(shlex.quote(c) for c in full_cmd), flush=True)   # quoted: exp_tag contains "|"
            # If Popen itself raises (fd exhaustion, ENOMEM, exec failure) the
            # thread would die and its dequeued job would vanish silently;
            # record it as failed and keep pulling jobs instead.
            p = None
            try:
                p = subprocess.Popen(full_cmd, env=env)
                with proc_lock:
                    live_procs.add(p)
                # If Ctrl+C landed in the tiny window while we were launching,
                # kill this child immediately (main thread may have already
                # snapshotted live_procs before we added ours).
                if stop.is_set():
                    _terminate(p)
                ret = p.wait()
            except Exception as e:
                with print_lock:
                    print(f"  [WARN] [gpu{gpu_id}] launch failed for {label}: "
                          f"{e!r}; marking failed, continuing", flush=True)
                    failed.append(label)
                if p is not None:
                    _terminate(p)
                    with proc_lock:
                        live_procs.discard(p)
                q.task_done()
                continue
            with proc_lock:
                live_procs.discard(p)
            if stop.is_set():
                q.task_done()
                return
            if ret != 0:
                with print_lock:
                    print(f"  [WARN] [gpu{gpu_id}] worker returned "
                          f"{ret} for: {label}; continuing",
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
        # Interruptible join: a bare t.join() on the main thread can swallow
        # the interrupt window; poll with a short timeout so Ctrl+C is caught
        # promptly.
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.2)
    except KeyboardInterrupt:
        # One Ctrl+C: stop the queue and hard-stop every in-flight child so
        # nothing new launches and no worker is orphaned.
        stop.set()
        with proc_lock:
            procs = list(live_procs)
        for p in procs:
            _terminate(p)
        print("\n[interrupted] in-flight runs killed; finished runs are "
              "already in the CSV — re-run to resume.", file=sys.stderr)
        raise
    return failed
