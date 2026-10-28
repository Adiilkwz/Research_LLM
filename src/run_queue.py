"""Run a list of jobs one after another on ONE GPU, always with --resume, and stop cleanly before the
Kaggle 12-hour session limit so that the notebook can finish and its outputs are saved.
An interrupted run continues from its last milestone checkpoint in the next session.

Usage:
  python src/run_queue.py --gpu 0 --hours 11 --jobs gpt2-xl:ROME:0 gpt2-xl:ROME:1 gpt2-xl:none:0
  job format  model:method:seed[:single]
"""
import argparse
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--hours", type=float, default=11.0, help="stop launching/kill jobs after this many hours")
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--log_dir", default=os.path.join(ROOT, "results", "logs"))
    args = ap.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    deadline = time.time() + args.hours * 3600
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1")

    for job in args.jobs:
        parts = job.split(":")
        model, method, seed = parts[0], parts[1], parts[2]
        mode = parts[3] if len(parts) > 3 else "seq"
        if time.time() > deadline - 600:
            print(f"[queue gpu{args.gpu}] no time left, skipping {job}", flush=True)
            continue
        cmd = [sys.executable, os.path.join(ROOT, "src", "run_experiment.py"), "--model", model, "--method", method,
               "--seed", seed, "--mode", mode, "--resume"]
        log = os.path.join(args.log_dir, f"{model}_{mode}_{method}_seed{seed}.log")
        print(f"[queue gpu{args.gpu}] start {job} -> {log}", flush=True)
        with open(log, "a") as lf:
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            while p.poll() is None:
                if time.time() > deadline:
                    print(f"[queue gpu{args.gpu}] session time limit reached, stopping {job} "
                          f"(it will resume from its last milestone)", flush=True)
                    p.send_signal(signal.SIGTERM)
                    try:
                        p.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        p.kill()
                    break
                time.sleep(20)
        print(f"[queue gpu{args.gpu}] {job} exit code {p.returncode}", flush=True)


if __name__ == "__main__":
    main()
