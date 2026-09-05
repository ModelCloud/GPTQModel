"""Dispatch explicit experiment commands to idle GPUs; persist outcomes.

Commands are argv arrays, never shell strings. Dependency reports must be
complete, and export-dependent jobs require recorded export-reload equality.
A failed command is recorded and not automatically restarted. Existing foreign
GPU workers are respected, including the long-running FP32 teacher evaluation.
"""

import argparse
import fcntl
import json
import os
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def idle(uuid):
    try:
        row = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=" + uuid,
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).split(",")
        apps = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return int(row[0]) <= 8 and int(row[1]) == 0 and uuid not in apps
    except (OSError, subprocess.CalledProcessError, ValueError):
        return False


def ready(job):
    for path in job.get("requires_complete", []):
        data = read_json(path)
        if not data or not data.get("complete"):
            return False
    if "export" in job:
        data = read_json(job["export_report"])
        if not data or not data.get("complete"):
            return False
        candidates = [
            c for c in data.get("candidates", []) if c["export"] == job["export"]
        ]
        if len(candidates) != 1 or not Path(job["export"]).is_file():
            return False
        rows = candidates[0].get("rows", [])
        if len(rows) != 9 or not all(r.get("reload_equal") for r in rows):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as resources:
        lock = resources.enter_context(open(str(args.state) + ".lock", "w"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = read_json(args.manifest)
        if not config:
            parser.error("Invalid queue manifest")
        previous = read_json(args.state)
        if previous and any(j["status"] == "running" for j in previous["jobs"]):
            raise RuntimeError(
                "Previous running jobs require explicit reconciliation; never duplicate them"
            )
        jobs = config["jobs"]
        if len({j["id"] for j in jobs}) != len(jobs):
            raise ValueError("Duplicate job IDs")
        records = {j["id"]: {"id": j["id"], "status": "pending"} for j in jobs}
        if previous:
            for r in previous["jobs"]:
                if r["id"] in records and r["status"] in ("complete", "failed"):
                    records[r["id"]] = r
        live = {}

        def save():
            state = {
                "dispatcher_pid": os.getpid(),
                "updated_unix": time.time(),
                "manifest": str(args.manifest),
                "jobs": list(records.values()),
            }
            tmp = args.state.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=2) + "\n")
            tmp.replace(args.state)

        while True:
            # Permit append-only queue expansion without interrupting GPU workers.
            latest = read_json(args.manifest)
            if latest:
                known = {j["id"]: j for j in jobs}
                for job in latest["jobs"]:
                    if job["id"] in known:
                        if job != known[job["id"]]:
                            raise ValueError("Existing queue commands are immutable; append a new job ID")
                    else:
                        jobs.append(job)
                        known[job["id"]] = job
                        records[job["id"]] = {"id": job["id"], "status": "pending"}
            for job_id, (process, gpu, log) in list(live.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                record = records[job_id]
                job = next(j for j in jobs if j["id"] == job_id)
                report = read_json(job["result"])
                record.update(
                    status="complete"
                    if code == 0 and report and report.get("complete")
                    else "failed",
                    returncode=code,
                    finished_unix=time.time(),
                )
                del live[job_id]
                print("FINISHED", job_id, record["status"], flush=True)
            occupied = {gpu["uuid"] for _, gpu, _ in live.values()}
            for gpu in config["gpus"]:
                if gpu["uuid"] in occupied or not idle(gpu["uuid"]):
                    continue
                job = next(
                    (
                        j
                        for j in jobs
                        if records[j["id"]]["status"] == "pending"
                        and (not j.get("gpu_uuid") or j["gpu_uuid"] == gpu["uuid"])
                        and ready(j)
                    ),
                    None,
                )
                if job is None:
                    continue
                # Existing results must be reviewed, never overwritten by a queue retry.
                if Path(job["result"]).exists():
                    records[job["id"]].update(
                        status="failed",
                        reason="Result already exists; review before scheduling",
                    )
                    continue
                command = [
                    str(v).replace("{uuid}", gpu["uuid"]) for v in job["command"]
                ]
                env = os.environ.copy()
                env.update(config.get("env", {}))
                env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
                env["GPTQMODEL_TORCH_EXTENSIONS_DIR"] = gpu["jit_cache"]
                Path(job["log"]).parent.mkdir(parents=True, exist_ok=True)
                log = resources.enter_context(open(job["log"], "w"))
                proc = subprocess.Popen(
                    command,
                    cwd=config["cwd"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                records[job["id"]].update(
                    status="running",
                    pid=proc.pid,
                    uuid=gpu["uuid"],
                    command=command,
                    started_unix=time.time(),
                    log=job["log"],
                )
                live[job["id"]] = (proc, gpu, log)
                print("STARTED", job["id"], proc.pid, gpu["uuid"], flush=True)
            save()
            if all(r["status"] in ("complete", "failed") for r in records.values()):
                return
            time.sleep(5)


if __name__ == "__main__":
    main()
