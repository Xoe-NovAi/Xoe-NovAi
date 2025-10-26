#!/usr/bin/env python3
# Xoe-NovAi Curation Worker
# Guide Ref: Section 8.2 (Pattern: Redis Job Queue Processing)
import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import subprocess
import redis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Config ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_KEY = os.getenv("QUEUE_KEY", "curation_queue")
JOB_PREFIX = os.getenv("JOB_PREFIX", "curation:")
LOG_DIR = Path(os.getenv("LOG_DIR", "/app/logs/curations"))
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data/curations"))
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))
WORKER_NAME = os.getenv("WORKER_NAME", "curation-worker-1")

LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","worker":"%(name)s","level":"%(levelname)s","msg":"%(message)s"}',
    handlers=[logging.FileHandler(LOG_DIR / f"{WORKER_NAME}.log"), logging.StreamHandler()]
)
logger = logging.getLogger(WORKER_NAME)

# --- Redis connection ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30),
       retry=retry_if_exception_type(redis.exceptions.ConnectionError))
def connect_redis() -> redis.Redis:
    logger.info("Connecting to Redis...")
    client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    client.ping()
    return client

rdb = connect_redis()

def get_job_data(job_id: str) -> dict:
    return rdb.hgetall(job_id)

def update_job_status(job_id: str, status: str, extra: dict = None):
    update = {"status": status, "updated_at": datetime.utcnow().isoformat()}
    if extra:
        update.update(extra)
    rdb.hset(job_id, mapping=update)
    logger.info(f"{job_id} status -> {status}")

def run_subprocess(job_id: str, command: list[str]):
    log_path = DATA_DIR / f"{job_id.replace(':', '_')}.log"
    with open(log_path, "w") as out:
        proc = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT)
        proc.wait()
        return proc.returncode

def process_job(job_id: str):
    meta = get_job_data(job_id)
    if not meta:
        logger.warning(f"No metadata for {job_id}")
        return

    attempts = int(meta.get("attempts", 0))
    if attempts >= MAX_ATTEMPTS:
        update_job_status(job_id, "failed", {"error": "max_attempts_reached"})
        return

    update_job_status(job_id, "processing", {"attempts": str(attempts + 1)})

    source = meta.get("source", "unknown")
    category = meta.get("category", "misc")
    query = meta.get("query", "")
    cmd = ["python3", "/app/XNAi_rag_app/crawl.py", "--source", source, "--category", category, "--query", query]

    try:
        ret = run_subprocess(job_id, cmd)
        if ret == 0:
            update_job_status(job_id, "completed")
        else:
            update_job_status(job_id, "failed", {"error": f"Exit {ret}"})
    except Exception as e:
        update_job_status(job_id, "failed", {"error": str(e)})

def main():
    logger.info(f"{WORKER_NAME} started, listening on {QUEUE_KEY}")
    while True:
        try:
            job = rdb.blpop(QUEUE_KEY, timeout=5)
            if not job:
                time.sleep(1)
                continue
            job_id = job[1]
            process_job(job_id)
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis connection lost: {e}")
            time.sleep(5)
            rdb = connect_redis()
        except Exception as e:
            logger.error(f"Unhandled exception: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()