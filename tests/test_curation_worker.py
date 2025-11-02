# tests/test_worker.py
import json
import os
import tempfile
from pathlib import Path

import fakeredis
import redis
import pytest

# Small helper to simulate enqueuing a job and reading meta
def test_enqueue_job_and_meta(tmp_path):
    r = fakeredis.FakeRedis()
    job_id = "curation:job123"
    job_meta = {
        "source": "gutenberg",
        "category": "classics",
        "query": "Plato",
        "queued_at": "2025-10-23T00:00:00Z",
        "status": "queued",
        "attempts": "0"
    }
    # write meta
    r.hset(job_id, mapping=job_meta)
    # push to queue
    r.rpush("curation_queue", job_id)

    # pop job (simulate worker brpop)
    popped = r.rpop("curation_queue")
    assert popped == job_id

    meta = r.hgetall(job_id)
    assert meta.get("source") == "gutenberg"
    assert meta.get("category") == "classics"
    assert meta.get("query") == "Plato"
    assert meta.get("status") == "queued"
    assert meta.get("attempts") == "0"