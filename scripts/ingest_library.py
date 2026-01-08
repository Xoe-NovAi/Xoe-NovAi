#!/usr/bin/env python3
"""
ingest_library.py - Robust ingestion for stack-cat snapshots into XNAi RAG

This module is intended to be imported by the small wrapper script
`xnai-snapshot-ingest.py` (which calls ingest_library.main() or
instantiates SnapshotIngestor). It understands the snapshots produced by
scripts/stack-cat-md.sh (layout: scripts/stack-cat-files/<timestamp>/).

Features:
 - find_latest_snapshot(): find newest snapshot dir under scripts/stack-cat-files
 - ingest_snapshot(snapshot_dir): chunk markdown, produce Documents, add to FAISS
 - watch_for_snapshots(): optional watch loop to auto-ingest new snapshots
 - uses LlamaCppEmbeddings via langchain_community if available (retry-enabled)
 - caches ingestion metadata in Redis if available
 - robust error handling and logging
"""
from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# CRITICAL FIX: Import path resolution (Pattern 1)
# Add parent directory to path for imports from app/XNAi_rag_app
sys.path.insert(0, str(Path(__file__).parent.parent / "app" / "XNAi_rag_app"))

# Resilience
try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except Exception:
    # If tenacity missing, provide a no-op decorator fallback
    def retry(*_a, **_k):
        def _d(fn):
            return fn
        return _d
    stop_after_attempt = lambda n: None
    wait_exponential = lambda **kw: None

# Optional heavy deps
_HAS_LC = False
_HAS_FAISS = False
_HAS_LLAMA_EMB = False
_HAS_ORJSON = False
_HAS_TOML = False
_HAS_REDIS = False

try:
    # langchain core Document
    from langchain_core.documents import Document
    _HAS_LC = True
except Exception:
    # fallback: create a lightweight Document shim if necessary
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

try:
    # FAISS vectorstore from langchain-community
    from langchain_community.vectorstores import FAISS
    _HAS_FAISS = True
except Exception:
    FAISS = None

try:
    # LlamaCppEmbeddings (langchain-community)
    from langchain_community.embeddings import LlamaCppEmbeddings
    _HAS_LLAMA_EMB = True
except Exception:
    LlamaCppEmbeddings = None

try:
    import orjson as _orjson
    _HAS_ORJSON = True
except Exception:
    _orjson = None

try:
    import toml
    _HAS_TOML = True
except Exception:
    toml = None

try:
    from redis import Redis
    _HAS_REDIS = True
except Exception:
    Redis = None

# Configure logging
logger = logging.getLogger("ingest_library")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Utilities
def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_prefix(s: str, n: int = 8) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def read_text_safe(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        try:
            return p.read_text(encoding="latin-1")
        except Exception:
            return ""


# Embedding initialization with retry to handle transient problems
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _init_llama_embeddings(model_path: str, n_threads: int = 6, verbose: bool = False):
    if not _HAS_LLAMA_EMB:
        raise RuntimeError("LlamaCppEmbeddings not available (install langchain-community)")
    if verbose:
        logger.info(f"[embeddings] Initializing LlamaCppEmbeddings model: {model_path}")
    emb = LlamaCppEmbeddings(model_path=model_path, n_threads=n_threads, verbose=False)
    return emb


class SnapshotIngestor:
    def __init__(
        self,
        config_path: str = "config.toml",
        index_root: str = "knowledge/coder/stack_snapshots",
        embedding_model_env: str = "EMBEDDING_MODEL_PATH",
        default_embedding_model: str = "embeddings/all-MiniLM-L12-v2.Q8_0.gguf",
        n_threads: int = 6,
        verbose: bool = False,
    ):
        self.config_path = config_path
        self.index_root = Path(index_root)
        self.n_threads = n_threads
        self.verbose = verbose

        # Load config (toml) if available
        self.config = self._load_config(config_path)

        # Determine embedding model path from env or config or default
        self.embedding_model_path = os.getenv(
            embedding_model_env,
            self.config.get("models", {}).get("embedding_path", default_embedding_model)
            if isinstance(self.config, dict) else default_embedding_model
        )

        # Initialize embeddings (retry-enabled). If not available, proceed without embeddings.
        self.embeddings = None
        try:
            if _HAS_LLAMA_EMB:
                self.embeddings = _init_llama_embeddings(self.embedding_model_path, n_threads=self.n_threads, verbose=self.verbose)
            else:
                logger.warning("[ingest] LlamaCppEmbeddings not installed - continuing without embeddings (will create vectorstore but embeddings may be missing)")
                self.embeddings = None
        except Exception as e:
            logger.warning(f"[ingest] Embeddings initialization failed: {e}")
            self.embeddings = None

        # Initialize Redis if available
        self.redis_client = None
        if _HAS_REDIS:
            try:
                self.redis_client = Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    password=os.getenv("REDIS_PASSWORD", None),
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                )
                # quick ping with timeout handling
                self.redis_client.ping()
                logger.info("[ingest] Redis connected")
            except Exception as e:
                logger.warning(f"[ingest] Redis initialization failed: {e}")
                self.redis_client = None
        else:
            if self.verbose:
                logger.debug("[ingest] redis-py not installed; skipping Redis caching")

        # Vectorstore path (config override supported)
        self.vectorstore_path = Path(
            self.config.get("paths", {}).get("snapshot_vectorstore", str(self.index_root))
            if isinstance(self.config, dict) else str(self.index_root)
        )

    def _load_config(self, path: str) -> Dict:
        if toml is None:
            if self.verbose:
                logger.debug("[ingest] toml not installed; skipping config load")
            return {}
        try:
            if Path(path).exists():
                conf = toml.load(path)
                if self.verbose:
                    logger.debug(f"[ingest] Config loaded from {path}")
                return conf
            return {}
        except Exception as e:
            logger.warning(f"[ingest] Failed to load config {path}: {e}")
            return {}

    def find_latest_snapshot(self, base_dir: str = "scripts/stack-cat-files") -> Optional[str]:
        base_path = Path(base_dir)
        if not base_path.exists():
            if self.verbose:
                logger.debug(f"[ingest] Snapshot base dir not found: {base_path}")
            return None
        snapshots = [d for d in base_path.iterdir() if d.is_dir()]
        if not snapshots:
            return None
        latest = max(snapshots, key=lambda d: d.stat().st_mtime)
        logger.info(f"[ingest] Found latest snapshot: {latest}")
        return str(latest)

    def _gather_markdown_files(self, snapshot_path: Path) -> List[Path]:
        # Collect all markdown files inside the snapshot dir (master + per-file)
        files = [p for p in snapshot_path.rglob("*.md")]
        files = sorted(files, key=lambda p: p.name)
        if self.verbose:
            logger.info(f"[ingest] Discovered {len(files)} markdown files in snapshot {snapshot_path}")
        return files

    def _extract_metadata_from_content(self, content: str) -> Dict:
        metadata: Dict = {}
        lines = content.splitlines()
        for line in lines[:80]:
            l = line.strip()
            if l.startswith("**Generated:**"):
                metadata["generated"] = l.split("**Generated:**", 1)[1].strip()
            elif l.startswith("**Version:**"):
                metadata["version"] = l.split("**Version:**", 1)[1].strip()
            elif l.startswith("**Root Directory:**"):
                metadata["root_dir"] = l.split("**Root Directory:**", 1)[1].strip(" `")
            elif l.startswith("**Files Processed:**"):
                try:
                    val = l.split("**Files Processed:**", 1)[1].strip()
                    metadata["files_processed_reported"] = int(val) if val.isdigit() else val
                except Exception:
                    pass
        return metadata

    def categorize_snapshot_file(self, filename: str) -> str:
        n = filename.lower()
        if filename == "summary.md":
            return "summary"
        if n.startswith("stack-concat"):
            return "master"
        if "docker" in n or n.endswith(".yml") or n.endswith(".yaml"):
            return "docker"
        if "config" in n or n.endswith(".env") or n.endswith(".toml"):
            return "config"
        if n.endswith(".py"):
            return "python"
        if n.endswith(".sh"):
            return "shell"
        if "test" in n:
            return "tests"
        return "misc"

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        # preserve paragraphs, then slice with overlap
        if not text:
            return []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= chunk_size:
                current = (current + "\n\n" + para).strip() if current else para
            else:
                if current:
                    chunks.extend(self._slice_with_overlap(current, chunk_size, overlap))
                if len(para) > chunk_size:
                    chunks.extend(self._slice_with_overlap(para, chunk_size, overlap))
                    current = ""
                else:
                    current = para
        if current:
            chunks.extend(self._slice_with_overlap(current, chunk_size, overlap))
        return chunks

    def _slice_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        out: List[str] = []
        start = 0
        L = len(text)
        while start < L:
            end = min(start + chunk_size, L)
            out.append(text[start:end].strip())
            if end == L:
                break
            start = max(0, end - overlap)
        return out

    def _load_or_create_vectorstore(self):
        """
        Attempt to load vectorstore if present, otherwise return None.
        When ingesting, if self.embeddings is None we still create a vectorstore
        using FAISS.from_documents which may require embeddings (handled by FAISS/embedding).
        """
        if not _HAS_FAISS:
            logger.warning("[ingest] FAISS support not available (langchain-community vectorstores). Vectorstore operations will fail.")
            return None

        try:
            if self.vectorstore_path.exists():
                logger.info(f"[ingest] Loading existing vectorstore from {self.vectorstore_path}")
                vs = FAISS.load_local(str(self.vectorstore_path), self.embeddings, allow_dangerous_deserialization=True)
                return vs
        except Exception as e:
            logger.warning(f"[ingest] Failed to load vectorstore: {e}")
            return None
        return None

    def ingest_snapshot(self, snapshot_dir: str, chunk_size: int = 2000, overlap: int = 200) -> Dict[str, int]:
        snapshot_path = Path(snapshot_dir)
        if not snapshot_path.exists() or not snapshot_path.is_dir():
            raise FileNotFoundError(f"Snapshot dir not found: {snapshot_dir}")

        logger.info(f"[ingest] Ingesting snapshot: {snapshot_path}")
        md_files = self._gather_markdown_files(snapshot_path)

        stats = {"files_processed": 0, "documents_created": 0, "chunks_created": 0, "errors": 0}
        documents: List[Document] = []

        for md in md_files:
            try:
                if md.name in ("processing.log",):
                    continue
                txt = read_text_safe(md)
                if not txt.strip():
                    logger.debug(f"[ingest] Skipping empty file: {md}")
                    continue

                meta = self._extract_metadata_from_content(txt)
                category = self.categorize_snapshot_file(md.name)
                base_meta = {
                    "source": str(md),
                    "filename": md.name,
                    "snapshot": snapshot_path.name,
                    "category": category,
                    "size": md.stat().st_size,
                    "modified": datetime.fromtimestamp(md.stat().st_mtime).isoformat(),
                    **meta,
                }

                # either chunk or single doc
                if len(txt) > chunk_size:
                    chunks = self._chunk_text(txt, chunk_size, overlap)
                    for i, chunk in enumerate(chunks):
                        doc_meta = {**base_meta, "chunk_index": i, "total_chunks": len(chunks)}
                        documents.append(Document(page_content=chunk, metadata=doc_meta))
                        stats["chunks_created"] += 1
                else:
                    documents.append(Document(page_content=txt, metadata=base_meta))

                stats["files_processed"] += 1
                stats["documents_created"] = len(documents)
                logger.info(f"[ingest] Processed {md.name} -> docs={len(documents)}, chunks_total={stats['chunks_created']}")

            except Exception as e:
                logger.exception(f"[ingest] Error processing {md}: {e}")
                stats["errors"] += 1

        # Add documents to vectorstore
        if not documents:
            logger.info("[ingest] No documents to add; exiting ingestion.")
            return stats

        if not _HAS_FAISS:
            logger.error("[ingest] FAISS vectorstore not available; cannot index documents.")
            return stats

        # Try load existing vectorstore
        vs = self._load_or_create_vectorstore()
        try:
            if vs is not None:
                logger.info(f"[ingest] Adding {len(documents)} documents to existing vectorstore")
                vs.add_documents(documents)
            else:
                logger.info(f"[ingest] Creating new vectorstore from {len(documents)} documents")
                # FAISS.from_documents will call self.embeddings as needed
                vs = FAISS.from_documents(documents, self.embeddings)
        except Exception as e:
            logger.exception(f"[ingest] Failed to add/create vectorstore: {e}")
            raise

        # Save vectorstore atomically
        try:
            self.vectorstore_path.mkdir(parents=True, exist_ok=True)
            
            # Create temporary path for atomic save
            tmp_path = self.vectorstore_path.with_suffix('.tmp')
            
            # Save to temporary location
            vs.save_local(str(tmp_path))
            
            # Ensure data is synced to disk
            try:
                for root, _, files in os.walk(tmp_path):
                    for file in files:
                        file_path = Path(root) / file
                        with open(file_path, 'rb') as f:
                            os.fsync(f.fileno())
            except Exception as e:
                logger.warning(f"[ingest] fsync failed for {tmp_path}, continuing with replace: {e}")
            
            # Atomic rename
            os.replace(str(tmp_path), str(self.vectorstore_path))
            logger.info(f"[ingest] Vectorstore saved atomically to: {self.vectorstore_path}")
        except Exception as e:
            logger.exception(f"[ingest] Failed to save vectorstore: {e}")
            # Cleanup temporary path if it exists
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception as cleanup_e:
                    logger.warning(f"[ingest] Failed to cleanup temporary path: {cleanup_e}")

        # Cache metadata into Redis (if available)
        try:
            if self.redis_client:
                key = f"xnai:snapshot:{snapshot_path.name}"
                payload = {
                    "snapshot": snapshot_path.name,
                    "ingested_at": _now_iso(),
                    "files_processed": stats["files_processed"],
                    "documents_created": stats["documents_created"],
                    "chunks_created": stats["chunks_created"],
                    "vectorstore_path": str(self.vectorstore_path),
                }
                value = _orjson.dumps(payload) if _HAS_ORJSON else json.dumps(payload).encode("utf-8")
                # Use setex for TTL to keep a short-lived cache
                self.redis_client.setex(key, 86400, value)
                # maintain sorted history
                try:
                    self.redis_client.zadd("xnai:snapshot:history", {snapshot_path.name: time.time()})
                except Exception:
                    # redis-py older API may accept a different signature; try fallback
                    try:
                        self.redis_client.zadd("xnai:snapshot:history", time.time(), snapshot_path.name)
                    except Exception:
                        logger.debug("[ingest] Redis zadd fallback failed")
                logger.info(f"[ingest] Cached ingestion metadata in Redis: {key}")
        except Exception as e:
            logger.warning(f"[ingest] Redis cache failed: {e}")

        logger.info("[ingest] Ingestion finished")
        return stats

    def watch_for_snapshots(self, base_dir: str = "scripts/stack-cat-files", interval: int = 60):
        logger.info(f"[ingest] Watching for new snapshots in {base_dir} (interval={interval}s)")
        processed = set()
        # Load processed list from Redis set if available
        if self.redis_client:
            try:
                history = self.redis_client.zrange("xnai:snapshot:history", 0, -1)
                if history:
                    processed.update([h.decode("utf-8") if isinstance(h, bytes) else h for h in history])
                logger.info(f"[ingest] Loaded {len(processed)} previously processed snapshots from Redis")
            except Exception as e:
                logger.warning(f"[ingest] Failed to load snapshot history from Redis: {e}")

        base_path = Path(base_dir)
        while True:
            try:
                if not base_path.exists():
                    logger.debug(f"[ingest] Snapshot base path does not exist yet: {base_path}")
                    time.sleep(interval)
                    continue

                snapshots = [d for d in base_path.iterdir() if d.is_dir()]
                snapshots.sort(key=lambda d: d.stat().st_mtime)
                for snap in snapshots:
                    name = snap.name
                    if name in processed:
                        continue
                    logger.info(f"[ingest] New snapshot discovered: {snap}")
                    # small sleep to let writing finish
                    time.sleep(3)
                    try:
                        stats = self.ingest_snapshot(str(snap))
                        logger.info(f"[ingest] Snapshot ingested: {snap} stats={stats}")
                        processed.add(name)
                    except Exception as e:
                        logger.exception(f"[ingest] Failed to ingest snapshot {snap}: {e}")
                time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("[ingest] Watch loop stopped by user")
                break
            except Exception as e:
                logger.exception(f"[ingest] Error in watch loop: {e}")
                time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ingest stack-cat snapshots into XNAi RAG")
    parser.add_argument("--snapshot", dest="snapshot_dir", help="Path to snapshot directory (scripts/stack-cat-files/<ts>)")
    parser.add_argument("--auto-latest", action="store_true", help="Auto-select and ingest the latest snapshot")
    parser.add_argument("--watch", action="store_true", help="Watch for new snapshots and auto-ingest")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval seconds")
    parser.add_argument("--index-root", type=str, default="knowledge/coder/stack_snapshots", help="Index root for storing snapshot indexes")
    parser.add_argument("--model-path", type=str, help="Override embedding model path")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=200, help="Chunk overlap in characters")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args(argv or sys.argv[1:])

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # If model-path provided, set env var for the Llama init helper
    if args.model_path:
        os.environ["EMBEDDING_MODEL_PATH"] = args.model_path

    ing = SnapshotIngestor(
        config_path="config.toml",
        index_root=args.index_root,
        n_threads=int(os.getenv("LLAMA_CPP_N_THREADS", "6")),
        verbose=args.verbose,
    )

    if args.watch:
        ing.watch_for_snapshots(interval=args.interval)
        return

    if args.auto_latest:
        snap = ing.find_latest_snapshot()
        if not snap:
            logger.error("No snapshots found to ingest (auto-latest).")
            sys.exit(1)
        stats = ing.ingest_snapshot(snap, chunk_size=args.chunk_size, overlap=args.overlap)
        print(json.dumps(stats, indent=2))
        return

    if args.snapshot_dir:
        stats = ing.ingest_snapshot(args.snapshot_dir, chunk_size=args.chunk_size, overlap=args.overlap)
        print(json.dumps(stats, indent=2))
        return

    parser.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()

