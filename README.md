Xoe-NovAi Phase 1 v0.1.2

Xoe-NovAi is a CPU-optimized, zero-telemetry local AI RAG (Retrieval-Augmented Generation) stack designed for AMD Ryzen 7 5700U systems. It integrates FAISS for vector search, LlamaCpp for inference, and CrawlModule v0.1.7 for automated library curation from sources like Project Gutenberg, arXiv, PubMed, and YouTube. Phase 1 focuses on core RAG functionality with <6GB memory footprint, 15-25 tok/s generation, and <90s startup.
Features

Zero-Telemetry Privacy: 8 explicit disables (CHAINLIT_NO_TELEMETRY=true, CRAWL4AI_NO_TELEMETRY=true, etc.).
Ryzen-Optimized: Threading (N_THREADS=6), f16_kv=true, OpenBLAS ZEN coretype for 15-25 tok/s.
RAG Pipeline: FAISS vectorstore (top_k=5, threshold=0.7), chunking (1000 chars, 200 overlap), SSE streaming.
Curation Engine: CrawlModule for 50-200 items/h; Sources: Gutenberg (classics), arXiv (physics), PubMed (psychology), YouTube (lectures); Sanitization, rate limiting (30/min), Redis caching (TTL=86400s).
UI/API: Chainlit async interface (/curate, /query, /stats); FastAPI backend with metrics (8002).
Security: Non-root containers (UID 1001), capability dropping, tmpfs for ephemeral data.
Monitoring: Prometheus metrics, JSON logging (max_size=10MB), healthchecks (90s start_period).

File Tree
.
├── app/
│   └── XNAi_rag_app/
│       ├── chainlit_app.py
│       ├── config_loader.py
│       ├── crawl.py
│       ├── dependencies.py
│       ├── healthcheck.py
│       ├── logging_config.py
│       ├── main.py
│       ├── metrics.py
│       └── verify_imports.py
├── config.toml
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.chainlit
├── Dockerfile.crawl
├── Dockerfile.redis
├── requirements-api.txt
├── requirements-chainlit.txt
├── requirements-crawl.txt
├── scripts/
│   ├── create_structure.sh
│   ├── ingest_library.py
│   ├── query_test.py
│   ├── setup-new-stack.sh
│   └── validate_config.py
├── tests/
│   ├── conftest.py
│   ├── test_crawl.py
│   ├── test_healthcheck.py
│   ├── test_integration.py
│   └── test_truncation.py
├── .gitignore
├── LICENSE
└── README.md

Quick Start
Prerequisites

Python 3.12, Docker Compose v2, Git.
AMD Ryzen 7 5700U (or equivalent; adjust N_THREADS for other CPUs).
8GB+ RAM, 50GB+ storage (models ~3GB).

Setup

Clone repo: git clone https://github.com/Xoe-NovAi/Xoe-NovAi.git && cd Xoe-NovAi.
Copy .env: cp .env.example .env (197 vars; customize REDIS_PASSWORD, APP_UID=$(id -u)).
Copy config.toml: cp config.toml.example config.toml (23 sections; verify telemetry_enabled=false).
Download models: make download-models (gemma-3-4b-it-UD-Q5_K_XL.gguf ~2.8GB, all-MiniLM-L12-v2.Q8_0.gguf ~45MB).
Chown dirs: sudo chown -R 1001:1001 ./app ./backups ./data/faiss_index ./logs ./library ./knowledge; sudo chown -R 999:999 ./data/redis.
Validate: python3 scripts/validate_config.py (exits 0, 197 vars, 8 telemetry disables).

Build & Run

sudo docker compose up -d --build (<90s startup).
Check status: sudo docker compose ps (all healthy).
Logs: sudo docker compose logs -f (uvicorn 8000, chainlit 8001, crawler tail -f, redis Ready).
Health: curl http://localhost:8000/health ({"status": "healthy"}).

Usage

UI: Open http://localhost:8001; Commands: /help, /query "test", /curate gutenberg classics "Plato", /stats (session info), /reset, /rag on/off.
API: curl -X POST http://localhost:8000/query -d '{"query":"test"}' (SSE stream).
Curation: sudo docker exec xnai_crawler python3 crawl.py --curate gutenberg -c classics -q "Plato" --max-items=50 (50 items to /library, embedded FAISS).
Ingest: sudo docker exec xnai_rag_api python3 ingest_library.py --library-path /library (load to FAISS).
Metrics: http://localhost:8002/metrics (Prometheus).

Architecture

Services: Redis (cache, 6379), RAG API (FastAPI, 8000/8002), Chainlit UI (8001), Crawler (daemon).
Data Flow: Crawler → library/ → Ingest → FAISS (/data/faiss_index) → RAG query → SSE stream to UI.
Security: Non-root (1001), no-new-privileges=false for rag (exec fix), cap_drop ALL.
Config: .env (197 vars), config.toml (23 sections); Validate with make validate.

Configuration

.env: 197 vars (e.g., REDIS_PASSWORD, LLAMA_CPP_N_THREADS=6). Copy .env.example; Customize passwords.
config.toml: 23 sections (metadata, models, performance). Edit memory_limit_gb=6.0, telemetry_enabled=false.
Validate: python3 scripts/validate_config.py (exits 0).

Testing & Validation

Health Check: make health (7/7 OK: llm, embeddings, memory, redis, vectorstore, ryzen, crawler).
Benchmark: make benchmark (15-25 tok/s, <6GB memory).
Tests: pytest --cov (>90% coverage).
Curation: make curate (50-200 items/h, e.g., Gutenberg classics).
Debug: make debug (docker exec bash for inspection).

Troubleshooting

Exec Permission Denied: Set security_opt: no-new-privileges=false in compose; Chown host dirs to 1001:1001.
Config.toml Not Found: Add mount - ./config.toml:/app/XNAi_rag_app/config.toml.
Var Count <197: Copy full .env (197 vars).
Low Tok/s: Check N_THREADS=6, f16_kv=true; Run make benchmark.
Logs: sudo docker compose logs -f rag (uvicorn 8000).

Contributing

Fork repo, create feature branch (git checkout -b feature/fix-var-count).
Commit (git commit -m "fix: complete .env to 197 vars").
Push branch, open PR to main.
Ensure PR passes CI (pytest >90%, make validate).

CI/CD

GitHub Actions: .github/workflows/ci.yml runs pytest --cov, python3 scripts/validate_config.py, and make benchmark.
Ensure green CI before PR merge.

License
MIT License. See LICENSE for details.