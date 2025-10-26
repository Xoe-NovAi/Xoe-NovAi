xoe-novai/
├── app/XNAi_rag_app/          # Core Python application
│   ├── chainlit_app.py        # Async UI with subprocess tracking
│   ├── config_loader.py       # Centralized config management
│   ├── crawl.py               # CrawlModule with security fixes
│   ├── dependencies.py        # Dependency init with retry logic
│   ├── healthcheck.py         # 7-target health checks
│   ├── logging_config.py      # JSON structured logging
│   ├── main.py                # FastAPI with SSE streaming
│   ├── metrics.py             # Prometheus metrics
│   ├── verify_imports.py      # Dependency validation
│   ├── logs/                  # LOG FILES (created in Dockerfile, 777)
│   ├── faiss_index/           # Primary FAISS vectorstore
│   └── __init__.py            # Package marker
├── data/                      # Runtime data (gitignored)
│   ├── redis/                 # Redis persistence
│   ├── faiss_index/           # Development FAISS copy
│   ├── cache/                 # Crawler cache
│   └── prometheus-multiproc/  # Metrics storage
├── library/                   # Curated documents (gitignored)
│   ├── psychology/
│   ├── physics/
│   ├── classical-works/
│   ├── esoteric/
│   └── technical-manuals/
├── knowledge/                 # Phase 2 agent knowledge (gitignored)
│   ├── curator/               # CrawlModule metadata (index.toml)
│   ├── coder/                 # Coding expert (Phase 2)
│   ├── editor/                # Writing assistant (Phase 2)
│   ├── manager/               # Project manager (Phase 2)
│   └── learner/               # Self-learning (Phase 2)
├── models/                    # LLM models (gitignored)
│   └── gemma-3-4b-it-UD-Q5_K_XL.gguf (2.8GB)
├── embeddings/                # Embedding models (gitignored)
│   └── all-MiniLM-L12-v2.Q8_0.gguf (45MB)
├── backups/                   # FAISS backups (gitignored)
├── scripts/                   # Utility scripts
│   ├── ingest_library.py      # Ingestion with checkpointing
│   ├── query_test.py          # Performance benchmarking
│   └── validate_config.py     # Config validation (197 vars)
├── tests/                     # pytest test suite
│   ├── conftest.py            # 15+ fixtures (COMPLETE in v0.1.3)
│   ├── test_healthcheck.py    # 12 health check tests
│   ├── test_integration.py    # 8 integration tests
│   ├── test_crawl.py          # 15 CrawlModule tests
│   └── test_truncation.py     # 10 context truncation tests
├── config.toml                # Application config (23 sections)
├── docker-compose.yml         # Service orchestration (Compose v2)
├── Dockerfile.api             # RAG API multi-stage build
├── Dockerfile.chainlit        # UI multi-stage build
├── Dockerfile.crawl           # Crawler multi-stage build
├── .env                       # Environment variables (197 vars)
├── .env.example               # Template copy
├── .gitignore                 # Exclude runtime data
├── .dockerignore              # Exclude build artifacts
├── Makefile                   # Convenience targets (15 commands)
├── README.md                  # Quick start guide
├── requirements-api.txt       # FastAPI dependencies
├── requirements-chainlit.txt  # Chainlit dependencies
└── requirements-crawl.txt     # CrawlModule dependencies