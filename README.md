# Xoe-NovAi Phase 1 v0.1.3-beta

Xoe-NovAi is a CPU-optimized, zero-telemetry local AI RAG (Retrieval-Augmented Generation) stack designed for AMD Ryzen 7 5700U systems. It integrates FAISS for vector search, LlamaCpp for inference, and LlamaCppEmbeddings for embeddings. CrawlModule v0.7.3 is used for automated library curation from sources like Project Gutenberg, arXiv, PubMed, and YouTube. Phase 1 focuses on core RAG functionality with <6GB memory footprint, 15-25 tok/s generation, and <90s startup.

## Features

- **Zero-Telemetry Privacy**: 13 explicit disables (CHAINLIT_NO_TELEMETRY=true, CRAWL4AI_NO_TELEMETRY=true, etc.).
- **Ryzen-Optimized**: Threading (N_THREADS=6), F16_KV=true, OPENBLAS_CORETYPE=ZEN for 15-25 tok/s.
- **RAG Pipeline**: FAISS vectorstore (top_k=5, threshold=0.7), chunking (1000 chars, 200 overlap), SSE streaming.
- **Curation Engine**: CrawlModule for 50-200 items/h; Sources: Gutenberg (classics), arXiv (physics), PubMed (psychology), YouTube (lectures); Sanitization, rate limiting (30/min), Redis caching (TTL=86400s).
- **UI/API**: Chainlit async interface (/curate, /query, /stats); FastAPI backend with metrics (8002).
- **Security**: Non-root containers (UID 1001), capability dropping, tmpfs for ephemeral data.
- **Monitoring**: Prometheus metrics, JSON logging (max_size=10MB), healthchecks (90s start_period).
- **Testing**: Pytest with >90% coverage, CI/CD workflow.
- **Configuration**: 197 .env vars, 23 config.toml sections; Validation scripts.

## File Tree
```
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
```

## Quick Start

### Prerequisites

- Python 3.12.7, Docker 27.3+, Compose v2.29.2+, Git.
- AMD Ryzen 7 5700U (or equivalent; adjust N_THREADS for other CPUs).
- 8GB+ RAM, 50GB+ storage (models ~3GB).

### Setup

1. Clone repo: `git clone https://github.com/Xoe-NovAi/Xoe-NovAi.git && cd Xoe-NovAi`.
2. Copy .env: `cp .env.example .env` (197 vars; customize REDIS_PASSWORD, APP_UID=$(id -u)).
3. Copy config.toml: `cp config.toml.example config.toml` (23 sections; verify telemetry_enabled=false).
4. Download models: `make download-models` (gemma-3-4b-it-UD-Q5_K_XL.gguf ~2.8GB, all-MiniLM-L12-v2.Q8_0.gguf ~45MB).
5. Chown dirs: `sudo chown -R 1001:1001 ./app ./backups ./data/faiss_index ./logs ./library ./knowledge; sudo chown -R 999:999 ./data/redis`.
6. Validate: `python3 scripts/validate_config.py` (exits 0, 197 vars, 13 telemetry disables).

### Build & Run

1. `sudo docker compose up -d --build` (<90s startup).
2. Check status: `sudo docker compose ps` (all healthy).
3. Logs: `sudo docker compose logs -f` (uvicorn 8000, chainlit 8001, crawler tail -f, redis Ready).
4. Health: `curl http://localhost:8000/health` ({"status": "healthy"}).

## Usage

- **UI**: Open http://localhost:8001; Commands: /help, /query "test", /curate gutenberg classics "Plato", /stats (session info), /reset, /rag on/off.
- **API**: `curl -X POST http://localhost:8000/query -d '{"query":"test"}'` (SSE stream).
- **Curation**: `sudo docker exec xnai_crawler python3 crawl.py --curate gutenberg -c classics -q "Plato" --max-items=50` (50 items to /library, embedded FAISS).
- **Ingest**: `sudo docker exec xnai_rag_api python3 ingest_library.py --library-path /library` (load to FAISS).
- **Metrics**: http://localhost:8002/metrics (Prometheus).

## Architecture

- **Services**: Redis (cache, 6379), RAG API (FastAPI, 8000/8002), Chainlit UI (8001), Crawler (daemon).
- **Data Flow**: Crawler → library/ → Ingest → FAISS (/data/faiss_index) → RAG query → SSE stream to UI.
- **Security**: Non-root (1001), no-new-privileges, cap_drop ALL.
- **Config**: .env (197 vars), config.toml (23 sections); Validate with make validate.

## Configuration

- **.env**: 197 vars (e.g., REDIS_PASSWORD, LLAMA_CPP_N_THREADS=6). Copy .env.example; Customize passwords.
- **config.toml**: 23 sections (metadata, models, performance). Edit memory_limit_gb=6.0, telemetry_enabled=false.
- **Validate**: `python3 scripts/validate_config.py` (exits 0).

## Testing & Validation

- **Health Check**: `make health` (7/7 OK: llm, embeddings, memory, redis, vectorstore, ryzen, crawler).
- **Benchmark**: `make benchmark` (15-25 tok/s, <6GB memory).
- **Tests**: `pytest --cov` (>90% coverage).
- **Curation**: `make curate` (50-200 items/h, e.g., Gutenberg classics).
- **Debug**: `make debug` (docker exec bash for inspection).

## Troubleshooting

- **Exec Permission Denied**: Set security_opt: no-new-privileges in compose; Chown host dirs to 1001:1001.
- **Config.toml Not Found**: Add mount - ./config.toml:/app/XNAi_rag_app/config.toml.
- **Low Tok/s**: Check N_THREADS=6, f16_kv=true; Run make benchmark.
- **Logs**: `sudo docker compose logs -f rag` (uvicorn 8000).

## Future Development Plans

We are actively working on a meta-guide generator system that allows users to specify parameters for their desired AI stack (e.g., hardware targets, preferred models, features like multi-agent support or cloud integration). This tool will produce a customized, detailed guide similar to the one used for Xoe-NovAi Phase 1, which can be fed directly to AI coding assistants (e.g., Claude, Grok, GPT) for automated full-stack implementation. This will democratize building production-grade AI systems, enabling rapid prototyping and deployment tailored to specific needs.

### Phase 1.5 (Next 3 Months)
- Config templating for dev/staging/prod environments.
- Blue-green deployments for zero-downtime updates.
- Observability stack integration (Grafana, Loki, Tempo for dashboards, logs, and tracing).
- Advanced RAG strategies (HyDE, MultiQuery retrievers, self-query with metadata filtering).
- Integration of specialized expert agents (e.g., domain-specific LLMs for coding, research, or creative tasks) to enhance multi-agent coordination.

### Phase 2 (6-12 Months)
- Multi-agent coordination implemented via Redis Streams (e.g., coder, editor, manager, learner agents).
- Multi-model support beyond Gemma-3 (e.g., dynamic switching to Mistral or Llama 3).
- Kubernetes deployment with StatefulSets for Redis and Horizontal Pod Autoscaler (HPA) rules.
- Distributed architecture for multi-node scaling (HAProxy load balancing, shared Redis state).
- Load testing framework (Locust or k6 for simulating 100+ concurrent users).
- Chaos engineering integration (Chaos Mesh for K8s to test resilience).
- Qdrant integration for enhanced vector search capabilities, replacing FAISS with distributed indexing and real-time updates.
- Vulkan iGPU offloading for experimental GPU acceleration on compatible hardware.

### Phase 3 (12-18 Months)
- Auto-scaling with ML-based forecasting (predictive scaling based on usage trends).
- Performance regression testing in CI/CD pipelines (fail builds on >10% degradation).
- Canary deployments with traffic splitting for safe rollouts.

### Phase 4 (18-24 Months)
- Full VR worlds integration where users can interact with agents and models in avatar form, leveraging frameworks like GenLARP for immersive role-playing with AI agents.
- Cross-stack connectivity enabling users to connect to other users' XNAi stacks, interact with their agents, and collaborate with other stack users in virtual environments.
- Agent-to-agent interactions across different XNAi stacks, allowing agents to learn from each other and share knowledge (e.g., via federated learning protocols or shared knowledge graphs).
- Scalable VR ecosystems with multi-user support for collaborative AI-driven projects, such as joint research, creative writing, or code development in immersive settings.

## Contributing

Fork repo, create feature branch (`git checkout -b feature/fix-var-count`).
Commit (`git commit -m "fix: complete .env to 197 vars"`).
Push branch, open PR to main.
Ensure PR passes CI (pytest >90%, make validate).

## CI/CD

GitHub Actions: .github/workflows/ci.yml runs pytest --cov, python3 scripts/validate_config.py, and make benchmark.
Ensure green CI before PR merge.

## License
MIT License. See LICENSE for details.