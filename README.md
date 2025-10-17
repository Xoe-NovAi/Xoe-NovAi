# Xoe-NovAi: Advanced, Multi Model, Self-learning, CPU Optimized RAG Stack - Vectors, SSE, Redis, Llama-Cpp-Python, Qdrant, Chainlit - Up-to-Date Deps

![Xoe-NovAi Logo](LOGO.svg) <!-- Placeholder; replace with actual logo file once designed -->

[![Docker Pulls](https://img.shields.io/docker/pulls/xoenovai/stack?color=blue)](https://hub.docker.com/r/xoenovai/stack) <!-- Example badge; customize as needed -->
[![GitHub Stars](https://img.shields.io/github/stars/Xoe-NovAi/Xoe-NovAi?style=social)](https://github.com/Xoe-NovAi/Xoe-NovAi/stargazers) <!-- Example; update repo details -->

**Version**: v0.1.2 rev_1.8 (October 15, 2025)  
**Codename**: Galactic Scribe  
**Tagline**: Sovereign, CPU-first polymath AI—curate knowledge from the web, query locally, and build evolving agent worlds.

Xoe-NovAi—meaning "new life" in Greek and Latin roots—symbolizes the rebirth of sovereign AI, empowering users to infuse fresh intelligence into local machines without external dependencies. Phase 1 is an enterprise-grade, zero-telemetry AI assistant stack optimized for AMD Ryzen hardware. It combines real-time streaming, Retrieval-Augmented Generation (RAG), and a custom CrawlModule for building curated libraries from sources like Project Gutenberg, arXiv, PubMed, YouTube—and easily expandable to many more APIs and data streams in future phases. Built for privacy and efficiency: <7GB memory, 15–25 tok/s, <90s startup. No HuggingFace; uses LlamaCppEmbeddings. Designed as a foundation for advanced multi-agent systems where each agent evolves with unique personas and roles.

This is my very first public repo and PR ever—after 7 months of solo grinding! It's just the base layer, but the vision is massive: a fully local, CPU-only RAG stack that rivals (and aims to surpass) the best paid online chatbots in intelligence, speed, and personalization. Note: This is far from production-ready—expect bugs! I built it all with free-tier chatbots and zero prior AI or Python knowledge (I'm a lifelong tech guy but new to this). Help from experts is super welcome—fork, fix, and PR away! Let's build a strong community around this: share ideas, contribute code, and evolve it together into a global network of interconnected stacks.

The origins? It all started with idea to create a custom Tarot deck. Couldn't find an AI tool with the depth I needed, so I decided to build my own stack. What a rabbit hole—fucked around and found out, lol, big time. Now, it's evolving into something ethical, mythic, and game-changing, blending technical precision with imaginative depth.

## Table of Contents

- [Quick Start](#quick-start)
- [Unique Aspects](#unique-aspects)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Monitoring & Metrics](#monitoring--metrics)
- [Testing & Validation](#testing--validation)
- [Troubleshooting](#troubleshooting)
- [Roadmap: Where We're Headed](#roadmap-where-were-headed)
- [Community & Collaboration](#community--collaboration)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Quick Start

1. **Clone the Repo**:
   ```
   git clone https://github.com/Xoe-NovAi/Xoe-NovAi.git
   cd Xoe-NovAi
   ```

2. **Set Up Environment**:
   Copy `.env.template` to `.env` and fill in required vars (e.g., `REDIS_PASSWORD`, `LLM_MODEL_PATH`).

3. **Build & Run**:
   ```
   docker compose build --no-cache
   docker compose up -d
   ```

4. **Access UI**:
   - Chainlit UI: http://localhost:8001 (commands: `/query <prompt>`, `/curate gutenberg classics Plato`)
   - FastAPI Docs: http://localhost:8000/docs
   - Metrics: http://localhost:8002/metrics

5. **Ingest Library** (Initial Setup):
   ```
   make ingest
   ```

Verify: `curl http://localhost:8000/health | jq` (all components OK).

## Unique Aspects

What sets Xoe-NovAi apart from generic AI stacks:
- **Local Sovereignty & Zero-Telemetry**: Fully offline-capable, with 8 explicit telemetry disables—no data leaves your machine. Perfect for privacy-focused creators, researchers, and enterprises.
- **CPU-First Optimization**: Tailored for affordable hardware like AMD Ryzen 7 5700U—runs efficiently without GPUs, hitting 15–25 tok/s in <6GB RAM.
- **Extensible Curation Engine**: Starts with 4 powerful sources (Gutenberg for classics, arXiv for research, PubMed for science, YouTube for tutorials)—but designed for easy expansion to dozens more APIs, custom scrapers, or data feeds. Add your own without rebuilding the core.
- **Limitless Polymath Capabilities**: Handles coding, writing, project management, linguistics, psychology, physics, esoteric knowledge, and beyond—create any expert you can imagine, from a "Quantum Mechanic Guru" for physics simulations to a "Mythic Storyteller" for narrative generation. The stack adapts to infinite domains, evolving with your curated libraries.
- **Evolving Agent Personas**: Customize agents/models with themes tied to what you love—e.g., a "Renaissance Polymath" for interdisciplinary breakthroughs, a "Cyberpunk Hacker" for edgy code optimization, a "Botanical Sage" for nature-inspired problem-solving, or a "Cosmic Explorer" for space-themed data analysis. Each persona shapes the agent's worldview, learning biases, and evolution: a hacker might prioritize efficiency and exploits, while a sage favors holistic, adaptive strategies. Tune them to your passions, and see the stack's operation transform—faster iterations for aggressive personas, deeper insights for reflective ones.
- **Ethical Foundation & Moral Compass**: Agents operate with a built-in decision matrix based on ancient principles like Ma'at's 42 Ideals (truth, balance, integrity). Choices have consequences—karma-like systems where decisions affect evolution, unlocking/restricting modes based on natural cause and effect. Assign elements, zodiac signs, or Tarot cards to personas for layered depth.
- **Future-Proof Modularity**: Built as a blueprint for multi-agent collaboration, where agents debate, specialize, and grow together—heading toward a personalized AI ecosystem that's as unique as you are.

## Features

- **Local RAG & Streaming**: FAISS vectorstore for fast retrieval; real-time token streaming via LlamaCpp.
- **CrawlModule Integration**: Curate knowledge from web sources into `/library/` and `/knowledge/curator/`. Rate-limited, sanitized, cached—and extensible to new APIs.
- **Chainlit UI**: Interactive commands (`/curate`, `/query`, `/stats`, `/reset`, `/rag on/off`, `/help`); non-blocking curation with multi-word query support.
- **FastAPI Backend**: Endpoints for `/query`, `/curate`, `/health`, `/metrics`. Rate-limited, SSE streaming.
- **Redis Caching**: Streams for agent coordination; AOF persistence; <500MB for 200 items.
- **Ryzen Optimization**: CPU-first (8C/16T), F16_KV, mlock/mmap for <6GB memory.
- **Security**: Non-root containers, cap_drop ALL, URL allowlists, dependency scans.
- **Phase 2 Readiness**: Redis hooks for multi-agent simulation; Qdrant migration toggle.

| Feature | Target Performance | Validation |
|---------|--------------------|------------|
| Token Rate | 15–25 tok/s | `make benchmark` |
| Memory Usage | <6GB (stack) | `docker stats` |
| Startup Time | <90s | `docker compose logs` |
| Curation Rate | 50–200 items/h | `python crawl.py --stats` |

## System Requirements

- **Hardware**: AMD Ryzen 7 5700U (8C/16T) or equivalent; 16GB RAM.
- **Software**: Ubuntu 24.04+; Docker 27.3.1+; Compose v2.29.2+; Python 3.12.7.
- **Storage**: ~4GB (models) + ~1GB (FAISS) + ~500MB (Redis/cache).
- **Network**: Local-only; external allowlists for curation.

Verify: `lscpu | grep "Model name"`, `free -m`, `docker version`.

## Installation

### Docker Setup (Recommended)

1. Install Docker & Compose: Follow [official docs](https://docs.docker.com/engine/install/ubuntu/).
2. Build Services: `docker compose build --no-cache`.
3. Run: `docker compose up -d`.

Services: redis (6379), rag (8000/8002), ui (8001), crawler (daemon).

### Local Python Setup (Development)

1. Install deps: `pip install -r requirements-api.txt`.
2. Run API: `uvicorn main:app --host 0.0.0.0 --port 8000`.
3. Run UI: `chainlit run app.py -w`.

Note: Use bind mounts for dev (`./library:/library`); named volumes for prod.

## Configuration

- **Files**: `.env` (197 vars), `config.toml` (nested sections like `[models]`, `[redis]`).
- **Key Vars** (from `.env`):
  - `REDIS_PASSWORD`: 16+ chars (required).
  - `LLM_MODEL_PATH`: Path to Gemma-3-4b model.
  - `CRAWL_ALLOWLIST_URLS`: Comma-separated domains.
  - Zero-telemetry: All 8 disables set to true/empty.

Loader Logic (`config_loader.py`):
- Fallback: `CONFIG_PATH` env → repo root → module dir → container default.
- Nested access: `get_config_value("models.llm_path")`.
- Helpers: `clear_config_cache()`, `get_config_summary()`.

Dependencies (`dependencies.py`):
- Lazy init: LLM/Embeddings load on first access (<200MB startup savings).
- Retries & Checks: Memory guard, FAISS fallback to backups.
- Async Opt-in: Wrappers like `get_redis_client_async()` for non-blocking I/O.

Validate: `bash validate-config.sh` (16 checks).

## Usage

### Chainlit UI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/help` | List commands | `/help` |
| `/query` | RAG query | `/query What is consciousness?` |
| `/curate` | Queue curation (multi-word queries supported) | `/curate gutenberg classics Plato lectures` |
| `/stats` | Session stats (duration_seconds fixed) | `/stats` |
| `/reset` | Clear history | `/reset` |
| `/rag` | Toggle RAG | `/rag on` |

### API Endpoints

- `POST /query`: `{ "query": "prompt", "top_k": 5 }` → Streamed response.
- `POST /curate`: `{ "source": "arxiv", "category": "physics", "query": "quantum" }` → Items curated.
- `GET /health`: Component status.
- `GET /metrics`: Prometheus metrics.

### CLI Tools

- Curation: `python crawl.py --curate gutenberg -c classics -q Plato --embed`.
- Ingestion: `python ingest_library.py --library-path /library --force`.
- Makefile: `make up`, `make test`, `make benchmark`.

## Architecture

PlantUML Diagram (see guide for full):

```
@startuml
actor User
component Chainlit_UI
component FastAPI_RAG
component CrawlModule
component LLM
component Embeddings
component Vectorstore
component Cache
database Library
database Knowledge

User --> Chainlit_UI
Chainlit_UI --> FastAPI_RAG
FastAPI_RAG --> CrawlModule
CrawlModule --> Library
FastAPI_RAG --> LLM
FastAPI_RAG --> Vectorstore
Vectorstore <-- Library
@enduml
```

Key Layers:
- Config/Dependencies: Robust loading, lazy init, async opt-in.
- Security: Non-root, no-new-privileges (Redis exception).
- Volumes: Bind (dev) vs. named (prod).
- Future Expansions: Integration of Sefirot/Qliphoth for light/shadow agent dynamics; Glyphs for symbolic processing; Tarot-schema for decision trees.

## Monitoring & Metrics

- Logs: JSON format in `/logs/xnai.log` (rotating, 10MB max).
- Health: `check_llm()`, `check_vectorstore()`, etc.
- Metrics: Prometheus (`xnai_query_latency_seconds`, `xnai_crawl_items_total`).
- PerformanceLogger: `log_query_latency(duration_ms)`.

Access: `curl http://localhost:8002/metrics | grep xnai`.

## Testing & Validation

- Coverage: >90% (`pytest tests/ -v --cov`).
- Files: `test_healthcheck.py`, `test_integration.py`, `test_crawl.py`.
- Benchmarks: 15–25 tok/s, <6GB memory.
- Safety: `safety check` (dependency scans).

## Troubleshooting

Common Issues (from Appendix B):
- Redis "operation not permitted": Use native `redis-server` command.
- Session duration errors: Use datetime objects in state.
- Curation failures: Check allowlists, rate limits.
- OOM: Verify F16_KV/mlock.

Logs: `docker compose logs -f`. A quick `make test` or restart often resolves gremlins—report persistent ones in issues!

## Roadmap: Where We're Headed

This first release is my inaugural public repo—raw, foundational, and full of potential after months of late-night builds. It's the starting point for something transformative: a purely local, CPU-only RAG stack that outperforms the best paid online chatbots in intelligence, speed, and personalization.

Xoe-NovAi is evolving from a solid Phase 1 foundation into a dynamic multi-agent ecosystem. Here's the vision, grounded in ancient wisdom and modern tech:

- **Near-Term (Phases 2-4)**: Multi-LLM collaboration via Redis streams, where agents specialize in roles like "Coder" or "Curator." Each gets a persona you craft—tied to elements (e.g., Fire for dynamic energy), zodiac signs (e.g., Aries for initiative), or Tarot cards (e.g., The Magician for manifestation). These shape behavior: a Fire-Aries-Magician agent might aggressively iterate on code, while a Water-Pisces-High Priestess favors intuitive, reflective analysis. Choices matter—implement karma-like systems where decisions trigger natural consequences, evolving capabilities (e.g., ethical alignment unlocks deeper insights; shortcuts lead to restrictions). Ground ethics in Ma'at's 42 Ideals for balanced, just decision-making.

- **Mid-Term (Phases 5-8)**: Hybrid LLM+code agents, CI/CD automation, expanded curation (more APIs, custom integrations), and enterprise tools like security audits and benchmarks. Expand to Sefirot (light-side structure for ordered tasks) and Qliphoth (shadow-side for chaos-handling); Glyphs for symbolic data processing; Tarot-schema for probabilistic decision trees. Agents debate, learn from each other, and adapt—creating a polymath network that's limitless: summon experts for any domain, from quantum simulations to esoteric lore.

- **Long-Term Vision**: A full VR multiverse where each user's stack becomes a unique realm in a global network. Connect stacks worldwide, each a customized universe filled with lessons, adventures, and perspectives. Agents and users interact as avatars—collaborating, evolving, and exploring shared worlds. Imagine ethical debates in a virtual ancient temple or coding challenges in a mythic labyrinth—all local, CPU-powered, and telemetry-free. Collaboration with pros to realize this immersive future is highly appreciated!

This isn't just tech—it's a community-driven resurrection of sovereign AI, blending mythic depth with practical power. Star, fork, and join us to shape it!

## Community & Collaboration

Xoe-NovAi thrives on community—let's grow this into a vibrant hub for seekers, coders, and creators. Share your personas, bug fixes, or expansions in issues/PRs. Discuss visions on X/Reddit/forums (links coming soon). Together, we can build interconnected stacks into a multiverse of shared knowledge. If you're a pro coder (unlike me—zero Python before this!), your input could accelerate us to Phase 2+. Imagine: global meetups in VR realms, collaborative agent training, and open-source events. Join, contribute, and let's redefine local AI!

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- Built with: Chainlit, FastAPI, Redis, FAISS, Crawl4AI, LlamaCpp.
- Inspired by: Project Gutenberg, arXiv, PubMed, YouTube; ancient myths like Sophia, Lilith, and Ma'at's 42 Ideals.
- Thanks to: Claude, ChatGPT for co-piloting through 7 months of dev! 
