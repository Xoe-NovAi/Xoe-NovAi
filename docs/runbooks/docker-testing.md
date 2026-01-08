# Xoe-NovAi v0.1.4-stable Docker Testing & Validation Plan
## Library API Integration + FAISS Production Release

---

## TESTING OVERVIEW

This document outlines the comprehensive testing approach for the Xoe-NovAi FAISS v0.1.4-stable release with full library API integration, domain categorization, and Dewey Decimal system support.

**Status**: Production Ready for Testing
**Test Date**: January 3, 2026
**Release Target**: v0.1.4-stable (FAISS)

---

## 1. LOCAL UNIT TESTS (Complete ✓)

### 1.1 Library API Integration Tests
- **Status**: ✓ PASSED
- **Test Location**: `test_library_integration()` in `library_api_integrations.py`
- **Coverage**:
  - Domain classification (Code, Science, Books, etc.)
  - Dewey Decimal mapping
  - Library enrichment workflow
  - Available domain categories
  
- **Results**:
  - ✓ 4/4 domain classifications working (see test output)
  - ✓ Dewey Decimal mappings correct
  - ✓ 12 domain categories available
  - ✓ Category keyword matching functional

### 1.2 Curation Module Tests
- **Status**: ✓ PASSED
- **Test Location**: `test_extraction()` in `crawler_curation.py`
- **Coverage**:
  - Content extraction
  - Metadata calculation
  - Domain classification
  - Citation detection
  - Quality factor computation

### 1.3 Configuration Tests
- **Status**: ✓ CREATED
- **Files**:
  - `.env.library_apis` - Configuration template
  - All API keys optional (fallback to free APIs)
  - Dewey system enabled by default

---

## 2. DOCKER BUILD TESTS

### 2.1 Dockerfile Validation
- **Status**: READY FOR TESTING
- **Test Method**: `docker build` for each service
- **Dockerfiles**:
  1. `Dockerfile.api` - RAG API service
  2. `Dockerfile.chainlit` - Chainlit UI
  3. `Dockerfile.crawl` - Web crawler
  4. `Dockerfile.curation_worker` - Curation processor

- **Expected Outcomes**:
  - All images build without errors
  - Correct base images (python:3.12.7-slim)
  - Multi-stage builds clean site-packages
  - Size targets met (19% reduction)

### 2.2 Build Performance
- **Expected Time**: ~10-15 minutes (first build)
- **Image Sizes Expected**:
  - API: ~950MB (was 1100MB)
  - Crawler: ~350MB (was 550MB)
  - UI: ~280MB (was 320MB)
  - Worker: ~180MB (was 200MB)
  - **Total**: ~1.76GB (was 2.17GB)

---

## 3. DOCKER COMPOSE TESTS

### 3.1 Service Startup
- **Test Script**: `test_docker_integration.sh`
- **Services Tested**:
  1. Redis (cache & streams)
  2. RAG API (FastAPI)
  3. Chainlit UI (web frontend)
  4. Crawler (CrawlModule)
  5. Curation Worker (job processor)

- **Health Checks**:
  - Redis: PING response + password auth
  - RAG API: HTTP health endpoint
  - Chainlit: HTTP health endpoint
  - Crawler: Python import validation
  - Worker: Redis queue validation

### 3.2 Service Communication
- **Test Coverage**:
  - Crawler → Redis communication
  - API → Redis communication
  - API → Crawler communication (via Redis)
  - All services on `xnai_network` bridge

- **Expected Results**:
  - ✓ All services start within 30 seconds
  - ✓ All health checks pass
  - ✓ Services can reach each other

### 3.3 Port Accessibility
- **Expected Open Ports**:
  - 8000: RAG API
  - 8001: Chainlit UI
  - 8002: Prometheus metrics (RAG API)
  - 6379: Redis (internal only)

---

## 4. FUNCTIONAL TESTS

### 4.1 Library API Integration in Docker
- **Test Method**: Tests run inside Docker containers
- **Coverage**:
  - Open Library API search
  - Internet Archive API search
  - Project Gutenberg search
  - Metadata enrichment pipeline
  - Domain classification in containers
  - Dewey Decimal assignment

- **Expected Results**:
  - All API clients initialize correctly
  - Caching works
  - Rate limiting functional
  - Error handling graceful

### 4.2 Crawler Functionality
- **Test Coverage**:
  - URL crawling capability
  - HTML parsing
  - Metadata extraction
  - Cache management
  - Redis queue integration
  - Library metadata enrichment

- **Test URLs** (Safe for testing):
  - `https://example.com` (simple page)
  - `https://python.org/about/gettingstarted/` (medium page)
  - `https://github.com/Xoe-NovAi/Xoe-NovAi` (GitHub repo)

### 4.3 API Endpoints
- **Test Endpoints**:
  - `GET /health` - Health check
  - `POST /rag/query` - RAG query (if available)
  - `POST /crawl` - Initiate crawl (if available)
  - `GET /metrics` - Prometheus metrics

- **Expected Responses**:
  - ✓ 200 OK with proper JSON
  - ✓ Proper error codes (400, 404, 500)
  - ✓ CORS headers if configured

### 4.4 UI Functionality
- **Test Coverage**:
  - Chainlit UI loads at http://localhost:8001
  - Chat interface responsive
  - Connection to RAG API
  - Session management
  - Message history

---

## 5. INTEGRATION TESTS

### 5.1 End-to-End Workflow
1. **Crawl Phase**:
   - User submits URL
   - Crawler extracts content
   - Library metadata enrichment
   - Domain classification
   - Results stored

2. **Curation Phase**:
   - Content analyzed
   - Domain-specific rules applied
   - Dewey Decimal classification
   - Queue for processing

3. **RAG Phase**:
   - Embeddings generated
   - Documents indexed
   - Queries processed
   - Results ranked by relevance

4. **UI Phase**:
   - Results displayed in Chainlit
   - User interaction
   - Session persistence

### 5.2 Multi-Service Communication
- Crawler → Redis (store crawl results)
- Curation Worker → Redis (process queue)
- API → Redis (cache, embeddings)
- API → Crawler (coordination)
- UI → API (queries)

---

## 6. PERFORMANCE TESTS

### 6.1 Response Times
- **Expected**:
  - API health check: <100ms
  - Simple query: <1000ms
  - Crawl operation: <5000ms
  - UI load: <2000ms

### 6.2 Resource Usage
- **Memory Limits**:
  - Redis: 512MB max
  - API: No limit (but monitor)
  - Crawler: No limit (but monitor)
  - UI: No limit (but monitor)

- **CPU Usage**:
  - Normal operation: <20% per service
  - During crawl: <50% crawler
  - During RAG: <30% API

### 6.3 Concurrent Operations
- **Test**: Multiple users
- **Expected**: All requests handled gracefully

---

## 7. ERROR HANDLING TESTS

### 7.1 Service Failures
- **Test**: Kill a service, verify recovery
- **Expected**: Service restarts, others continue

### 7.2 Network Issues
- **Test**: Disconnect and reconnect services
- **Expected**: Graceful degradation, recovery

### 7.3 Invalid Inputs
- **Test**: Malformed URLs, bad JSON, missing fields
- **Expected**: Proper error messages, HTTP error codes

### 7.4 Rate Limiting
- **Test**: Exceed API rate limits
- **Expected**: Graceful 429 responses, automatic backoff

---

## 8. SECURITY TESTS

### 8.1 Container Security
- ✓ Non-root user (appuser:1001)
- ✓ Dropped capabilities
- ✓ No new privileges flag
- ✓ tmpfs for temp files

### 8.2 API Security
- [ ] No sensitive data in logs
- [ ] CORS properly configured
- [ ] Rate limiting active
- [ ] Input validation working

### 8.3 Data Protection
- ✓ Redis password required
- ✓ No telemetry (CRAWL4AI_NO_TELEMETRY=true)
- ✓ Data stored locally only
- ✓ No external tracking

---

## 9. LOGGING & MONITORING

### 9.1 Service Logs
- **Test**: Check logs from all services
- **Expected**:
  - No ERROR level messages
  - Few (if any) WARNING level messages
  - INFO level showing operation flow

### 9.2 Log Aggregation
- **Commands**:
  ```bash
  docker-compose logs -f          # All services
  docker-compose logs -f rag      # RAG API only
  docker-compose logs -f crawler  # Crawler only
  ```

### 9.3 Metrics
- **Prometheus metrics** available at:
  - `http://localhost:8002/metrics`
  - Custom metrics for RAG operations
  - Redis metrics via Redis Exporter (optional)

---

## 10. CLEANUP & SHUTDOWN

### 10.1 Graceful Shutdown
```bash
# Stop containers gracefully
docker-compose down

# Remove all volumes (WARNING: Data loss)
docker-compose down -v

# Stop without removing
docker-compose stop
```

### 10.2 State Verification
- [ ] All containers stopped
- [ ] No zombie processes
- [ ] Volume data persistent (unless -v used)
- [ ] Redis data saved (AOF enabled)

---

## 11. ACCEPTANCE CRITERIA

### Core Functionality
- [ ] All 5 services start and run without errors
- [ ] All services communicate via Redis
- [ ] Health checks pass for all services
- [ ] Logs show no error messages

### Library API Integration
- [ ] Domain classification working
- [ ] Library metadata enrichment functional
- [ ] Dewey Decimal mapping applied
- [ ] All 12 domain categories available

### Performance
- [ ] API responds in <1000ms
- [ ] UI loads in <2000ms
- [ ] Memory usage reasonable (<2GB total)
- [ ] CPU usage <50% under normal load

### Security
- [ ] Non-root user running containers
- [ ] No sensitive data in logs
- [ ] Redis password required
- [ ] Zero telemetry

### Documentation
- [ ] Logs clear and understandable
- [ ] Error messages helpful
- [ ] Configuration documented
- [ ] Startup messages clear

---

## 12. TEST EXECUTION

### Run All Tests
```bash
# From repository root
chmod +x test_docker_integration.sh
./test_docker_integration.sh
```

### Step-by-Step Testing
```bash
# 1. Build images
docker-compose build --no-cache

# 2. Start services
docker-compose up -d

# 3. Wait for startup
sleep 30

# 4. Check services
docker-compose ps
docker-compose logs

# 5. Test specific service
docker exec xnai_crawler python3 -m app.XNAi_rag_app.library_api_integrations

# 6. View metrics
curl http://localhost:8002/metrics | head -20

# 7. Stop services
docker-compose down
```

---

## 13. POST-TESTING ACTIONS

### If All Tests Pass ✓
1. Update PRODUCTION_RELEASE_SUMMARY.md
2. Create GitHub PR with v0.1.4-stable tag
3. Merge to main branch
4. Create release notes
5. Announce v0.1.4-stable release

### If Tests Fail ✗
1. Collect logs from all services
2. Document error messages
3. Check .env configuration
4. Verify Docker/docker-compose versions
5. Check disk space and memory
6. Review recent changes
7. Fix issues and retest

---

## 14. KNOWN LIMITATIONS

### Current Testing Scope
- **Tested**: Local Docker environment
- **Not Tested**: Kubernetes, Swarm, cloud deployments
- **Not Tested**: High-concurrency (100+ simultaneous users)
- **Not Tested**: Extended runtime (24+ hours)

### Library API Limitations
- **Open Library**: Unlimited but needs User-Agent header
- **Google Books**: 100 queries/day free tier
- **ISBNdb**: 100 requests/day free tier
- **Others**: Generally unlimited with reasonable rate limits

---

## 15. SUCCESS METRICS

### Test Coverage
- [ ] Unit tests: 100% codebase
- [ ] Integration tests: All services
- [ ] Performance tests: All endpoints
- [ ] Security tests: All attack vectors tested

### Quality Gates
- [ ] No ERROR logs in production run
- [ ] All health checks pass
- [ ] Response times < thresholds
- [ ] Memory/CPU < limits
- [ ] Zero security issues

### Release Readiness
- [ ] Documentation complete
- [ ] Tests all passing
- [ ] Performance acceptable
- [ ] Security hardened
- [ ] User guide prepared

---

## NEXT STEPS

1. Execute `./test_docker_integration.sh`
2. Review all test outputs
3. Fix any identified issues
4. Verify acceptance criteria met
5. Update documentation
6. Create GitHub PR
7. Tag release as v0.1.4-stable

---

**Status**: ✅ READY FOR DOCKER TESTING

**Test Date**: [Fill in when testing begins]
**Test Results**: [To be updated with test runs]
**Release Decision**: [To be decided after testing]

---

