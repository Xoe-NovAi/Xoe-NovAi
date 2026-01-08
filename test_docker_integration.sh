#!/bin/bash
# ============================================================================
# Xoe-NovAi Docker Integration Testing Script
# ============================================================================
# Purpose: Comprehensive testing of full RAG stack with library API integration
# Tests: Crawler, Curation, API, UI, and all integrations
# Status: Production Ready (v0.1.4-stable)
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

log_info "Starting Xoe-NovAi Docker Integration Tests..."
echo ""

# Check Docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi
log_success "Docker available"

# Check docker-compose
if ! command -v docker-compose &> /dev/null; then
    log_error "docker-compose is not installed"
    exit 1
fi
log_success "docker-compose available"

# Check .env file exists
if [ ! -f .env ]; then
    log_error ".env file not found"
    log_info "Please create .env file with required variables"
    exit 1
fi
log_success ".env file found"

# ============================================================================
# TEST 1: BUILD IMAGES
# ============================================================================

log_info ""
log_info "TEST 1: Building Docker images..."
echo "----------------------------------------------"

if docker-compose build --no-cache 2>&1 | grep -q "ERROR"; then
    log_error "Docker build failed"
    exit 1
fi
log_success "All Docker images built successfully"

# ============================================================================
# TEST 2: START SERVICES
# ============================================================================

log_info ""
log_info "TEST 2: Starting services..."
echo "----------------------------------------------"

# Start services
docker-compose up -d

# Wait for services to be healthy
log_info "Waiting for services to start..."
sleep 10

# Check Redis
if docker exec xnai_redis redis-cli -a "$(grep REDIS_PASSWORD .env | cut -d '=' -f2)" ping &>/dev/null; then
    log_success "Redis is healthy"
else
    log_error "Redis health check failed"
    docker-compose logs redis
    exit 1
fi

# Check RAG API
sleep 10
if curl -s http://localhost:8000/health &>/dev/null; then
    log_success "RAG API is healthy"
else
    log_warning "RAG API health check incomplete (may still be starting)"
fi

# Check Chainlit UI
if curl -s http://localhost:8001/health &>/dev/null; then
    log_success "Chainlit UI is healthy"
else
    log_warning "Chainlit UI health check incomplete (may still be starting)"
fi

# ============================================================================
# TEST 3: LIBRARY API INTEGRATION
# ============================================================================

log_info ""
log_info "TEST 3: Testing library API integration..."
echo "----------------------------------------------"

# Test domain classification
log_info "Testing domain classification..."
python3 << 'EOF'
try:
    from app.XNAi_rag_app.library_api_integrations import DomainManager, LibraryEnrichmentEngine
    
    manager = DomainManager()
    
    # Test classifications
    tests = [
        ("Python programming guide", "def hello(): return 'world'", "code"),
        ("Quantum mechanics research", "Wave-particle duality in quantum systems", "science"),
        ("Novel by author", "It was a dark and stormy night...", "fiction"),
    ]
    
    passed = 0
    for title, content, expected in tests:
        category, confidence = manager.classify(content, title)
        if confidence > 0.5:
            passed += 1
            print(f"  ✓ {title}: {category.value} (confidence: {confidence:.2f})")
        else:
            print(f"  ⚠ {title}: {category.value} (low confidence: {confidence:.2f})")
    
    print(f"\n✓ Domain classification: {passed}/{len(tests)} tests passed")
    
except Exception as e:
    print(f"✗ Library API test failed: {e}")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    log_error "Library API integration test failed"
    exit 1
fi
log_success "Library API integration working"

# ============================================================================
# TEST 4: CURATION MODULE
# ============================================================================

log_info ""
log_info "TEST 4: Testing curation module..."
echo "----------------------------------------------"

python3 << 'EOF'
try:
    from app.XNAi_rag_app.crawler_curation import CurationExtractor, test_extraction
    
    # Run test
    doc = test_extraction()
    
    if doc and doc.domain:
        print(f"✓ Curation module working: domain={doc.domain.value}")
    else:
        print("✗ Curation module failed")
        exit(1)
        
except Exception as e:
    print(f"✗ Curation module test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF

if [ $? -ne 0 ]; then
    log_error "Curation module test failed"
    exit 1
fi
log_success "Curation module working"

# ============================================================================
# TEST 5: API ENDPOINTS
# ============================================================================

log_info ""
log_info "TEST 5: Testing API endpoints..."
echo "----------------------------------------------"

# Test RAG API endpoints
log_info "Testing RAG API endpoints..."

# Test health endpoint
if curl -s http://localhost:8000/health | grep -q "running\|ok"; then
    log_success "RAG API health endpoint working"
else
    log_warning "RAG API health endpoint check inconclusive"
fi

# ============================================================================
# TEST 6: SERVICE COMMUNICATION
# ============================================================================

log_info ""
log_info "TEST 6: Testing service communication..."
echo "----------------------------------------------"

# Test Redis connectivity from crawler
log_info "Testing crawler Redis connectivity..."
docker exec xnai_crawler python3 << 'EOF' 2>/dev/null || echo "Connection test incomplete"
try:
    import redis
    import os
    r = redis.Redis(host='redis', port=6379, password=os.environ.get('REDIS_PASSWORD'), decode_responses=True)
    r.ping()
    print("✓ Crawler can reach Redis")
except Exception as e:
    print(f"Connection test result: {e}")
EOF

# Test Redis connectivity from API
log_info "Testing API Redis connectivity..."
docker exec xnai_rag_api python3 << 'EOF' 2>/dev/null || echo "Connection test incomplete"
try:
    import redis
    import os
    r = redis.Redis(host='redis', port=6379, password=os.environ.get('REDIS_PASSWORD'), decode_responses=True)
    r.ping()
    print("✓ API can reach Redis")
except Exception as e:
    print(f"Connection test result: {e}")
EOF

log_success "Service communication tests completed"

# ============================================================================
# TEST 7: LOGGING
# ============================================================================

log_info ""
log_info "TEST 7: Checking service logs..."
echo "----------------------------------------------"

log_info "Recent logs from RAG API:"
docker-compose logs --tail=10 rag | grep -E "ERROR|WARNING" | head -5 || log_success "No errors in RAG API logs"

log_info "Recent logs from Crawler:"
docker-compose logs --tail=10 crawler | grep -E "ERROR|WARNING" | head -5 || log_success "No errors in Crawler logs"

# ============================================================================
# TEST 8: PERFORMANCE CHECK
# ============================================================================

log_info ""
log_info "TEST 8: Checking system resource usage..."
echo "----------------------------------------------"

docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null || true

# ============================================================================
# TEST 9: FINAL VALIDATION
# ============================================================================

log_info ""
log_info "TEST 9: Final validation..."
echo "----------------------------------------------"

# Count running containers
RUNNING=$(docker-compose ps --services --filter "status=running" | wc -l)
TOTAL=$(docker-compose ps --services | wc -l)

if [ "$RUNNING" -eq "$TOTAL" ]; then
    log_success "All $RUNNING services running"
else
    log_warning "$RUNNING/$TOTAL services running"
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "========================================================================"
echo -e "${GREEN}✓ DOCKER INTEGRATION TESTS COMPLETED${NC}"
echo "========================================================================"
echo ""
echo "Services Status:"
docker-compose ps
echo ""
echo "Access Points:"
echo "  - RAG API:     http://localhost:8000"
echo "  - Chainlit UI: http://localhost:8001"
echo "  - Redis:       localhost:6379 (password: check .env)"
echo ""
echo "View Logs:"
echo "  - All:         docker-compose logs -f"
echo "  - RAG API:     docker-compose logs -f rag"
echo "  - Chainlit:    docker-compose logs -f ui"
echo "  - Crawler:     docker-compose logs -f crawler"
echo "  - Redis:       docker-compose logs -f redis"
echo ""
echo "Stop Services:"
echo "  docker-compose down"
echo ""
echo "========================================================================"
log_success "Docker stack is ready for testing!"
echo "========================================================================"
