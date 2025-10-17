#!/usr/bin/env python3
# ============================================================================
# Xoe-NovAi Phase 1 v0.1.1 - Prometheus Metrics Module
# ============================================================================
# Purpose: Real-time metrics collection and exposure for monitoring
# Guide Reference: Section 5.2 (Prometheus Metrics)
# Last Updated: 2025-10-11
# Features:
#   - 9 metrics (3 gauges, 2 histograms, 4 counters)
#   - Automatic background updates (30s interval)
#   - HTTP server on port 8002
#   - Multiprocess mode for Gunicorn/Uvicorn
#   - Performance targets validation
# ============================================================================

import os
import time
import logging
import threading
from typing import Dict, Any, Optional
from pathlib import Path

# Prometheus client
from prometheus_client import (
    start_http_server,
    Gauge,
    Histogram,
    Counter,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    multiprocess,
    MetricsHandler
)

# System monitoring
import psutil

# Configuration
from config_loader import load_config, get_config_value

logger = logging.getLogger(__name__)
CONFIG = load_config()

# ============================================================================
# METRICS DEFINITIONS
# ============================================================================

# Gauges (current state)
memory_usage_gb = Gauge(
    'xnai_memory_usage_gb',
    'Current memory usage in gigabytes',
    ['component']  # Labels: 'system', 'process', 'llm', 'embeddings'
)

token_rate_tps = Gauge(
    'xnai_token_rate_tps',
    'Token generation rate in tokens per second',
    ['model']  # Labels: 'gemma-3-4b'
)

active_sessions = Gauge(
    'xnai_active_sessions',
    'Number of active user sessions'
)

# Histograms (distributions)
response_latency_ms = Histogram(
    'xnai_response_latency_ms',
    'API response latency in milliseconds',
    ['endpoint', 'method'],  # Labels: endpoint path, HTTP method
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]  # Milliseconds
)

rag_retrieval_time_ms = Histogram(
    'xnai_rag_retrieval_time_ms',
    'RAG document retrieval time in milliseconds',
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000]
)

# Counters (cumulative)
requests_total = Counter(
    'xnai_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status']  # Labels: path, method, status code
)

errors_total = Counter(
    'xnai_errors_total',
    'Total number of errors',
    ['error_type', 'component']  # Labels: error type, component name
)

tokens_generated_total = Counter(
    'xnai_tokens_generated_total',
    'Total tokens generated',
    ['model']  # Labels: model name
)

queries_processed_total = Counter(
    'xnai_queries_processed_total',
    'Total queries processed',
    ['rag_enabled']  # Labels: 'true', 'false'
)

# Info (metadata)
stack_info = Info(
    'xnai_stack',
    'Stack version and metadata'
)

# ============================================================================
# METRICS TIMER (Context Manager)
# ============================================================================

class MetricsTimer:
    """
    Context manager for timing operations and recording to histogram.
    
    Guide Reference: Section 5.2 (Metrics Timer)
    
    Example:
        >>> with MetricsTimer(response_latency_ms, endpoint='/query', method='POST'):
        ...     # Process query
        ...     pass
        # Automatically records duration to histogram
    """
    
    def __init__(
        self,
        histogram: Histogram,
        **labels
    ):
        """
        Initialize timer.
        
        Args:
            histogram: Histogram to record to
            **labels: Label values for histogram
        """
        self.histogram = histogram
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and record duration."""
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.histogram.labels(**self.labels).observe(duration_ms)

# ============================================================================
# METRICS UPDATE FUNCTIONS
# ============================================================================

def update_memory_metrics():
    """
    Update memory usage metrics.
    
    Guide Reference: Section 5.2 (Memory Metrics)
    
    This records:
    - System memory usage
    - Process memory usage
    - Component-specific memory (if available)
    """
    try:
        # System memory
        memory = psutil.virtual_memory()
        system_used_gb = memory.used / (1024 ** 3)
        memory_usage_gb.labels(component='system').set(system_used_gb)
        
        # Process memory
        process = psutil.Process()
        process_used_gb = process.memory_info().rss / (1024 ** 3)
        memory_usage_gb.labels(component='process').set(process_used_gb)
        
        # Log warning if approaching limit
        memory_limit = CONFIG['performance']['memory_limit_gb']
        warning_threshold = CONFIG['performance']['memory_warning_threshold_gb']
        
        if system_used_gb > warning_threshold:
            logger.warning(
                f"Memory usage high: {system_used_gb:.2f}GB / {memory_limit:.1f}GB"
            )
        
    except Exception as e:
        logger.error(f"Failed to update memory metrics: {e}")
        errors_total.labels(error_type='metrics', component='memory').inc()

def update_cpu_metrics():
    """
    Update CPU-related metrics.
    
    Guide Reference: Section 5.2 (CPU Metrics)
    
    This could record CPU usage, but we focus on token rate instead.
    """
    # CPU metrics are less critical for this stack
    # Token rate is the key performance indicator
    pass

def update_stack_info():
    """
    Update stack metadata.
    
    Guide Reference: Section 5.2 (Stack Info)
    
    This sets static information about the stack version and config.
    """
    try:
        stack_info.info({
            'version': CONFIG['metadata']['stack_version'],
            'codename': CONFIG['metadata']['codename'],
            'phase': str(CONFIG['project']['phase']),
            'architecture': CONFIG['metadata']['architecture'],
            'cpu_threads': str(CONFIG['performance']['cpu_threads']),
            'memory_limit_gb': str(CONFIG['performance']['memory_limit_gb']),
            'f16_kv_enabled': str(CONFIG['performance']['f16_kv_enabled']),
        })
    except Exception as e:
        logger.error(f"Failed to update stack info: {e}")

# ============================================================================
# BACKGROUND METRICS UPDATER
# ============================================================================

class MetricsUpdater:
    """
    Background thread for updating gauges periodically.
    
    Guide Reference: Section 5.2 (Background Updates)
    
    This runs in a daemon thread and updates metrics every 30 seconds.
    """
    
    def __init__(self, interval_s: int = 30):
        """
        Initialize updater.
        
        Args:
            interval_s: Update interval in seconds
        """
        self.interval_s = interval_s
        self.running = False
        self.thread = None
    
    def start(self):
        """Start background updater thread."""
        if self.running:
            logger.warning("Metrics updater already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        
        logger.info(f"Metrics updater started (interval: {self.interval_s}s)")
    
    def stop(self):
        """Stop background updater thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info("Metrics updater stopped")
    
    def _update_loop(self):
        """
        Background update loop.
        
        This runs continuously, updating metrics every interval_s seconds.
        """
        # Initial update
        self._update_all()
        
        # Periodic updates
        while self.running:
            try:
                time.sleep(self.interval_s)
                if self.running:
                    self._update_all()
            except Exception as e:
                logger.error(f"Metrics update loop error: {e}")
                errors_total.labels(error_type='metrics', component='updater').inc()
    
    def _update_all(self):
        """Update all gauge metrics."""
        try:
            update_memory_metrics()
            update_cpu_metrics()
            # Stack info only needs to be set once, but safe to call multiple times
            update_stack_info()
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")

# Global updater instance
_metrics_updater = None

# ============================================================================
# METRICS SERVER
# ============================================================================

def start_metrics_server(port: int = None):
    """
    Start Prometheus metrics HTTP server.
    
    Guide Reference: Section 5.2 (Metrics Server)
    
    This starts an HTTP server on the specified port (default: 8002)
    and begins background metrics updates.
    
    Args:
        port: HTTP port (default: from config)
        
    Example:
        >>> start_metrics_server(port=8002)
        >>> # Metrics available at http://localhost:8002/metrics
    """
    global _metrics_updater
    
    # Get port from config if not specified
    if port is None:
        port = get_config_value('metrics.port', 8002)
    
    # Check if metrics enabled
    if not get_config_value('metrics.enabled', True):
        logger.info("Metrics disabled in configuration")
        return
    
    # Start HTTP server
    try:
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
    except OSError as e:
        if "Address already in use" in str(e):
            logger.warning(f"Metrics server already running on port {port}")
        else:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    # Start background updater
    if _metrics_updater is None:
        interval_s = get_config_value('metrics.update_interval_s', 30)
        _metrics_updater = MetricsUpdater(interval_s=interval_s)
        _metrics_updater.start()

def stop_metrics_server():
    """
    Stop metrics server and background updater.
    
    Guide Reference: Section 5.2 (Metrics Shutdown)
    """
    global _metrics_updater
    
    if _metrics_updater:
        _metrics_updater.stop()
        _metrics_updater = None

# ============================================================================
# CONVENIENCE FUNCTIONS FOR APPLICATION CODE
# ============================================================================

def record_request(
    endpoint: str,
    method: str,
    status: int
):
    """
    Record an API request.
    
    Guide Reference: Section 5.2 (Request Recording)
    
    Args:
        endpoint: Endpoint path (e.g., '/query')
        method: HTTP method (e.g., 'POST')
        status: HTTP status code (e.g., 200)
        
    Example:
        >>> record_request('/query', 'POST', 200)
    """
    requests_total.labels(
        endpoint=endpoint,
        method=method,
        status=str(status)
    ).inc()

def record_error(
    error_type: str,
    component: str
):
    """
    Record an error.
    
    Args:
        error_type: Type of error (e.g., 'timeout', 'validation', 'llm')
        component: Component where error occurred (e.g., 'api', 'rag', 'llm')
        
    Example:
        >>> record_error('timeout', 'llm')
    """
    errors_total.labels(
        error_type=error_type,
        component=component
    ).inc()

def record_tokens_generated(
    tokens: int,
    model: str = 'gemma-3-4b'
):
    """
    Record tokens generated.
    
    Args:
        tokens: Number of tokens generated
        model: Model name
        
    Example:
        >>> record_tokens_generated(50, model='gemma-3-4b')
    """
    tokens_generated_total.labels(model=model).inc(tokens)

def record_query_processed(rag_enabled: bool):
    """
    Record a processed query.
    
    Args:
        rag_enabled: Whether RAG was used
        
    Example:
        >>> record_query_processed(rag_enabled=True)
    """
    queries_processed_total.labels(
        rag_enabled=str(rag_enabled).lower()
    ).inc()

def update_token_rate(
    tokens_per_second: float,
    model: str = 'gemma-3-4b'
):
    """
    Update token generation rate gauge.
    
    Args:
        tokens_per_second: Current token rate
        model: Model name
        
    Example:
        >>> update_token_rate(20.5)
    """
    token_rate_tps.labels(model=model).set(tokens_per_second)

def update_active_sessions(count: int):
    """
    Update active sessions count.
    
    Args:
        count: Number of active sessions
        
    Example:
        >>> update_active_sessions(5)
    """
    active_sessions.set(count)

def record_rag_retrieval(duration_ms: float):
    """
    Record RAG document retrieval time.
    
    Args:
        duration_ms: Retrieval time in milliseconds
        
    Example:
        >>> record_rag_retrieval(45.2)
    """
    rag_retrieval_time_ms.observe(duration_ms)

# ============================================================================
# PERFORMANCE VALIDATION
# ============================================================================

def check_performance_targets() -> Dict[str, Any]:
    """
    Check if current metrics meet performance targets.
    
    Guide Reference: Section 5.2 (Performance Validation)
    
    Returns:
        Dict with validation results
        
    Example:
        >>> results = check_performance_targets()
        >>> print(results['memory']['status'])
        'OK'
    """
    results = {}
    
    # Memory check
    try:
        memory = psutil.virtual_memory()
        memory_gb = memory.used / (1024 ** 3)
        memory_limit = CONFIG['performance']['memory_limit_gb']
        
        results['memory'] = {
            'current_gb': round(memory_gb, 2),
            'limit_gb': memory_limit,
            'status': 'OK' if memory_gb < memory_limit else 'EXCEEDED',
        }
    except Exception as e:
        results['memory'] = {'status': 'ERROR', 'error': str(e)}
    
    # Token rate check (would need to track recent generations)
    # This is more complex - typically done via monitoring dashboard
    results['token_rate'] = {
        'target_min': CONFIG['performance']['token_rate_min'],
        'target_max': CONFIG['performance']['token_rate_max'],
        'status': 'UNKNOWN',  # Would need tracking
    }
    
    return results

# ============================================================================
# MULTIPROCESS MODE (for Gunicorn/Uvicorn workers)
# ============================================================================

def setup_multiprocess_metrics(multiproc_dir: str = None):
    """
    Setup metrics for multiprocess mode.
    
    Guide Reference: Section 5.2 (Multiprocess Metrics)
    
    This is needed when running with multiple Uvicorn workers.
    
    Args:
        multiproc_dir: Directory for shared metrics (default: from config)
        
    Example:
        >>> setup_multiprocess_metrics('/prometheus_data')
    """
    if multiproc_dir is None:
        multiproc_dir = get_config_value('metrics.multiproc_dir', '/prometheus_data')
    
    # Ensure directory exists
    Path(multiproc_dir).mkdir(parents=True, exist_ok=True)
    
    # Set environment variable
    os.environ['prometheus_multiproc_dir'] = multiproc_dir
    
    logger.info(f"Multiprocess metrics enabled: {multiproc_dir}")

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test metrics module.
    
    Usage: python3 metrics.py
    
    This validates the metrics module and generates test data.
    """
    import sys
    
    print("=" * 70)
    print("Xoe-NovAi Metrics Module - Test Suite")
    print("=" * 70)
    print()
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Start metrics server
    print("Test 1: Start metrics server")
    try:
        start_metrics_server(port=8002)
        print("✓ Metrics server started on port 8002")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Failed to start metrics server: {e}")
        tests_failed += 1
    
    print()
    
    # Test 2: Record sample metrics
    print("Test 2: Record sample metrics")
    try:
        # Record requests
        record_request('/query', 'POST', 200)
        record_request('/query', 'POST', 200)
        record_request('/health', 'GET', 200)
        
        # Record tokens
        record_tokens_generated(100, model='gemma-3-4b')
        record_tokens_generated(50, model='gemma-3-4b')
        
        # Update gauges
        update_token_rate(20.5)
        update_active_sessions(3)
        
        # Record query
        record_query_processed(rag_enabled=True)
        
        # Record RAG retrieval
        record_rag_retrieval(45.2)
        
        print("✓ Sample metrics recorded")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Failed to record metrics: {e}")
        tests_failed += 1
    
    print()
    
    # Test 3: Test timer context manager
    print("Test 3: Test timer context manager")
    try:
        with MetricsTimer(response_latency_ms, endpoint='/test', method='GET'):
            time.sleep(0.1)  # Simulate 100ms operation
        
        print("✓ Timer context manager works")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Timer test failed: {e}")
        tests_failed += 1
    
    print()
    
    # Test 4: Update memory metrics
    print("Test 4: Update memory metrics")
    try:
        update_memory_metrics()
        print("✓ Memory metrics updated")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Memory metrics failed: {e}")
        tests_failed += 1
    
    print()
    
    # Test 5: Check performance targets
    print("Test 5: Check performance targets")
    try:
        results = check_performance_targets()
        print(f"✓ Performance check: {results['memory']['status']}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Performance check failed: {e}")
        tests_failed += 1
    
    print()
    
    # Test 6: Generate metrics output
    print("Test 6: Generate metrics output")
    try:
        from prometheus_client import generate_latest
        metrics_output = generate_latest().decode('utf-8')
        
        # Check for expected metrics
        expected_metrics = [
            'xnai_memory_usage_gb',
            'xnai_token_rate_tps',
            'xnai_requests_total',
            'xnai_tokens_generated_total',
        ]
        
        found = [m for m in expected_metrics if m in metrics_output]
        
        print(f"✓ Found {len(found)}/{len(expected_metrics)} expected metrics")
        print(f"  Sample output (first 500 chars):")
        print(f"  {metrics_output[:500]}")
        
        if len(found) == len(expected_metrics):
            tests_passed += 1
        else:
            print(f"  Missing: {set(expected_metrics) - set(found)}")
            tests_failed += 1
    except Exception as e:
        print(f"✗ Metrics output test failed: {e}")
        tests_failed += 1
    
    print()
    
    # Wait a bit for background updater
    print("Waiting 2s for background updater...")
    time.sleep(2)
    
    # Final summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Passed: {tests_passed}")
    print(f"Failed: {tests_failed}")
    print()
    
    if tests_failed == 0:
        print("✓ All tests passed!")
        print()
        print("Metrics server is running at http://localhost:8002/metrics")
        print("Press Ctrl+C to stop...")
        print()
        
        # Keep server running for manual testing
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping metrics server...")
            stop_metrics_server()
        
        sys.exit(0)
    else:
        print(f"✗ {tests_failed} test(s) failed")
        stop_metrics_server()
        sys.exit(1)
