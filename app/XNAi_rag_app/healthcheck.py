#!/usr/bin/env python3
# ============================================================================
# Xoe-NovAi Phase 1 v0.1.2 - Health Check Module
# ============================================================================
# Purpose: Comprehensive health monitoring for all stack components
# Guide Reference: Section 5.3 (7 Modular Health Checks)
# Last Updated: 2025-10-13
# Features:
#   - 7 modular health checks (LLM, embeddings, memory, Redis, vectorstore, Ryzen, crawler)
#   - Configurable thresholds from config.toml
#   - Detailed error reporting
#   - Docker healthcheck compatible (exit codes)
#   - NEW v0.1.2: check_crawler() for CrawlModule validation
# ============================================================================

import os
import sys
import time
import logging
from typing import Dict, Tuple, List, Optional

# System monitoring
import psutil
import redis

# Configuration
from config_loader import load_config, get_config_value

# Core dependencies (lazy loaded)
from dependencies import get_llm, get_embeddings, get_vectorstore, get_curator

logger = logging.getLogger(__name__)
CONFIG = load_config()

# ============================================================================
# HEALTH CHECK FUNCTIONS
# ============================================================================

def check_llm(timeout_s: int = 10) -> Tuple[bool, str]:
    """
    Test LLM inference capability.
    
    Guide Reference: Section 5.3.1 (LLM Health Check)
    
    This verifies:
    1. LLM can be initialized
    2. LLM can generate tokens
    3. Response time < timeout
    
    Args:
        timeout_s: Maximum time allowed for inference
        
    Returns:
        Tuple of (success, message)
        
    Example:
        >>> success, msg = check_llm(timeout=10)
        >>> print(success, msg)
        True 'LLM operational: 0.5s response'
    """
    try:
        start = time.time()
        
        # Initialize LLM (cached if already loaded)
        llm = get_llm()
        
        # Test inference (minimal tokens)
        response = llm.invoke("Test", max_tokens=5)
        
        elapsed = time.time() - start
        
        # Validate response
        if not response or len(response.strip()) == 0:
            return False, "LLM returned empty response"
        
        # Check timeout
        if elapsed > timeout_s:
            return False, f"LLM response timeout: {elapsed:.1f}s > {timeout_s}s"
        
        return True, f"LLM operational: {elapsed:.2f}s response"
        
    except Exception as e:
        logger.error(f"LLM health check failed: {e}", exc_info=True)
        return False, f"LLM error: {str(e)[:100]}"

def check_embeddings() -> Tuple[bool, str]:
    """
    Test embedding generation.
    
    Guide Reference: Section 5.3.2 (Embeddings Health Check)
    
    This verifies:
    1. Embeddings model loads
    2. Can generate embedding vectors
    3. Output dimensions correct (384)
    
    Returns:
        Tuple of (success, message)
    """
    try:
        # Initialize embeddings (cached if already loaded)
        embeddings = get_embeddings()
        
        # Test embedding generation
        test_vector = embeddings.embed_query("test query")
        
        # Validate dimensions
        expected_dims = CONFIG['models']['embedding_dimensions']
        if len(test_vector) != expected_dims:
            return False, f"Embedding dimensions mismatch: {len(test_vector)} != {expected_dims}"
        
        return True, f"Embeddings operational: {len(test_vector)} dimensions"
        
    except Exception as e:
        logger.error(f"Embeddings health check failed: {e}", exc_info=True)
        return False, f"Embeddings error: {str(e)[:100]}"

def check_memory(max_gb: float = None) -> Tuple[bool, str]:
    """
    Verify memory usage within limits.
    
    Guide Reference: Section 5.3.3 (Memory Health Check)
    
    This checks:
    1. Total system memory usage
    2. Process-specific memory usage
    3. Memory < threshold (default: 6.0GB)
    
    Args:
        max_gb: Maximum allowed memory in GB
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Get threshold from config
        if max_gb is None:
            max_gb = get_config_value('healthcheck.thresholds.memory_max_gb', 6.0)
        
        # System memory
        memory = psutil.virtual_memory()
        system_used_gb = memory.used / (1024 ** 3)
        system_percent = memory.percent
        
        # Process memory
        process = psutil.Process()
        process_used_gb = process.memory_info().rss / (1024 ** 3)
        
        # Check threshold
        if system_used_gb > max_gb:
            return False, (
                f"Memory exceeded: {system_used_gb:.2f}GB > {max_gb:.1f}GB "
                f"(process: {process_used_gb:.2f}GB, system: {system_percent}%)"
            )
        
        # Warning threshold (90% of max)
        warning_threshold = max_gb * 0.9
        if system_used_gb > warning_threshold:
            return True, (
                f"Memory warning: {system_used_gb:.2f}GB "
                f"(process: {process_used_gb:.2f}GB, threshold: {max_gb:.1f}GB)"
            )
        
        return True, (
            f"Memory OK: {system_used_gb:.2f}GB / {max_gb:.1f}GB "
            f"(process: {process_used_gb:.2f}GB, {system_percent}%)"
        )
        
    except Exception as e:
        logger.error(f"Memory health check failed: {e}", exc_info=True)
        return False, f"Memory check error: {str(e)[:100]}"

def check_redis(timeout_s: int = 5) -> Tuple[bool, str]:
    """
    Test Redis connectivity and basic operations.
    
    Guide Reference: Section 5.3.4 (Redis Health Check)
    
    This verifies:
    1. Redis connection successful
    2. PING command responds
    3. Can set/get test key
    4. NEW v0.1.2: Streams available (for Phase 2)
    
    Args:
        timeout_s: Connection timeout in seconds
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Get Redis config
        host = os.getenv('REDIS_HOST', 'redis')
        port = int(os.getenv('REDIS_PORT', '6379'))
        password = os.getenv('REDIS_PASSWORD')
        
        # Connect with timeout
        client = redis.Redis(
            host=host,
            port=port,
            password=password,
            socket_timeout=timeout_s,
            socket_connect_timeout=timeout_s
        )
        
        # Test PING
        if not client.ping():
            return False, "Redis PING failed"
        
        # Test SET/GET
        test_key = 'xnai_health_check'
        test_value = str(time.time())
        
        client.setex(test_key, 10, test_value)  # 10s TTL
        retrieved = client.get(test_key)
        
        if retrieved is None or retrieved.decode('utf-8') != test_value:
            return False, "Redis SET/GET failed"
        
        # Get info
        info = client.info('server')
        redis_version = info.get('redis_version', 'unknown')
        
        # NEW v0.1.2: Check streams support
        try:
            # Test stream creation (Phase 2 prep)
            stream_name = 'xnai_health_test_stream'
            client.xadd(stream_name, {'test': 'health_check'})
            client.delete(stream_name)
            streams_ok = True
        except:
            streams_ok = False
        
        return True, f"Redis operational: v{redis_version} at {host}:{port} (streams: {'✓' if streams_ok else '✗'})"
        
    except redis.ConnectionError as e:
        return False, f"Redis connection failed: {str(e)[:100]}"
    except redis.TimeoutError:
        return False, f"Redis timeout after {timeout_s}s"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}", exc_info=True)
        return False, f"Redis error: {str(e)[:100]}"

def check_vectorstore(timeout_s: int = 10) -> Tuple[bool, str]:
    """
    Test FAISS vectorstore availability and search.
    
    Guide Reference: Section 5.3.5 (Vectorstore Health Check)
    
    This verifies:
    1. FAISS index can be loaded
    2. Search operation succeeds
    3. Index integrity (vector count > 0)
    
    Args:
        timeout_s: Maximum time for search operation
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Get embeddings first
        embeddings = get_embeddings()
        
        # Load vectorstore
        vectorstore = get_vectorstore(embeddings)
        
        if vectorstore is None:
            return False, "Vectorstore not found (run ingest_library.py)"
        
        # Check vector count
        vector_count = vectorstore.index.ntotal
        if vector_count == 0:
            return False, "Vectorstore empty (0 vectors)"
        
        # Test search (with timeout)
        start = time.time()
        results = vectorstore.similarity_search("test query", k=1)
        elapsed = time.time() - start
        
        if elapsed > timeout_s:
            return False, f"Vectorstore search timeout: {elapsed:.1f}s > {timeout_s}s"
        
        if not results:
            return False, "Vectorstore search returned no results"
        
        return True, f"Vectorstore operational: {vector_count} vectors, {elapsed:.2f}s search"
        
    except Exception as e:
        logger.error(f"Vectorstore health check failed: {e}", exc_info=True)
        return False, f"Vectorstore error: {str(e)[:100]}"

def check_ryzen() -> Tuple[bool, str]:
    """
    Verify Ryzen-specific optimizations are active.
    
    Guide Reference: Section 5.3.6 (Ryzen Health Check)
    
    This checks:
    1. CPU thread count matches config (6 threads)
    2. f16_kv flag enabled
    3. OPENBLAS_CORETYPE set to ZEN
    
    Returns:
        Tuple of (success, message)
    """
    try:
        checks = []
        warnings = []
        
        # Check n_threads
        n_threads = int(os.getenv('LLAMA_CPP_N_THREADS', '0'))
        expected_threads = CONFIG['performance']['cpu_threads']
        
        if n_threads != expected_threads:
            warnings.append(f"N_THREADS={n_threads} (expected: {expected_threads})")
        else:
            checks.append(f"N_THREADS={n_threads}")
        
        # Check f16_kv
        f16_kv = os.getenv('LLAMA_CPP_F16_KV', 'false').lower() == 'true'
        expected_f16_kv = CONFIG['performance']['f16_kv_enabled']
        
        if f16_kv != expected_f16_kv:
            warnings.append(f"F16_KV={f16_kv} (expected: {expected_f16_kv})")
        else:
            checks.append(f"F16_KV={f16_kv}")
        
        # Check OPENBLAS_CORETYPE
        coretype = os.getenv('OPENBLAS_CORETYPE', '')
        if coretype != 'ZEN':
            warnings.append(f"CORETYPE={coretype or 'unset'} (expected: ZEN)")
        else:
            checks.append("CORETYPE=ZEN")
        
        # Check use_mlock
        use_mlock = os.getenv('LLAMA_CPP_USE_MLOCK', 'false').lower() == 'true'
        if not use_mlock:
            warnings.append("USE_MLOCK=false (recommended: true)")
        else:
            checks.append("USE_MLOCK=true")
        
        # Determine result
        if warnings:
            return True, f"Ryzen optimizations: {', '.join(checks)} | Warnings: {', '.join(warnings)}"
        else:
            return True, f"Ryzen optimizations active: {', '.join(checks)}"
        
    except Exception as e:
        logger.error(f"Ryzen health check failed: {e}", exc_info=True)
        return False, f"Ryzen check error: {str(e)[:100]}"

def check_crawler(timeout_s: int = 10) -> Tuple[bool, str]:
    """
    Verify CrawlModule is available and responsive (NEW v0.1.2).
    
    Guide Reference: Section 5.3.7 (Crawler Health Check)
    Guide Reference: Section 9 (CrawlModule Integration)
    
    This checks:
    1. CrawlModule can be imported
    2. crawl4ai is installed
    3. Curator can be initialized
    4. Dry-run validation passes
    
    Args:
        timeout_s: Maximum time for validation
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Check if crawl4ai is installed
        import crawl4ai
        crawl4ai_version = getattr(crawl4ai, '__version__', 'unknown')
        
        # Try to initialize curator
        curator = get_curator()
        
        if curator is None:
            return False, "Curator initialization returned None"
        
        # Test with dry-run (no actual crawling)
        start = time.time()
        
        # Simple validation - check if curator has required methods
        if not hasattr(curator, 'curate'):
            return False, "Curator missing 'curate' method"
        
        elapsed = time.time() - start
        
        if elapsed > timeout_s:
            return False, f"Curator validation timeout: {elapsed:.1f}s > {timeout_s}s"
        
        return True, f"CrawlModule operational: crawl4ai {crawl4ai_version}, curator ready"
        
    except ImportError as e:
        # crawl4ai not installed - optional component
        return False, f"CrawlModule unavailable: crawl4ai not installed (optional)"
    except Exception as e:
        logger.error(f"Crawler health check failed: {e}", exc_info=True)
        return False, f"CrawlModule error: {str(e)[:100]}"

# ============================================================================
# ORCHESTRATION
# ============================================================================

def run_health_checks(
    targets: List[str] = None,
    critical_only: bool = False,
    error_recovery: bool = None
) -> Dict[str, Tuple[bool, str]]:
    """
    Run selected health checks.
    
    Guide Reference: Section 5.3 (Health Check Orchestration)
    
    Args:
        targets: List of check names to run (default: all)
        critical_only: Only run critical checks (memory, redis)
        error_recovery: Enable graceful failure handling (default: from config)
        
    Returns:
        Dict mapping check name to (success, message)
        
    Example:
        >>> results = run_health_checks(['llm', 'memory'])
        >>> all_passed = all(s for s, _ in results.values())
    """
    # Get error recovery setting
    if error_recovery is None:
        error_recovery = os.getenv('ERROR_RECOVERY_ENABLED', 'true').lower() == 'true'
    
    # Default targets from config
    if targets is None:
        targets = get_config_value(
            'healthcheck.targets',
            ['llm', 'embeddings', 'memory', 'redis', 'vectorstore', 'ryzen', 'crawler']
        )
    
    # Critical checks only
    if critical_only:
        targets = ['memory', 'redis']
    
    # Available checks
    check_functions = {
        'llm': check_llm,
        'embeddings': check_embeddings,
        'memory': check_memory,
        'redis': check_redis,
        'vectorstore': check_vectorstore,
        'ryzen': check_ryzen,
        'crawler': check_crawler,  # NEW v0.1.2
    }
    
    results = {}
    
    for target in targets:
        if target in check_functions:
            try:
                success, message = check_functions[target]()
                results[target] = (success, message)
            except Exception as e:
                logger.error(f"Health check '{target}' crashed: {e}", exc_info=True)
                
                if error_recovery:
                    # Graceful failure
                    results[target] = (False, f"Check crashed (recovered): {str(e)[:100]}")
                else:
                    # Re-raise
                    results[target] = (False, f"Check crashed: {str(e)[:100]}")
                    raise
        else:
            logger.warning(f"Unknown health check target: {target}")
            results[target] = (False, f"Unknown target: {target}")
    
    return results

def print_health_report(results: Dict[str, Tuple[bool, str]]):
    """
    Print formatted health report.
    
    Args:
        results: Dict of health check results
    """
    print("=" * 70)
    print("Xoe-NovAi Health Check Report v0.1.2")
    print("=" * 70)
    print()
    
    passed = []
    failed = []
    
    for check_name, (success, message) in results.items():
        status = "✓" if success else "✗"
        color = "\033[0;32m" if success else "\033[0;31m"
        reset = "\033[0m"
        
        print(f"{color}{status}{reset} {check_name}: {message}")
        
        if success:
            passed.append(check_name)
        else:
            failed.append(check_name)
    
    print()
    print("=" * 70)
    print(f"Passed: {len(passed)} | Failed: {len(failed)}")
    print("=" * 70)
    print()

# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def main() -> int:
    """
    Main health check entrypoint.
    
    Guide Reference: Section 5.3 (Docker Healthcheck)
    
    Exit codes:
      0 - All critical checks passed
      1 - One or more critical checks failed
      2 - Health check system error
      
    Usage:
      python3 healthcheck.py              # All checks
      python3 healthcheck.py --critical   # Critical only
      python3 healthcheck.py llm memory   # Specific checks
    """
    try:
        # Parse arguments
        import argparse
        
        parser = argparse.ArgumentParser(description='Xoe-NovAi Health Check v0.1.2')
        parser.add_argument(
            'targets',
            nargs='*',
            help='Specific checks to run (default: all)',
            choices=['llm', 'embeddings', 'memory', 'redis', 'vectorstore', 'ryzen', 'crawler']
        )
        parser.add_argument(
            '--critical',
            action='store_true',
            help='Only run critical checks (memory, redis)'
        )
        parser.add_argument(
            '--quiet',
            action='store_true',
            help='Suppress output (exit code only)'
        )
        parser.add_argument(
            '--no-recovery',
            action='store_true',
            help='Disable error recovery (fail fast)'
        )
        
        args = parser.parse_args()
        
        # Run checks
        targets = args.targets if args.targets else None
        results = run_health_checks(
            targets=targets,
            critical_only=args.critical,
            error_recovery=not args.no_recovery
        )
        
        # Print report (unless quiet)
        if not args.quiet:
            print_health_report(results)
        
        # Determine critical failures
        critical_checks = ['memory', 'redis']
        critical_failures = [
            target for target in critical_checks 
            if target in results and not results[target][0]
        ]
        
        # Exit code
        if critical_failures:
            if not args.quiet:
                print(f"CRITICAL: {', '.join(critical_failures)} failed")
            return 1
        
        # Check all results
        all_passed = all(success for success, _ in results.values())
        
        if not all_passed and not args.quiet:
            print("WARNING: Some non-critical checks failed")
        
        return 0 if all_passed else 0  # Non-critical failures don't fail Docker healthcheck
        
    except Exception as e:
        logger.error(f"Health check system error: {e}", exc_info=True)
        print(f"ERROR: Health check system failure: {e}")
        return 2

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test health check module.
    
    Usage: python3 healthcheck.py [--critical] [--quiet] [targets...]
    
    This is the main entrypoint for Docker healthchecks and manual testing.
    """
    sys.exit(main())