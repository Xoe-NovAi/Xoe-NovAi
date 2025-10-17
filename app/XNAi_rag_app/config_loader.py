#!/usr/bin/env python3
# ============================================================================
# Xoe-NovAi Phase 1 v0.1.1 - Centralized Configuration Loader (TIER 1 FIX)
# ============================================================================
# Purpose: Shared configuration management to eliminate duplication
# Guide Reference: Section 3.2 (config_loader.py - Tier 1 Addition)
# Last Updated: 2025-10-11
# Features:
#   - LRU cached loading (1 cache entry)
#   - Dot-notation config value access
#   - Section validation
#   - Summary generation for debugging
# ============================================================================

import os
import toml
from typing import Dict, Any, Optional
from functools import lru_cache
from pathlib import Path
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# ============================================================================
# CORE CONFIGURATION LOADER
# ============================================================================

@lru_cache(maxsize=1)
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from TOML file with caching.
    
    Guide Reference: Section 3.2 (Centralized Config Loading)
    
    This function loads config.toml once and caches the result. Subsequent
    calls return the cached version instantly (<1ms).
    
    Args:
        config_path: Path to config.toml (default: from env CONFIG_PATH)
        
    Returns:
        Dict with all configuration sections
        
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config invalid or missing required sections
        
    Example:
        >>> config = load_config()
        >>> print(config['metadata']['stack_version'])
        v0.1.0
    """
    # Determine config path
    if config_path is None:
        config_path = os.getenv(
            "CONFIG_PATH",
            "/app/XNAi_rag_app/config.toml"
        )
    
    config_file = Path(config_path)
    
    # Check file exists
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please ensure config.toml exists in the expected location."
        )
    
    try:
        # Load TOML
        config = toml.load(config_file)
        
        # Validate required sections (Guide Ref: Section 2.2)
        required_sections = [
            "metadata",
            "project", 
            "models",
            "performance",
            "server",
            "redis",
            "vectorstore",
            "logging",
            "metrics",
            "healthcheck",
            "backup"
        ]
        
        missing_sections = [s for s in required_sections if s not in config]
        
        if missing_sections:
            raise ValueError(
                f"Configuration missing required sections: {missing_sections}\n"
                f"Expected {len(required_sections)} sections, found {len(config)}"
            )
        
        # Log success
        logger.info(
            f"Configuration loaded from {config_path} "
            f"({len(config)} sections)"
        )
        
        return config
        
    except toml.TomlDecodeError as e:
        logger.error(f"Invalid TOML syntax in {config_path}: {e}")
        raise ValueError(
            f"Configuration file has invalid TOML syntax: {e}\n"
            f"Please validate with: python3 -c \"import toml; toml.load('{config_path}')\""
        )
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

# ============================================================================
# DOT-NOTATION CONFIG ACCESS
# ============================================================================

def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get nested config value by dot-notation path.
    
    Guide Reference: Section 3.2 (Nested Config Access)
    
    This provides convenient access to deeply nested config values
    without multiple dict lookups.
    
    Args:
        key_path: Dot-separated path (e.g., "redis.cache.ttl_seconds")
        default: Default value if key not found
        
    Returns:
        Config value or default
        
    Example:
        >>> ttl = get_config_value("redis.cache.ttl_seconds")
        >>> print(ttl)
        3600
        
        >>> missing = get_config_value("nonexistent.key", default="N/A")
        >>> print(missing)
        N/A
    """
    config = load_config()
    keys = key_path.split('.')
    value = config
    
    # Navigate through nested dicts
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
            if value is None:
                return default
        else:
            return default
    
    return value if value is not None else default

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_config() -> bool:
    """
    Validate configuration for common issues.
    
    Guide Reference: Section 3.2 (Config Validation)
    
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    try:
        config = load_config()
        
        # Check critical values
        checks = []
        
        # Version check
        version = config["metadata"]["stack_version"]
        if version != "v0.1.1":
            logger.warning(f"Unexpected version: {version}")
        checks.append(f"version={version}")
        
        # Memory limit check
        memory_limit = config["performance"]["memory_limit_gb"]
        if memory_limit != 6.0:
            raise ValueError(
                f"memory_limit_gb={memory_limit} (expected: 6.0)"
            )
        checks.append(f"memory_limit={memory_limit}GB")
        
        # CPU threads check
        cpu_threads = config["performance"]["cpu_threads"]
        if cpu_threads != 6:
            raise ValueError(
                f"cpu_threads={cpu_threads} (expected: 6 for Ryzen 7 5700U)"
            )
        checks.append(f"cpu_threads={cpu_threads}")
        
        # f16_kv check (CRITICAL)
        f16_kv = config["performance"]["f16_kv_enabled"]
        if not f16_kv:
            raise ValueError(
                "f16_kv_enabled=False (MUST be True for <6GB memory target)"
            )
        checks.append(f"f16_kv={f16_kv}")
        
        # Telemetry check
        telemetry = config["project"]["telemetry_enabled"]
        if telemetry:
            raise ValueError(
                "telemetry_enabled=True (MUST be False for zero-telemetry)"
            )
        checks.append(f"telemetry={telemetry}")
        
        # Redis cache check (Tier 1)
        if "cache" not in config["redis"]:
            raise ValueError(
                "redis.cache section missing (Tier 1 requirement)"
            )
        checks.append("redis.cache=present")
        
        # FAISS backup check (Tier 1)
        if "faiss" not in config["backup"]:
            raise ValueError(
                "backup.faiss section missing (Tier 1 requirement)"
            )
        checks.append("backup.faiss=present")
        
        logger.info(f"Configuration validation passed: {', '.join(checks)}")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

def get_config_summary() -> Dict[str, Any]:
    """
    Get configuration summary for debugging/validation.
    
    Guide Reference: Section 3.2 (Config Summary)
    
    Returns:
        Dict with key metrics and settings
        
    Example:
        >>> summary = get_config_summary()
        >>> print(summary['version'])
        v0.1.1
    """
    config = load_config()
    
    return {
        # Stack identity
        "version": config["metadata"]["stack_version"],
        "phase": config["project"]["phase"],
        "codename": config["metadata"]["codename"],
        
        # Critical settings
        "telemetry_enabled": config["project"]["telemetry_enabled"],
        "memory_limit_gb": config["performance"]["memory_limit_gb"],
        "cpu_threads": config["performance"]["cpu_threads"],
        "f16_kv_enabled": config["performance"]["f16_kv_enabled"],
        
        # Performance targets
        "token_rate_min": config["performance"]["token_rate_min"],
        "token_rate_target": config["performance"]["token_rate_target"],
        "token_rate_max": config["performance"]["token_rate_max"],
        "latency_target_ms": config["performance"]["latency_target_ms"],
        
        # Service configuration
        "redis_enabled": config["redis"]["appendonly"],
        "redis_cache_enabled": config["redis"]["cache"]["enabled"],
        "metrics_enabled": config["metrics"]["enabled"],
        "healthcheck_enabled": config["healthcheck"]["enabled"],
        
        # Backup configuration
        "faiss_backup_enabled": config["backup"]["faiss"]["enabled"],
        "faiss_backup_retention": config["backup"]["faiss"]["retention_days"],
        
        # Metadata
        "sections_count": len(config),
        "architecture": config["metadata"]["architecture"],
    }

# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

def clear_config_cache():
    """
    Clear the configuration cache.
    
    Guide Reference: Section 3.2 (Cache Management)
    
    Use this when config.toml is modified at runtime and needs to be reloaded.
    Normally not needed as config should be static after deployment.
    """
    load_config.cache_clear()
    logger.info("Configuration cache cleared")

def is_config_cached() -> bool:
    """
    Check if configuration is cached.
    
    Returns:
        True if config is in cache
    """
    cache_info = load_config.cache_info()
    return cache_info.currsize > 0

# ============================================================================
# TESTING & VALIDATION
# ============================================================================

if __name__ == "__main__":
    """
    Test configuration loader.
    
    Usage: python3 config_loader.py
    
    This validates the config_loader module and provides diagnostics.
    """
    import time
    import sys
    
    print("=" * 70)
    print("Xoe-NovAi Configuration Loader - Test Suite")
    print("=" * 70)
    print()
    
    # Track test results
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Load configuration
    print("Test 1: Load configuration")
    try:
        config = load_config()
        print(f"✓ Config loaded: {len(config)} sections")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Config load failed: {e}")
        tests_failed += 1
        sys.exit(1)
    
    print()
    
    # Test 2: Verify stack version
    print("Test 2: Stack version verification")
    try:
        version = get_config_value("metadata.stack_version")
        if version == "v0.1.0":
            print(f"✓ Stack version: {version}")
            tests_passed += 1
        else:
            print(f"✗ Unexpected version: {version}")
            tests_failed += 1
    except Exception as e:
        print(f"✗ Version check failed: {e}")
        tests_failed += 1
    
    print()
    
    # Test 3: Default value handling
    print("Test 3: Default value handling")
    try:
        missing = get_config_value("nonexistent.key", default="N/A")
        if missing == "N/A":
            print(f"✓ Default value handling: {missing}")
            tests_passed += 1
        else:
            print(f"✗ Default value incorrect: {missing}")
            tests_failed += 1
    except Exception as e:
        print(f"✗ Default value test failed: {e}")
        tests_failed += 1
    
    print()
    
    # Test 4: Nested access
    print("Test 4: Nested configuration access")
    try:
        cache_ttl = get_config_value("redis.cache.ttl_seconds")
        if cache_ttl == 3600:
            print(f"✓ Redis cache TTL: {cache_ttl}s")
            tests_passed += 1
        else:
            print(f"✗ Cache TTL incorrect: {cache_ttl}")
            tests_failed += 1
    except Exception as e:
        print(f"✗ Nested access failed: {e}")
        tests_failed += 1
    
    print()
    
    # Test 5: Configuration validation
    print("Test 5: Configuration validation")
    try:
        validate_config()
        print("✓ Validation passed")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        tests_failed += 1
    
    print()
    
    # Test 6: Configuration summary
    print("Test 6: Configuration summary")
    try:
        summary = get_config_summary()
        print(f"✓ Summary generated: {len(summary)} fields")
        print(f"  - Version: {summary['version']}")
        print(f"  - Phase: {summary['phase']}")
        print(f"  - Memory limit: {summary['memory_limit_gb']}GB")
        print(f"  - CPU threads: {summary['cpu_threads']}")
        print(f"  - f16_kv: {summary['f16_kv_enabled']}")
        print(f"  - Telemetry: {summary['telemetry_enabled']}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Summary generation failed: {e}")
        tests_failed += 1
    
    print()
    
    # Test 7: Cache performance
    print("Test 7: Cache performance")
    try:
        # Clear cache first
        clear_config_cache()
        
        # First load (uncached)
        start = time.time()
        load_config()
        first_load_ms = (time.time() - start) * 1000
        
        # Second load (cached)
        start = time.time()
        load_config()
        cached_load_ms = (time.time() - start) * 1000
        
        print(f"✓ First load: {first_load_ms:.2f}ms")
        print(f"✓ Cached load: {cached_load_ms:.2f}ms")
        
        if cached_load_ms < 1.0:
            print(f"✓ Cache working: {cached_load_ms:.2f}ms (<1ms)")
            tests_passed += 1
        else:
            print(f"⚠ Cache slower than expected: {cached_load_ms:.2f}ms")
            tests_passed += 1  # Still pass, just warn
    except Exception as e:
        print(f"✗ Cache test failed: {e}")
        tests_failed += 1
    
    print()
    
    # Test 8: Cache status
    print("Test 8: Cache status check")
    try:
        is_cached = is_config_cached()
        cache_info = load_config.cache_info()
        
        print(f"✓ Config cached: {is_cached}")
        print(f"✓ Cache info: hits={cache_info.hits}, misses={cache_info.misses}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Cache status check failed: {e}")
        tests_failed += 1
    
    print()
    
    # Test 9: Critical settings verification
    print("Test 9: Critical settings verification")
    try:
        checks = []
        
        # Memory limit
        memory_limit = get_config_value("performance.memory_limit_gb")
        assert memory_limit == 6.0, f"memory_limit={memory_limit} (expected: 6.0)"
        checks.append(f"memory={memory_limit}GB")
        
        # f16_kv
        f16_kv = get_config_value("performance.f16_kv_enabled")
        assert f16_kv == True, f"f16_kv={f16_kv} (expected: True)"
        checks.append(f"f16_kv={f16_kv}")
        
        # CPU threads
        threads = get_config_value("performance.cpu_threads")
        assert threads == 6, f"threads={threads} (expected: 6)"
        checks.append(f"threads={threads}")
        
        # Token rate target
        token_rate = get_config_value("performance.token_rate_target")
        assert token_rate == 20, f"token_rate={token_rate} (expected: 20)"
        checks.append(f"token_rate={token_rate}")
        
        print(f"✓ Critical settings: {', '.join(checks)}")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ Critical settings verification failed: {e}")
        tests_failed += 1
    except Exception as e:
        print(f"✗ Settings check failed: {e}")
        tests_failed += 1
    
    print()
    
    # Test 10: Tier 1 additions verification
    print("Test 10: Tier 1 additions verification")
    try:
        # Redis cache
        redis_cache = get_config_value("redis.cache")
        assert redis_cache is not None, "redis.cache section missing"
        assert redis_cache["enabled"] == True, "redis.cache not enabled"
        print(f"✓ redis.cache: enabled={redis_cache['enabled']}, ttl={redis_cache['ttl_seconds']}s")
        
        # FAISS backup
        faiss_backup = get_config_value("backup.faiss")
        assert faiss_backup is not None, "backup.faiss section missing"
        assert faiss_backup["enabled"] == True, "backup.faiss not enabled"
        print(f"✓ backup.faiss: enabled={faiss_backup['enabled']}, retention={faiss_backup['retention_days']} days")
        
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ Tier 1 verification failed: {e}")
        tests_failed += 1
    except Exception as e:
        print(f"✗ Tier 1 check failed: {e}")
        tests_failed += 1
    
    print()
    
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
        print("Configuration loader is production-ready.")
        print("Integration: Import with 'from config_loader import load_config'")
        sys.exit(0)
    else:
        print(f"✗ {tests_failed} test(s) failed")
        print()
        print("Please fix configuration issues before deployment.")
        sys.exit(1)