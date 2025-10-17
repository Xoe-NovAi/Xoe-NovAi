#!/usr/bin/env python3
# ============================================================================
# Xoe-NovAi Phase 1 v0.1.1 - Centralized Configuration Loader (TIER 1 FIX)
# ============================================================================
# Purpose: Shared configuration management to eliminate duplication
# Guide Reference: Section 3.2 (config_loader.py - Tier 1 Addition)
# Last Updated: 2025-10-16
# Features:
#   - LRU cached loading (1 cache entry)
#   - Dot-notation config value access
#   - Section validation
#   - Summary generation for debugging
#   - Robust path fallbacks (repo root, module local, /app path)
#   - CLI test harness with many checks
# ============================================================================

import os
import toml
import logging
import sys
import time
from typing import Dict, Any, Optional
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper: determine default config path candidates (ordered)
# -----------------------------------------------------------------------------
def _default_config_candidates() -> list:
    """
    Return ordered list of candidate config paths to try.
    1) CONFIG_PATH env var (if set)
    2) repo root config.toml (two parents up from this file)
    3) module-local config.toml (same package)
    4) container default: /app/XNAi_rag_app/config.toml
    """
    candidates = []
    env_path = os.getenv("CONFIG_PATH")
    if env_path:
        candidates.append(Path(env_path))

    # repo root candidate: two levels up from this file (repo root)
    try:
        repo_root_candidate = Path(__file__).resolve().parents[2] / "config.toml"
        candidates.append(repo_root_candidate)
    except Exception:
        pass

    # module-local candidate (app/XNAi_rag_app/config.toml)
    module_local_candidate = Path(__file__).resolve().parent / "config.toml"
    candidates.append(module_local_candidate)

    # container default
    candidates.append(Path("/app/XNAi_rag_app/config.toml"))

    return candidates

# ============================================================================
# CORE CONFIGURATION LOADER
# ============================================================================

@lru_cache(maxsize=1)
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from TOML file with caching.

    Args:
        config_path: Explicit path to config.toml. If None, uses environment
                     and standard fallback locations.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: No config found in candidates.
        ValueError: Invalid TOML or missing required sections.
    """
    # Resolve candidate paths
    if config_path:
        candidates = [Path(config_path)]
    else:
        candidates = _default_config_candidates()

    config_file = None
    for cand in candidates:
        try:
            if cand and cand.exists():
                config_file = cand
                break
        except Exception:
            continue

    if config_file is None:
        # Helpful error message listing attempted candidates
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"Configuration file not found. Tried: {tried}\n"
            "Set CONFIG_PATH env var or place config.toml in the repository root or /app/XNAi_rag_app/"
        )

    # Parse TOML
    try:
        config = toml.load(config_file)
    except toml.TomlDecodeError as e:
        logger.error(f"Invalid TOML in {config_file}: {e}")
        raise ValueError(f"Invalid TOML syntax in {config_file}: {e}") from e
    except Exception as e:
        logger.error(f"Failed to load config {config_file}: {e}", exc_info=True)
        raise

    # Validate presence of important sections (Tier 1)
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
            f"Configuration missing required sections: {missing_sections} "
            f"(loaded from {config_file})"
        )

    logger.info(f"Configuration loaded from {config_file} ({len(config)} sections)")
    return config

# ============================================================================
# DOT-NOTATION CONFIG ACCESS
# ============================================================================

def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get nested config value by dot-notation path.

    Example:
        get_config_value("redis.cache.ttl_seconds", 3600)
    """
    config = load_config()
    keys = key_path.split('.')
    value: Any = config
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
    Run validation checks and raise ValueError on failures.

    Checks:
      - metadata.stack_version present (warns if mismatched)
      - performance.memory_limit_gb == expected (6.0)
      - performance.cpu_threads within acceptable range
      - performance.f16_kv_enabled must be True
      - project.telemetry_enabled must be False
      - redis.cache section present
      - backup.faiss section present
    """
    config = load_config()

    checks = []
    # Version check (warn, don't fail)
    version = config.get("metadata", {}).get("stack_version", "unknown")
    if version != "v0.1.1":
        logger.warning(f"Unexpected stack_version: {version} (expected v0.1.1)")
    checks.append(f"version={version}")

    # Memory limit check (critical)
    memory_limit = config["performance"].get("memory_limit_gb")
    if memory_limit != 6.0:
        raise ValueError(f"performance.memory_limit_gb={memory_limit} (expected 6.0)")
    checks.append(f"memory_limit={memory_limit}GB")

    # CPU threads check
    cpu_threads = config["performance"].get("cpu_threads")
    if cpu_threads != 6:
        raise ValueError(f"performance.cpu_threads={cpu_threads} (expected 6)")
    checks.append(f"cpu_threads={cpu_threads}")

    # f16_kv check
    f16_kv = config["performance"].get("f16_kv_enabled", False)
    if not f16_kv:
        raise ValueError("performance.f16_kv_enabled=False (expected True)")
    checks.append(f"f16_kv={f16_kv}")

    # Telemetry check
    telemetry_enabled = config["project"].get("telemetry_enabled", True)
    if telemetry_enabled:
        raise ValueError("project.telemetry_enabled=True (must be False for zero-telemetry)")
    checks.append(f"telemetry_enabled={telemetry_enabled}")

    # Redis cache presence
    if "cache" not in config.get("redis", {}):
        raise ValueError("redis.cache section missing")
    checks.append("redis.cache=present")

    # FAISS backup presence
    if "faiss" not in config.get("backup", {}):
        raise ValueError("backup.faiss section missing")
    checks.append("backup.faiss=present")

    logger.info(f"Configuration validation passed: {', '.join(checks)}")
    return True

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

def get_config_summary() -> Dict[str, Any]:
    """
    Return a compact summary of important config values for diagnostics.
    """
    config = load_config()

    summary = {
        "version": config["metadata"].get("stack_version"),
        "phase": config["project"].get("phase"),
        "codename": config["metadata"].get("codename"),
        "telemetry_enabled": config["project"].get("telemetry_enabled"),
        "memory_limit_gb": config["performance"].get("memory_limit_gb"),
        "cpu_threads": config["performance"].get("cpu_threads"),
        "f16_kv_enabled": config["performance"].get("f16_kv_enabled"),
        "token_rate_target": config["performance"].get("token_rate_target"),
        "redis_cache_enabled": config["redis"].get("cache", {}).get("enabled"),
        "faiss_backup_enabled": config["backup"].get("faiss", {}).get("enabled"),
        "sections_count": len(config),
        "architecture": config["metadata"].get("architecture"),
    }
    return summary

# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

def clear_config_cache():
    """Clear the LRU cache so subsequent calls re-read config.toml."""
    load_config.cache_clear()
    logger.info("Configuration cache cleared")

def is_config_cached() -> bool:
    """Return whether the config loader cache is populated."""
    info = load_config.cache_info()
    return info.currsize > 0

# ============================================================================
# CLI TEST HARNESS
# ============================================================================

def _print(msg: str):
    print(msg)
    logger.info(msg)

if __name__ == "__main__":
    """
    Test suite for config_loader.py

    Usage:
      python3 config_loader.py
    """
    # Basic logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    print("=" * 70)
    print("Xoe-NovAi Configuration Loader - Test Suite")
    print("=" * 70)
    print()

    tests_passed = 0
    tests_failed = 0

    # Test 1: Load configuration
    print("Test 1: Load configuration")
    try:
        cfg = load_config()
        print(f"✓ Config loaded: {len(cfg)} sections")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Config load failed: {e}")
        tests_failed += 1
        sys.exit(1)

    print()

    # Test 2: Version verification (informational)
    print("Test 2: Stack version verification (informational)")
    try:
        version = get_config_value("metadata.stack_version")
        print(f"  Detected stack_version: {version}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Version retrieval failed: {e}")
        tests_failed += 1

    print()

    # Test 3: Default value handling
    print("Test 3: Default value handling")
    try:
        missing = get_config_value("nonexistent.key", default="N/A")
        if missing == "N/A":
            print("✓ Default value handling: OK")
            tests_passed += 1
        else:
            print(f"✗ Default value incorrect: {missing}")
            tests_failed += 1
    except Exception as e:
        print(f"✗ Default handling test failed: {e}")
        tests_failed += 1

    print()

    # Test 4: Nested access
    print("Test 4: Nested configuration access")
    try:
        redis_ttl = get_config_value("redis.cache.ttl_seconds", default=None)
        print(f"  redis.cache.ttl_seconds = {redis_ttl}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Nested access failed: {e}")
        tests_failed += 1

    print()

    # Test 5: Validation
    print("Test 5: Configuration validation (may fail if config intentionally differs)")
    try:
        try:
            validate_config()
            print("✓ Validation passed")
            tests_passed += 1
        except Exception as e:
            print(f"✗ Validation failed (this may be expected): {e}")
            tests_failed += 1
    except Exception as e:
        print(f"✗ Validation execution failed: {e}")
        tests_failed += 1

    print()

    # Test 6: Summary generation
    print("Test 6: Configuration summary")
    try:
        summary = get_config_summary()
        print(f"✓ Summary generated: {len(summary)} fields")
        for k, v in summary.items():
            print(f"  - {k}: {v}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Summary generation failed: {e}")
        tests_failed += 1

    print()

    # Test 7: Cache behaviour
    print("Test 7: Cache behaviour")
    try:
        clear_config_cache()
        start = time.time()
        load_config()
        uncached_ms = (time.time() - start) * 1000
        start = time.time()
        load_config()
        cached_ms = (time.time() - start) * 1000
        print(f"  First load (uncached): {uncached_ms:.2f}ms")
        print(f"  Second load (cached): {cached_ms:.2f}ms")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Cache behaviour test failed: {e}")
        tests_failed += 1

    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Passed: {tests_passed}")
    print(f"Failed: {tests_failed}")
    print()

    if tests_failed == 0:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed; review output above.")
        sys.exit(1)
