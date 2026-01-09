#!/usr/bin/env python3
# ============================================================================
# Xoe-NovAi Phase 1 v0.1.4-stable - Centralized Configuration Loader
# ============================================================================
# Purpose: Shared configuration management to eliminate duplication
# Guide Reference: Section 3.2 (config_loader.py)
# Last Updated: 2025-10-18
# Features:
#   - LRU cached loading (1 cache entry)
#   - Dot-notation config value access
#   - Section validation
#   - Summary generation for debugging
#   - Robust path fallbacks (repo root, module local, /app path, env var)
#   - CLI test harness with comprehensive checks
# ============================================================================

import os
import toml
import logging
import sys
import time
from typing import Dict, Any, Optional
from functools import lru_cache
from pathlib import Path

# CRITICAL FIX: Import path resolution (Pattern 1)
sys.path.insert(0, str(Path(__file__).parent))

# Pydantic for schema validation (NEW v0.1.4)
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC CONFIGURATION SCHEMA (NEW v0.1.4)
# ============================================================================

class MetadataConfig(BaseModel):
    """Metadata section schema."""
    stack_version: str
    release_date: str
    codename: str
    description: Optional[str] = None
    architecture: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow additional fields


class ProjectConfig(BaseModel):
    """Project section schema."""
    name: str
    phase: int
    telemetry_enabled: bool = False  # CRITICAL: Must be False
    privacy_mode: str = "local-only"
    data_sovereignty: bool = True
    multi_agent_coordination: bool = False
    
    class Config:
        extra = "allow"


class ModelsConfig(BaseModel):
    """Models section schema."""
    llm_path: str
    llm_size_gb: float
    llm_quantization: str
    llm_context_window: int = 2048
    embedding_path: str
    embedding_size_mb: float
    embedding_dimensions: int = 384
    embedding_model_name: Optional[str] = None
    embedding_device: str = "cpu"
    
    class Config:
        extra = "allow"


class PerformanceConfig(BaseModel):
    """Performance section schema."""
    token_rate_min: int
    token_rate_target: int
    token_rate_max: int
    memory_limit_bytes: int
    memory_warning_threshold_bytes: int
    memory_critical_threshold_bytes: int
    memory_limit_gb: float = Field(..., ge=4.0, le=32.0)  # Must be 4-32GB
    memory_warning_threshold_gb: float
    memory_critical_threshold_gb: float
    latency_target_ms: int
    latency_warning_ms: int
    cpu_threads: int = 6
    cpu_architecture: Optional[str] = None
    f16_kv_enabled: bool = True  # CRITICAL: Must be True for Ryzen optimization
    per_doc_chars: int = 500
    total_chars: int = 2048
    
    class Config:
        extra = "allow"
    
    @validator('memory_limit_gb')
    def validate_memory(cls, v):
        """Ensure memory limit is reasonable."""
        if v < 5.0:
            raise ValueError('memory_limit_gb must be >= 5.0')
        return v


class ServerConfig(BaseModel):
    """Server section schema."""
    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1024, le=65535)
    workers: int = Field(1, ge=1, le=16)
    timeout_seconds: int = 30
    cors_origins: list = []
    
    class Config:
        extra = "allow"


class RedisConfig(BaseModel):
    """Redis section schema."""
    version: Optional[str] = None
    host: str = "redis"
    port: int = Field(6379, ge=1024, le=65535)
    password: Optional[str] = None
    timeout_seconds: int = 60
    max_connections: int = Field(50, ge=1, le=500)
    
    class Config:
        extra = "allow"


class XnaiConfig(BaseModel):
    """Complete Xoe-NovAi configuration schema."""
    metadata: MetadataConfig
    project: ProjectConfig
    models: ModelsConfig
    performance: PerformanceConfig
    server: ServerConfig
    redis: RedisConfig
    
    class Config:
        extra = "allow"  # Allow additional sections not in schema
    
    @validator('project', pre=False)
    def validate_telemetry(cls, v):
        """Ensure telemetry is disabled."""
        if v.telemetry_enabled:
            raise ValueError('project.telemetry_enabled must be False (privacy-first requirement)')
        return v


# ============================================================================
# HELPER: DETERMINE DEFAULT CONFIG PATH CANDIDATES
# ============================================================================

def _default_config_candidates() -> list:
    """
    Return ordered list of candidate config paths to try.
    
    Priority order:
    1. CONFIG_PATH env var (if set)
    2. Repo root config.toml (two parents up from this file)
    3. Module-local config.toml (same package as this file)
    4. Container default: /app/XNAi_rag_app/config.toml
    
    Returns:
        List of Path objects to check
    """
    candidates = []
    
    # 1. Environment variable (highest priority)
    env_path = os.getenv("CONFIG_PATH")
    if env_path:
        candidates.append(Path(env_path))
    
    # 2. Repo root candidate (two levels up from this file)
    try:
        repo_root_candidate = Path(__file__).resolve().parents[2] / "config.toml"
        candidates.append(repo_root_candidate)
    except Exception:
        pass
    
    # 3. Module-local candidate (app/XNAi_rag_app/config.toml)
    try:
        module_local_candidate = Path(__file__).resolve().parent / "config.toml"
        candidates.append(module_local_candidate)
    except Exception:
        pass
    
    # 4. Container default
    candidates.append(Path("/app/XNAi_rag_app/config.toml"))
    
    return candidates

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
        config_path: Explicit path to config.toml. If None, uses environment
                     and standard fallback locations.
    
    Returns:
        Parsed config dict with all sections
    
    Raises:
        FileNotFoundError: No config found in candidates
        ValueError: Invalid TOML or missing required sections
    
    Example:
        >>> config = load_config()
        >>> print(config['metadata']['stack_version'])
        v0.1.4-stable
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
    
    # Validate presence of important sections
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
    
    # Pydantic validation (NEW v0.1.4) - Blueprint compliant config schema
    try:
        validated_config = XnaiConfig(**config)
        logger.info(f"Configuration validated against Pydantic schema (v0.1.4-compliant)")
    except Exception as e:
        logger.error(f"Configuration schema validation failed: {e}")
        raise ValueError(f"Configuration validation error: {e}") from e
    
    logger.info(f"Configuration loaded from {config_file} ({len(config)} sections)")
    return config

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
    value: Any = config
    
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
    Run validation checks and raise ValueError on failures.
    
    Guide Reference: Section 3.2 (Config Validation)
    
    Checks:
      - metadata.stack_version present (warns if mismatched)
      - performance.memory_limit_gb == expected (6.0)
      - performance.cpu_threads within acceptable range
      - performance.f16_kv_enabled must be True
      - project.telemetry_enabled must be False
      - redis.cache section present
      - backup.faiss section present
    
    Returns:
        True if validation passes
    
    Raises:
        ValueError: If validation fails
    """
    config = load_config()
    
    checks = []
    
    # Version check (warn, don't fail)
    version = config.get("metadata", {}).get("stack_version", "unknown")
    if version not in ["v0.1.1_rev_1.4", "v0.1.2", "v0.1.4-stable", "v0.1.5"]:
        logger.warning(f"Unexpected stack_version: {version} (expected v0.1.4-stable or v0.1.5)")
    checks.append(f"version={version}")
    
    # Memory limit check (critical)
    memory_limit = config["performance"].get("memory_limit_gb")
    if memory_limit not in [5.0, 6.0]:  # Support both 5.0 and 6.0 GB configurations
        raise ValueError(f"performance.memory_limit_gb={memory_limit} (expected 5.0 or 6.0)")
    checks.append(f"memory_limit={memory_limit}GB")
    
    # CPU threads check
    cpu_threads = config["performance"].get("cpu_threads")
    if cpu_threads != 6:
        raise ValueError(f"performance.cpu_threads={cpu_threads} (expected 6)")
    checks.append(f"cpu_threads={cpu_threads}")
    
    # f16_kv check (CRITICAL)
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
    
    Guide Reference: Section 3.2 (Config Summary)
    
    Returns:
        Dict with key metrics and settings
    
    Example:
        >>> summary = get_config_summary()
        >>> print(summary['version'])
        v0.1.4-stable
    """
    config = load_config()
    
    summary = {
        # Stack identity
        "version": config["metadata"].get("stack_version"),
        "phase": config["project"].get("phase"),
        "codename": config["metadata"].get("codename"),
        "architecture": config["metadata"].get("architecture"),
        
        # Critical settings
        "telemetry_enabled": config["project"].get("telemetry_enabled"),
        "memory_limit_gb": config["performance"].get("memory_limit_gb"),
        "cpu_threads": config["performance"].get("cpu_threads"),
        "f16_kv_enabled": config["performance"].get("f16_kv_enabled"),
        
        # Performance targets
        "token_rate_target": config["performance"].get("token_rate_target"),
        "latency_target_ms": config["performance"].get("latency_target_ms"),
        
        # Service configuration
        "redis_cache_enabled": config["redis"].get("cache", {}).get("enabled"),
        "faiss_backup_enabled": config["backup"].get("faiss", {}).get("enabled"),
        "metrics_enabled": config["metrics"].get("enabled"),
        
        # Counts
        "sections_count": len(config),
    }
    return summary

# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

def clear_config_cache():
    """
    Clear the LRU cache so subsequent calls re-read config.toml.
    
    Guide Reference: Section 3.2 (Cache Management)
    
    Use this when config.toml is modified at runtime and needs to be reloaded.
    Normally not needed as config should be static after deployment.
    """
    load_config.cache_clear()
    logger.info("Configuration cache cleared")

def is_config_cached() -> bool:
    """
    Return whether the config loader cache is populated.
    
    Returns:
        True if config is in cache
    """
    info = load_config.cache_info()
    return info.currsize > 0

# ============================================================================
# CLI TEST HARNESS
# ============================================================================

def _print(msg: str):
    """Helper to print and log simultaneously."""
    print(msg)
    logger.info(msg)

if __name__ == "__main__":
    """
    Test suite for config_loader.py
    
    Usage:
      python3 config_loader.py
    
    This validates the config_loader module and provides diagnostics.
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
        validate_config()
        print("✓ Validation passed")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Validation failed (this may be expected): {e}")
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
        
        if cached_ms < 1.0:
            print(f"✓ Cache working: {cached_ms:.2f}ms (<1ms)")
        else:
            print(f"⚠  Cache slower than expected: {cached_ms:.2f}ms")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Cache behaviour test failed: {e}")
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
        assert memory_limit in [5.0, 6.0], f"memory_limit={memory_limit} (expected: 5.0 or 6.0)"
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
    
    # Test 10: Required sections verification
    print("Test 10: Required sections verification")
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
        
        # CrawlModule (new in v0.1.2)
        crawl_config = get_config_value("crawl")
        if crawl_config:
            print(f"✓ crawl: enabled={crawl_config.get('enabled')}, version={crawl_config.get('version')}")
        else:
            print("⚠  crawl section not found (may be older config)")
        
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ Section verification failed: {e}")
        tests_failed += 1
    except Exception as e:
        print(f"✗ Section check failed: {e}")
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