#!/usr/bin/env python3
# ============================================================================
# Xoe-NovAi Phase 1 v0.1.2 - Environment Validation Script
# ============================================================================
# Purpose: Validate .env and config.toml for completeness and correctness
# Guide Reference: Section 2.4 (Validation Tools)
# Last Updated: 2025-10-16
# Features:
#   - .env variable count ==197
#   - Telemetry disables ==8 (all 'true')
#   - Required vars present and not 'CHANGE_ME'
#   - Ryzen optimization flags match expected values
#   - config.toml sections present (23 total)
#   - Basic type checks for config.toml values
#   - Exit 1 on failure with error logs
# ============================================================================

import sys
import logging
from typing import Dict, List
import toml  # For config.toml validation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_env(file_path: str = '../.env') -> Dict[str, str]:  # Relative from /app/XNAi_rag_app/
    """Load .env file into dict, ignoring comments and empty lines."""
    env = {}
    try:
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env[key.strip()] = value.strip()
    except FileNotFoundError:
        logger.error(f".env file not found at {file_path}")
        sys.exit(1)
    return env

def load_toml(file_path: str = 'config.toml') -> Dict:
    """Load config.toml into dict."""
    try:
        return toml.load(file_path)
    except FileNotFoundError:
        logger.error(f"config.toml not found at {file_path}")
        sys.exit(1)
    except toml.TomlDecodeError as e:
        logger.error(f"Invalid TOML syntax: {e}")
        sys.exit(1)

def validate_env_count(env: Dict[str, str], expected: int = 197) -> bool:
    """Validate number of environment variables."""
    count = len(env)
    if count != expected:
        logger.error(f"Env var count {count} != expected {expected}")
        return False
    logger.info(f"Env var count OK: {count}")
    return True

def validate_telemetry_disables(env: Dict[str, str], expected: int = 8) -> bool:
    """Validate telemetry disable vars are 'true'."""
    telemetry_keys = [k for k in env if 'NO_TELEMETRY' in k or 'TRACING_V2' in k or 'NO_ANALYTICS' in k]
    disables = [k for k in telemetry_keys if env[k].lower() == 'true']
    count = len(disables)
    if count != expected:
        logger.error(f"Telemetry disables {count} != {expected}: {disables}")
        return False
    logger.info(f"Telemetry disables OK: {count} ({disables})")
    return True

def validate_required_env(env: Dict[str, str]) -> bool:
    """Validate required env vars are present and not 'CHANGE_ME'."""
    required = [
        'REDIS_PASSWORD', 'APP_UID', 'APP_GID', 'MODEL_PATH', 'EMBEDDING_MODEL_PATH',
        'RAG_API_URL', 'API_TIMEOUT_SECONDS', 'CHAINLIT_HOST', 'CHAINLIT_PORT',
        'CRAWL_ALLOWLIST_URLS', 'CRAWL_RATE_LIMIT_PER_MIN', 'BACKUP_ENABLED',
        'LLAMA_CPP_N_THREADS', 'OPENBLAS_CORETYPE', 'MKL_DEBUG_CPU_TYPE', 'LLAMA_CPP_F16_KV'
    ]
    missing = [k for k in required if k not in env or 'CHANGE_ME' in env[k]]
    if missing:
        logger.error(f"Missing or CHANGE_ME vars: {missing}")
        return False
    logger.info("All required env vars OK")
    return True

def validate_ryzen_flags(env: Dict[str, str]) -> bool:
    """Validate Ryzen optimization flags."""
    flags = {
        'LLAMA_CPP_N_THREADS': '6',
        'OPENBLAS_CORETYPE': 'ZEN',
        'MKL_DEBUG_CPU_TYPE': '5',
        'LLAMA_CPP_F16_KV': 'true'
    }
    mismatched = [k for k, v in flags.items() if env.get(k, '').lower() != v.lower()]
    if mismatched:
        logger.error(f"Ryzen flag mismatch: {mismatched}")
        return False
    logger.info("Ryzen flags OK")
    return True

def validate_config_toml(toml_data: Dict) -> bool:
    """Validate config.toml sections and basic values."""
    required_sections = [
        'metadata', 'project', 'models', 'performance', 'server', 'redis',
        'backup', 'logging', 'crawl', 'chainlit', 'vectorstore', 'phase2', 'docker', 'validation', 'debug'
    ]
    missing_sections = [s for s in required_sections if s not in toml_data]
    if missing_sections:
        logger.error(f"Missing config.toml sections: {missing_sections}")
        return False
    # Basic value checks (e.g., telemetry_enabled = false)
    if toml_data.get('project', {}).get('telemetry_enabled', True):
        logger.error("Telemetry enabled in config.toml—must be false")
        return False
    if toml_data.get('performance', {}).get('memory_limit_gb', 0) != 6.0:
        logger.warning("Memory limit in config.toml not 6.0GB—expected for Ryzen")
    logger.info("config.toml sections and key values OK")
    return True

if __name__ == "__main__":
    env = load_env()
    toml_data = load_toml()
    
    checks = [
        validate_env_count(env),
        validate_telemetry_disables(env),
        validate_required_env(env),
        validate_ryzen_flags(env),
        validate_config_toml(toml_data)
    ]
    
    sys.exit(0 if all(checks) else 1)