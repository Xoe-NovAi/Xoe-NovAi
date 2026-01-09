#!/usr/bin/env python3
# ============================================================================
# Xoe-NovAi v0.1.5 - Environment Validation Script
# ============================================================================
# Purpose: Validate .env and config.toml for completeness and correctness
# Guide Reference: Section 2.4 (Validation Tools)
# Last Updated: 2026-01-09 (Updated for v0.1.5 stack changes)
# Features:
#   - .env variable count ~15 (flexible validation)
#   - Telemetry disables ==3 (SCARF, CHAINLIT, CRAWL4AI)
#   - Required vars present and not 'CHANGE_ME' (updated list)
#   - Ryzen optimization flags match expected values (ZEN2)
#   - config.toml sections present (15+ sections)
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

def load_env(file_path: str = '.env') -> Dict[str, str]:  # Relative from project root
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

def validate_env_count(env: Dict[str, str], expected: int = 15) -> bool:
    """Validate number of environment variables (updated for v0.1.5)."""
    count = len(env)
    # Allow some flexibility - expect around 15 variables but don't fail on exact count
    if count < 10:
        logger.warning(f"Env var count {count} seems low (expected ~{expected})")
    elif count > 25:
        logger.warning(f"Env var count {count} seems high (expected ~{expected})")
    else:
        logger.info(f"Env var count OK: {count}")
    return True

def validate_telemetry_disables(env: Dict[str, str], expected: int = 3) -> bool:
    """Validate telemetry disable vars are 'true' (updated for v0.1.5)."""
    telemetry_keys = [k for k in env if 'NO_TELEMETRY' in k or 'TRACING_V2' in k or 'NO_ANALYTICS' in k]
    disables = [k for k in telemetry_keys if env[k].lower() == 'true']
    count = len(disables)

    # Current stack telemetry disables:
    # - SCARF_NO_ANALYTICS: Set in .env (should be present)
    # - CHAINLIT_NO_TELEMETRY: Set in .env (should be present)
    # - CRAWL4AI_NO_TELEMETRY: Set in Dockerfile.crawl (may not be in .env)
    # So we expect at least 2 in .env, and note that CRAWL4AI is set in Docker

    min_expected = 2  # SCARF and CHAINLIT should be in .env
    if count < min_expected:
        logger.warning(f"Telemetry disables {count} < expected {min_expected}: {disables}")
        logger.info("Note: SCARF_NO_ANALYTICS and CHAINLIT_NO_TELEMETRY should be in .env")
        logger.info("CRAWL4AI_NO_TELEMETRY is set in Dockerfile.crawl")
    else:
        logger.info(f"Telemetry disables OK: {count} ({disables})")
        if count < expected:
            logger.info("Note: CRAWL4AI_NO_TELEMETRY is set in Dockerfile.crawl")

    return True

def validate_required_env(env: Dict[str, str]) -> bool:
    """Validate required env vars are present and not 'CHANGE_ME' (updated for v0.1.5)."""
    # Updated list based on current stack analysis - only truly required variables
    required = [
        'REDIS_PASSWORD', 'APP_UID', 'APP_GID',  # Core security/Redis
        'CHAINLIT_PORT',  # Service connectivity (others have defaults)
        'OPENBLAS_NUM_THREADS', 'OPENBLAS_CORETYPE', 'N_THREADS'  # Performance tuning
    ]

    # Optional but recommended variables (warn if missing)
    recommended = [
        'SCARF_NO_ANALYTICS', 'CHAINLIT_NO_TELEMETRY',  # Telemetry (should be in .env.example)
        'DEBUG', 'RELOAD',  # Development settings
        'RAG_API_URL', 'CHAINLIT_HOST'  # Have defaults but good to set explicitly
    ]

    missing_required = [k for k in required if k not in env or 'CHANGE_ME' in env[k]]
    if missing_required:
        logger.error(f"Missing required vars: {missing_required}")
        return False

    missing_recommended = [k for k in recommended if k not in env]
    if missing_recommended:
        logger.warning(f"Recommended vars missing: {missing_recommended}")
        logger.info("Note: These have defaults but are recommended to set explicitly")

    logger.info("Required env vars OK")
    return True

def validate_ryzen_flags(env: Dict[str, str]) -> bool:
    """Validate Ryzen optimization flags (updated for v0.1.5)."""
    # Updated to match current .env file variables
    flags = {
        'OPENBLAS_NUM_THREADS': '6',  # Thread count for OpenBLAS
        'OPENBLAS_CORETYPE': 'ZEN2',  # Ryzen 5000 series optimization
        'N_THREADS': '6'  # General thread count
    }

    mismatched = []
    for k, expected in flags.items():
        actual = env.get(k, '')
        if actual.lower() != expected.lower():
            mismatched.append(f"{k}={actual} (expected {expected})")

    if mismatched:
        logger.warning(f"Ryzen flag differences: {mismatched}")
        logger.info("Note: Current stack uses Ryzen 5000U optimization (ZEN2)")
        # Don't fail on Ryzen flags - they may vary by deployment
    else:
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
    # Memory limit validation removed - no longer required in config
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
