#!/usr/bin/env python3
"""
============================================================================
Xoe-NovAi Phase 1 v0.1.2 - Preflight Checks
============================================================================
Purpose: Validate environment and security settings before deployment
Guide Reference: Section 6 (Security)
Last Updated: 2025-10-28
============================================================================
"""

import os
import sys
import pwd
import grp
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment_vars() -> Tuple[bool, List[str]]:
    """Validate required environment variables."""
    required_vars = {
        'REDIS_PASSWORD': 'Redis password must be set',
        'PHASE2_QDRANT_ENABLED': 'Phase 2 feature flag must be set (true/false)',
    }
    
    missing = []
    for var, message in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing.append(f"{var}: {message}")
        elif var == 'PHASE2_QDRANT_ENABLED' and value.lower() not in ('true', 'false'):
            missing.append(f"{var}: Must be 'true' or 'false', got '{value}'")
            
    return len(missing) == 0, missing

def check_directory_permissions() -> Tuple[bool, List[str]]:
    """Check directory permissions and ownership."""
    dirs_to_check = {
        'logs': 0o750,
        'data/faiss_index': 0o750,
        'library': 0o750,
        'knowledge': 0o750,
    }
    
    issues = []
    for dir_path, expected_mode in dirs_to_check.items():
        path = Path(dir_path)
        if not path.exists():
            issues.append(f"Directory {dir_path} does not exist")
            continue
            
        try:
            stat = path.stat()
            mode = stat.st_mode & 0o777
            if mode != expected_mode:
                issues.append(
                    f"Directory {dir_path} has incorrect permissions: "
                    f"{oct(mode)[2:]} (expected {oct(expected_mode)[2:]})"
                )
                
            # Check ownership (UID 1001 for non-root)
            if stat.st_uid != 1001:
                issues.append(
                    f"Directory {dir_path} has incorrect owner: "
                    f"{stat.st_uid} (expected 1001)"
                )
        except Exception as e:
            issues.append(f"Error checking {dir_path}: {e}")
            
    return len(issues) == 0, issues

def check_docker_security() -> Tuple[bool, List[str]]:
    """Validate Docker security settings."""
    import yaml
    
    issues = []
    try:
        with open('docker-compose.yml', 'r') as f:
            config = yaml.safe_load(f)
            
        services = config.get('services', {})
        for service_name, service in services.items():
            # Check security_opt
            security_opt = service.get('security_opt', [])
            if 'no-new-privileges:true' not in security_opt:
                issues.append(
                    f"Service {service_name} missing "
                    "'security_opt: [no-new-privileges:true]'"
                )
            
            # Check tmpfs
            tmpfs = service.get('tmpfs', [])
            expected_tmpfs = '/tmp:mode=1777,size=512m'
            if expected_tmpfs not in tmpfs:
                issues.append(
                    f"Service {service_name} missing "
                    f"'tmpfs: [{expected_tmpfs}]'"
                )
    except Exception as e:
        issues.append(f"Error checking docker-compose.yml: {e}")
        
    return len(issues) == 0, issues

def main():
    """Run all preflight checks."""
    checks = [
        ('Environment Variables', check_environment_vars),
        ('Directory Permissions', check_directory_permissions),
        ('Docker Security', check_docker_security),
    ]
    
    all_passed = True
    for name, check_fn in checks:
        logger.info(f"\nRunning {name} check...")
        passed, issues = check_fn()
        
        if passed:
            logger.info(f"✅ {name} check passed")
        else:
            all_passed = False
            logger.error(f"❌ {name} check failed:")
            for issue in issues:
                logger.error(f"   - {issue}")
                
    if not all_passed:
        logger.error("\n❌ Some checks failed. Please fix the issues and retry.")
        sys.exit(1)
    else:
        logger.info("\n✅ All preflight checks passed!")

if __name__ == '__main__':
    main()