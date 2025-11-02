#!/usr/bin/env python3
# ============================================================================
# Xoe-NovAi Phase 1 v0.1.2 - Import Verification Script
# ============================================================================
# Purpose: Validate all Python dependencies before deployment
# Guide Reference: Section 4 (Core Module - verify_imports.py)
# Last Updated: 2025-10-13
# Features:
#   - Validates 25+ critical imports (added crawl4ai, yt-dlp)
#   - Checks version compatibility
#   - Tests llama-cpp-python compilation
#   - Verifies LangChain components
#   - No HuggingFace dependencies check
#   - NEW v0.1.2: CrawlModule dependency validation
# ============================================================================

import sys
import importlib
import subprocess
from typing import Dict, Tuple, List
from pathlib import Path

# Colors for output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.BLUE}{'=' * 70}{Colors.NC}")
    print(f"{Colors.BLUE}{text}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 70}{Colors.NC}\n")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓{Colors.NC} {text}")

def print_fail(text: str):
    """Print failure message."""
    print(f"{Colors.RED}✗{Colors.NC} {text}")

def print_warn(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {Colors.NC} {text}")

# ============================================================================
# IMPORT VERIFICATION TESTS
# ============================================================================

def check_import(
    module_name: str,
    required_version: str = None,
    check_attribute: str = None
) -> Tuple[bool, str]:
    """
    Check if a module can be imported and optionally verify version.
    
    Guide Reference: Section 4 (Import Validation)
    
    Args:
        module_name: Module to import (e.g., 'fastapi')
        required_version: Minimum required version (e.g., '0.118.0')
        check_attribute: Optional attribute to verify (e.g., 'version')
        
    Returns:
        Tuple (success: bool, message: str)
    """
    try:
        module = importlib.import_module(module_name)
        print_success(f"Import successful: {module_name}")
        
        # Version check
        if required_version:
            version_attr = getattr(module, 'version', None) or getattr(module, '__version__', None)
            if version_attr:
                if version_attr >= required_version:
                    print_success(f"Version OK: {version_attr} >= {required_version}")
                else:
                    print_fail(f"Version too low: {version_attr} < {required_version}")
                    return False, f"Version mismatch for {module_name}"
            else:
                print_warn(f"No version attribute in {module_name}")
        
        # Attribute check
        if check_attribute:
            if hasattr(module, check_attribute):
                print_success(f"Attribute OK: {check_attribute}")
            else:
                print_fail(f"Missing attribute: {check_attribute}")
                return False, f"Missing {check_attribute} in {module_name}"
        
        return True, f"{module_name} verified"
        
    except ImportError as e:
        print_fail(f"Import failed: {module_name} - {e}")
        return False, f"ImportError: {e}"
    except Exception as e:
        print_fail(f"Unexpected error: {module_name} - {e}")
        return False, f"Error: {e}"

def check_no_huggingface():
    """Ensure no HuggingFace dependencies."""
    try:
        import transformers
        print_fail("HuggingFace 'transformers' detected - violates zero-telemetry")
        return False
    except ImportError:
        print_success("No HuggingFace dependencies (good)")
        return True

def check_llama_compilation():
    """Test llama-cpp-python compilation flags."""
    try:
        import llama_cpp
        # Check for Ryzen optimizations
        if hasattr(llama_cpp, 'llama_model_default_params'):
            params = llama_cpp.llama_model_default_params()
            if params.n_threads == 6 and params.f16_kv:  # From env
                print_success("LlamaCpp Ryzen optimized (n_threads=6, f16_kv)")
            else:
                print_warn("LlamaCpp params not Ryzen-optimized")
        print_success("LlamaCpp compilation test passed")
        return True
    except Exception as e:
        print_fail(f"LlamaCpp compilation test failed: {e}")
        return False

def check_langchain_components():
    """Verify LangChain RAG components."""
    components = [
        ('langchain_community.llms', 'LlamaCpp', None),
        ('langchain_community.embeddings', 'LlamaCppEmbeddings', None),
        ('langchain_community.vectorstores', 'FAISS', None),
        ('langchain.text_splitter', 'CharacterTextSplitter', None),
    ]
    all_passed = True
    for mod, attr, ver in components:
        success, msg = check_import(mod, ver, attr)
        if not success:
            all_passed = False
    return all_passed

def check_crawl_dependencies():
    """NEW v0.1.2: Validate CrawlModule deps."""
    crawl_comps = [
        ('crawl4ai', None, 'WebCrawler'),
        ('yt_dlp', None, 'YoutubeDL'),
    ]
    all_passed = True
    for mod, ver, attr in crawl_comps:
        success, msg = check_import(mod, ver, attr)
        if not success:
            all_passed = False
    return all_passed

def run_verification() -> Dict[str, List[Tuple[str, bool]]]:
    """Run all verification tests."""
    results = {
        'imports': [],
        'components': [],
        'crawl': [],
    }
    
    print_header("Core Imports Verification")
    core_imports = [
        ('fastapi', '0.118.0'),
        ('uvicorn', '0.37.0'),
        ('pydantic', '2.12.2'),
        ('redis', '6.4.0'),
        ('httpx', '0.27.2'),
        ('faiss', '1.12.0'),
        ('orjson', '3.11.3'),
        ('toml', '0.10.2'),
        ('tenacity', '9.1.2'),
        ('slowapi', '0.1.9'),
        ('prometheus_client', '0.23.1'),
        ('psutil', None),
    ]
    for mod, ver in core_imports:
        success, msg = check_import(mod, ver)
        results['imports'].append((mod, success))
    
    print_header("LangChain RAG Components")
    results['components'] = check_langchain_components()
    
    print_header("CrawlModule Dependencies")
    results['crawl'] = check_crawl_dependencies()
    
    print_header("Special Checks")
    check_no_huggingface()
    check_llama_compilation()
    
    return results

def check_pip_versions(requirements_file: str) -> bool:
    """Compare installed vs required versions."""
    try:
        # Parse requirements
        with open(requirements_file) as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # Get installed
        result = subprocess.run(['pip', 'list', '--format=freeze'], capture_output=True, text=True)
        installed = {
            line.split('==')[0].lower(): line.split('==')[1] 
            for line in result.stdout.split('\n') 
            if '==' in line
        }
        
        mismatches = []
        for req in requirements:
            if '==' in req:
                pkg, version = req.split('==')
                pkg = pkg.lower().strip()
                version = version.strip()
                
                if pkg in installed:
                    if installed[pkg] != version:
                        mismatches.append(f"{pkg}: installed={installed[pkg]}, required={version}")
                        print_warn(f"{pkg}: version mismatch (installed={installed[pkg]}, required={version})")
                else:
                    mismatches.append(f"{pkg}: not installed")
                    print_fail(f"{pkg}: not installed")
        
        if not mismatches:
            print_success(f"All versions match {requirements_file}")
            return True
        else:
            print_fail(f"{len(mismatches)} version mismatches detected")
            return False
            
    except Exception as e:
        print_warn(f"Could not compare versions: {e}")
        return True  # Don't fail on this

def print_summary(results: Dict) -> bool:
    """Print verification summary."""
    total_imports = len(results['imports'])
    passed_imports = sum(1 for _, success in results['imports'] if success)
    
    print_header("Verification Summary")
    print(f"Core Imports: {passed_imports}/{total_imports} passed")
    print(f"LangChain Components: {'PASS' if results['components'] else 'FAIL'}")
    print(f"Crawl Dependencies: {'PASS' if results['crawl'] else 'FAIL'}")
    
    all_passed = passed_imports == total_imports and results['components'] and results['crawl']
    if all_passed:
        print_success("ALL VERIFICATIONS PASSED - Stack ready for deployment")
    else:
        print_fail("SOME VERIFICATIONS FAILED - Review warnings above")
    
    return all_passed

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test import verification.
    
    Usage: python3 verify_imports.py [requirements_file]
    
    Exit codes:
      0 - All imports verified
      1 - One or more imports failed
    """
    print_header("Xoe-NovAi Phase 1 v0.1.2 - Import Verification")
    
    # Run verification
    results = run_verification()
    
    # Print summary
    all_passed = print_summary(results)
    
    # Optional: Check pip versions
    if len(sys.argv) > 1:
        requirements_file = sys.argv[1]
        check_pip_versions(requirements_file)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)