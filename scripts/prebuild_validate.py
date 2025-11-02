#!/usr/bin/env python3
"""
Pre-build validation script for Xoe-NovAi
Checks for:
- Missing requirements files
- Missing wheelhouse directory and wheels
- Dependency conflicts using pipdeptree
- Required app files for each service
Logs results to logs/prebuild_validate.log
"""
import os
import sys
import glob
import subprocess
from pathlib import Path

LOGFILE = Path("logs/prebuild_validate.log")
LOGFILE.parent.mkdir(exist_ok=True)

def log(msg):
    print(msg)
    with open(LOGFILE, "a") as f:
        f.write(msg + "\n")

REQUIREMENTS = [
    "requirements-api.txt",
    "requirements-crawl.txt",
    "requirements-chainlit.txt",
    "requirements-curation_worker.txt",
]
WHEELHOUSE = Path("wheelhouse")
APP_FILES = {
    "api": ["app/XNAi_rag_app/main.py", "app/XNAi_rag_app/dependencies.py"],
    "curation_worker": ["app/XNAi_rag_app/curation_worker.py"],
}

log("Pre-build validation started.")

# Check requirements files
for req in REQUIREMENTS:
    if not Path(req).exists():
        log(f"ERROR: Missing requirements file: {req}")
    else:
        log(f"Found requirements file: {req}")

# Check wheelhouse
if not WHEELHOUSE.exists():
    log("ERROR: wheelhouse directory missing!")
else:
    wheels = list(WHEELHOUSE.glob("*.whl"))
    log(f"Found {len(wheels)} wheels in wheelhouse.")
    if len(wheels) == 0:
        log("ERROR: No wheels found in wheelhouse!")

# Check app files
for service, files in APP_FILES.items():
    for f in files:
        if not Path(f).exists():
            log(f"ERROR: Missing required file for {service}: {f}")
        else:
            log(f"Found required file for {service}: {f}")

# Check dependency conflicts
try:
    for req in REQUIREMENTS:
        if Path(req).exists():
            log(f"Checking dependency tree for {req}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "pipdeptree"
            ], capture_output=True)
            result = subprocess.run([
                sys.executable, "-m", "pipdeptree", "-p", req
            ], capture_output=True, text=True)
            log(result.stdout)
except Exception as e:
    log(f"ERROR running pipdeptree: {e}")

log("Pre-build validation complete.")
