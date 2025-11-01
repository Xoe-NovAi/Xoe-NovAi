#!/usr/bin/env python3
"""
scan_requirements.py - Initial scan of requirements files to populate dependency database
"""

import os
import sys
from pathlib import Path

# Add build_tools to path
build_tools_dir = Path(__file__).parent
workspace_root = build_tools_dir.parent.parent
sys.path.append(str(build_tools_dir))

from dependency_tracker import DependencyTracker

def parse_requirements(file_path: Path) -> list:
    """Parse a requirements file."""
    deps = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '==' in line:
                    name, version = line.split('==')
                    deps.append((name.strip(), version.strip()))
                else:
                    deps.append((line.strip(), 'latest'))
    return deps

def main():
    """Scan all requirements files and populate dependency database."""
    tracker = DependencyTracker(workspace_root)
    
    # Find all requirements files
    req_files = list(workspace_root.glob('requirements-*.txt'))
    req_files.extend(workspace_root.glob('**/requirements.txt'))
    
    for req_file in req_files:
        print(f"Scanning {req_file.relative_to(workspace_root)}")
        deps = parse_requirements(req_file)
        
        for name, version in deps:
            tracker.record_dependency(
                package=name,
                version=version,
                requester=str(req_file.relative_to(workspace_root)),
                source='requirements.txt'
            )
    
    # Generate initial report
    report = tracker.generate_report()
    report_path = workspace_root / 'scripts' / 'build_tools' / 'initial_dependency_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport written to {report_path}")

if __name__ == '__main__':
    main()