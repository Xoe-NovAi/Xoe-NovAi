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
    
    # Find all requirements files (exclude projects/ and other non-main directories)
    req_files = list(workspace_root.glob('requirements-*.txt'))
    # Only scan requirements.txt in root or app directories, exclude projects/
    for req_file in workspace_root.glob('**/requirements.txt'):
        rel_path = req_file.relative_to(workspace_root)
        # Skip projects/ directory and other non-main locations
        if 'projects/' in str(rel_path) or 'node_modules/' in str(rel_path) or '.venv/' in str(rel_path):
            continue
        req_files.append(req_file)
    
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
    
    # Check for conflicts (only in main requirements files)
    conflicts = tracker.analyze_conflicts()
    if conflicts:
        # Filter out conflicts from projects/ directory
        main_conflicts = [
            c for c in conflicts 
            if not any('projects/' in str(r) for r in c.get('requesters', []))
        ]
        if main_conflicts:
            print("\n⚠ Version conflicts detected in main requirements:")
            for conflict in main_conflicts:
                print(f"  - {conflict['package']}: {conflict['versions']}")
                print(f"    Requesters: {conflict['requesters']}")
            sys.exit(1)
        else:
            print("\n⚠ Version conflicts detected but only in projects/ directory (ignored)")
    
    # Generate initial report
    report = tracker.generate_report()
    report_path = workspace_root / 'scripts' / 'build_tools' / 'initial_dependency_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport written to {report_path}")

if __name__ == '__main__':
    main()