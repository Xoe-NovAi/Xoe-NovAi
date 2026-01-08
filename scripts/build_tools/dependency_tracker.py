#!/usr/bin/env python3
"""
dependency_tracker.py - Track and analyze Python package dependencies

This module provides tools for:
1. Tracking which packages are downloaded
2. Recording which files request each package
3. Analyzing version conflicts
4. Generating dependency graphs (requires graphviz: pip install graphviz)
5. Creating build reports
6. Verifying wheel integrity

Usage:
    ./dependency_tracker.py analyze-deps
    ./dependency_tracker.py generate-report
    ./dependency_tracker.py check-conflicts
    ./dependency_tracker.py --verify  # Verify wheelhouse integrity
"""

import hashlib
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import toml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('build_tools.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('dependency_tracker')

@dataclass
class DependencyInfo:
    """Information about a package dependency."""
    name: str
    version: str
    requesters: Set[str]
    downloaded_at: str
    source: str
    wheel_path: Optional[str] = None
    build_flags: Optional[Dict[str, str]] = None

class DependencyTracker:
    """Track and analyze Python package dependencies."""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.dep_db_path = workspace_root / 'scripts' / 'build_tools' / 'dependency_db.json'
        self.dependencies: Dict[str, DependencyInfo] = {}
        self._load_database()
    
    def _load_database(self):
        """Load existing dependency database."""
        if self.dep_db_path.exists():
            try:
                with open(self.dep_db_path) as f:
                    content = f.read().strip()
                    if not content:
                        # Empty file - initialize empty database
                        return
                    data = json.loads(content)
                    for pkg, info in data.items():
                        info['requesters'] = set(info['requesters'])
                        self.dependencies[pkg] = DependencyInfo(**info)
            except (json.JSONDecodeError, ValueError) as e:
                # Corrupted file - log warning and start fresh
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Corrupted dependency database at {self.dep_db_path}: {e}. Starting fresh.")
                # Backup corrupted file
                backup_path = self.dep_db_path.with_suffix('.json.bak')
                if self.dep_db_path.exists():
                    import shutil
                    shutil.move(str(self.dep_db_path), str(backup_path))
    
    def _save_database(self):
        """Save dependency database to disk."""
        self.dep_db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            pkg: {**asdict(info), 'requesters': list(info.requesters)}
            for pkg, info in self.dependencies.items()
        }
        with open(self.dep_db_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def record_dependency(self, package: str, version: str, requester: str,
                         source: str = 'requirements.txt', wheel_path: Optional[str] = None,
                         build_flags: Optional[Dict[str, str]] = None):
        """Record information about a package dependency."""
        now = datetime.now().isoformat()
        
        if package not in self.dependencies:
            self.dependencies[package] = DependencyInfo(
                name=package,
                version=version,
                requesters={requester},
                downloaded_at=now,
                source=source,
                wheel_path=wheel_path,
                build_flags=build_flags
            )
        else:
            dep = self.dependencies[package]
            dep.requesters.add(requester)
            
            # Version conflict detection
            if dep.version != version:
                logger.warning(
                    f"Version conflict for {package}: "
                    f"{dep.version} (existing) vs {version} (new) "
                    f"requested by {requester}"
                )
        
        self._save_database()
    
    def analyze_conflicts(self) -> List[Dict]:
        """Find version conflicts in dependencies."""
        conflicts = []
        pkg_versions = defaultdict(set)
        
        for pkg, info in self.dependencies.items():
            pkg_versions[info.name].add(info.version)
        
        for pkg, versions in pkg_versions.items():
            if len(versions) > 1:
                conflicts.append({
                    'package': pkg,
                    'versions': list(versions),
                    'requesters': list(self.dependencies[pkg].requesters)
                })
        
        return conflicts
    
    def generate_graph(self, output_path: str = 'dependency_graph.pdf'):
        """Generate a graphviz visualization of dependencies."""
        try:
            from graphviz import Digraph
        except ImportError:
            logger.warning("graphviz not installed. Skipping graph generation. Install with: pip install graphviz")
            return
        
        dot = Digraph(comment='Package Dependencies')
        dot.attr(rankdir='LR')
        
        # Add nodes for requirements files
        requesters = set()
        for dep in self.dependencies.values():
            requesters.update(dep.requesters)
        
        for req in requesters:
            dot.node(req, req, shape='box')
        
        # Add nodes and edges for packages
        for pkg, info in self.dependencies.items():
            dot.node(pkg, f"{pkg}\n{info.version}", shape='ellipse')
            for req in info.requesters:
                dot.edge(req, pkg)
        
        dot.render(output_path, cleanup=True)
    
    def generate_report(self) -> str:
        """Generate a markdown report of dependency status."""
        lines = [
            "# Dependency Analysis Report",
            f"\nGenerated: {datetime.now().isoformat()}",
            "\n## Package Summary\n"
        ]
        
        # Package statistics
        total_pkgs = len(self.dependencies)
        requesters = set()
        for dep in self.dependencies.values():
            requesters.update(dep.requesters)
        
        lines.extend([
            f"- Total Packages: {total_pkgs}",
            f"- Unique Requesters: {len(requesters)}",
            "\n## Version Conflicts\n"
        ])
        
        # List conflicts
        conflicts = self.analyze_conflicts()
        if conflicts:
            for conflict in conflicts:
                lines.append(f"### {conflict['package']}")
                lines.append("\nVersions:")
                for version in conflict['versions']:
                    lines.append(f"- {version}")
                lines.append("\nRequesters:")
                for requester in conflict['requesters']:
                    lines.append(f"- {requester}")
                lines.append("")
        else:
            lines.append("No version conflicts found.")
        
        return "\n".join(lines)

# New: SHA256 integrity verification functions
def compute_sha256(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for block in iter(lambda: f.read(4096), b''):
            sha256.update(block)
    return sha256.hexdigest()

def generate_manifest(wheelhouse_dir: str) -> Dict[str, str]:
    """Generate SHA256 manifest for all wheels in directory."""
    manifest = {}
    path = Path(wheelhouse_dir)
    for whl in path.glob('*.whl'):
        manifest[whl.name] = compute_sha256(str(whl))
    manifest_path = path / 'wheelhouse_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    logger.info(f"Generated manifest at {manifest_path}")
    return manifest

def verify_manifest(wheelhouse_dir: str) -> bool:
    """Verify wheels against manifest."""
    path = Path(wheelhouse_dir)
    manifest_path = path / 'wheelhouse_manifest.json'
    
    if not manifest_path.exists():
        logger.info("Manifest not found - generating new one")
        generate_manifest(wheelhouse_dir)
        return True
    
    with open(manifest_path) as f:
        data = json.load(f)
    
    # Check if this is a wheel manifest (dict with .whl filenames as keys)
    # or a download log (dict with 'downloads', 'errors', 'skipped' keys)
    if isinstance(data, dict):
        # Check if it's a download log format
        if 'downloads' in data or 'errors' in data or 'skipped' in data:
            logger.info("Found download log instead of wheel manifest - generating new manifest")
            generate_manifest(wheelhouse_dir)
            return True
        
        # It's a wheel manifest - verify it
        expected = data
        failed = False
        for whl_name, exp_hash in expected.items():
            # Skip non-wheel entries
            if not whl_name.endswith('.whl'):
                continue
            whl_path = path / whl_name
            if not whl_path.exists():
                logger.error(f"Missing wheel: {whl_name}")
                failed = True
                continue
            actual_hash = compute_sha256(str(whl_path))
            if actual_hash != exp_hash:
                logger.error(f"Hash mismatch for {whl_name}: expected {exp_hash}, got {actual_hash}")
                failed = True
        
        if failed:
            return False
        logger.info("All wheels verified successfully")
        return True
    else:
        logger.warning("Invalid manifest format - generating new one")
        generate_manifest(wheelhouse_dir)
        return True

def main():
    """CLI entrypoint."""
    if len(sys.argv) < 2:
        print("Usage: dependency_tracker.py <command>")
        print("Commands: analyze-deps, generate-report, check-conflicts, --verify")
        sys.exit(1)
    
    workspace_root = Path(__file__).parent.parent.parent
    tracker = DependencyTracker(workspace_root)
    
    command = sys.argv[1]
    if command == 'analyze-deps':
        conflicts = tracker.analyze_conflicts()
        print(json.dumps(conflicts, indent=2))
    
    elif command == 'generate-report':
        report = tracker.generate_report()
        report_path = workspace_root / 'scripts' / 'build_tools' / 'dependency_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Report written to {report_path}")
    
    elif command == 'check-conflicts':
        conflicts = tracker.analyze_conflicts()
        if conflicts:
            print("Found version conflicts:")
            print(json.dumps(conflicts, indent=2))
            sys.exit(1)
        else:
            print("No version conflicts found.")
    
    elif command == '--verify':
        wheelhouse_dir = str(workspace_root / 'wheelhouse')  # Adjust path if needed
        if len(sys.argv) > 2:
            wheelhouse_dir = sys.argv[2]
        verify_manifest(wheelhouse_dir)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == '__main__':
    main()