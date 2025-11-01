#!/usr/bin/env python3
"""
dependency_tracker.py - Track and analyze Python package dependencies

This module provides tools for:
1. Tracking which packages are downloaded
2. Recording which files request each package
3. Analyzing version conflicts
4. Generating dependency graphs
5. Creating build reports

Usage:
    ./dependency_tracker.py analyze-deps
    ./dependency_tracker.py generate-report
    ./dependency_tracker.py check-conflicts
"""

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
from graphviz import Digraph

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
            with open(self.dep_db_path) as f:
                data = json.load(f)
                for pkg, info in data.items():
                    info['requesters'] = set(info['requesters'])
                    self.dependencies[pkg] = DependencyInfo(**info)
    
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

def main():
    """CLI entrypoint."""
    if len(sys.argv) < 2:
        print("Usage: dependency_tracker.py <command>")
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

if __name__ == '__main__':
    main()