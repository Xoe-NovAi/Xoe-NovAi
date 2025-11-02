#!/usr/bin/env python3
"""
Version Management System for Xoe-NovAi Build Process
Purpose: Validate and update project dependencies while maintaining compatibility
"""

import sys
from pathlib import Path
import toml
import logging
from typing import Dict, List, Set, Tuple
import re

# Optional: packaging for semver (pip install packaging if needed)
try:
    from packaging import version as pkg_version
    HAS_PACKAGING = True
except ImportError:
    HAS_PACKAGING = False
    logging.warning("packaging not installed; using str compare for >= constraints")

# Ensure logs directory exists
Path('versions/logs').mkdir(parents=True, exist_ok=True)

# Configure logging (DEBUG for trace; change to INFO after)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('versions/logs/version_management.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class VersionManager:
    def __init__(self, versions_file: str = "versions/versions.toml"):
        self.versions_file = Path(versions_file)
        self.versions = self._load_versions()
        self.requirements_files = self._find_requirements_files()
        
    def _load_versions(self) -> dict:
        """Load versions from TOML file"""
        try:
            return toml.load(self.versions_file)
        except Exception as e:
            logging.error(f"Failed to load versions file: {e}")
            sys.exit(1)

    def _find_requirements_files(self) -> List[Path]:
        """Find all requirements files in the project"""
        return list(Path().glob("requirements-*.txt"))

    def update_requirements(self):
        """Update all requirements files with latest versions, preserving specifiers"""
        for req_file in self.requirements_files:
            self._update_single_requirements(req_file)

    def _update_single_requirements(self, req_file: Path):
        """Update a single requirements file with current versions"""
        try:
            with open(req_file) as f:
                requirements = f.readlines()

            updated = []
            for req in requirements:
                req_stripped = req.strip()
                if not req_stripped or req_stripped.startswith('#'):
                    updated.append(req)
                    continue
                
                # Regex: pkg[optional_extras][specifier]old_version
                match = re.match(r'^(\S+?)(?:\[[^\]]+\])?([><=~! ]*)([0-9a-zA-Z.-]+.*)?$', req_stripped)
                if match:
                    pkg, specifier, old_version_part = match.groups()
                    specifier = specifier.strip()  # Trim spaces in specifier
                    if pkg in self.versions['versions']:
                        new_version = self.versions['versions'][pkg]
                        old_version = old_version_part.strip() if old_version_part else ''
                        if specifier and old_version and old_version != new_version:
                            new_line = f"{pkg}{specifier}{new_version}\n"
                            updated.append(new_line)
                            logging.info(f"Updated {pkg} in {req_file.name} to {new_version} (preserved specifier: '{specifier}', old: '{old_version}')")
                        elif specifier:
                            # Match, no change
                            updated.append(req)
                            logging.debug(f"No change for {pkg} in {req_file.name} (matches '{new_version}')")
                        else:
                            # Pin if no specifier
                            new_line = f"{pkg}=={new_version}\n"
                            updated.append(new_line)
                            logging.info(f"Pinned {pkg} in {req_file.name} to {new_version}")
                    else:
                        updated.append(req)
                        logging.debug(f"Skipped {pkg} in {req_file.name} (not in versions.toml)")
                else:
                    # Fallback: Assume pin if no match
                    parts = re.split(r'\s*==\s*', req_stripped)
                    if len(parts) == 2:
                        pkg = parts[0].strip()
                        if pkg in self.versions['versions']:
                            new_version = self.versions['versions'][pkg]
                            new_line = f"{pkg}=={new_version}\n"
                            updated.append(new_line)
                            logging.info(f"Pinned {pkg} in {req_file.name} to {new_version} (fallback)")
                        else:
                            updated.append(req)
                            logging.debug(f"Skipped fallback {pkg} in {req_file.name} (not in versions.toml)")
                    else:
                        updated.append(req)
                        logging.debug(f"No match/fallback for line in {req_file.name}: '{req_stripped}'")

            with open(req_file, 'w') as f:
                f.writelines(updated)
            logging.info(f"Updated {req_file}")
        except Exception as e:
            logging.error(f"Failed to update {req_file}: {e}")

    def validate_constraints(self) -> bool:
        """Validate version constraints across all requirements"""
        constraints = self.versions.get('constraints', {})
        violations = []

        for pkg, constraint in constraints.items():
            logging.debug(f"Validating constraint for {pkg}: '{constraint}'")
            if not self._check_constraint(pkg, constraint):
                violations.append(f"{pkg}: {constraint}")

        if violations:
            logging.error("Constraint violations found:")
            for v in violations:
                logging.error(f"  - {v}")
            return False
        return True

    def _check_constraint(self, package: str, constraint: str) -> bool:
        """Check if a package version meets its constraints"""
        # Treat bare constraint as exact match "== {constraint}"
        if not any(op in constraint for op in ['>=', '<=', '==', '!=', '~=', '> ', '< ']):
            constraint = f"=={constraint}"
            logging.debug(f"Treating bare constraint '{constraint}' as exact match for {package}")
        
        # Check if package is present in requirements
        found = False
        for req_file in self.requirements_files:
            logging.debug(f"Checking file: {req_file}")
            with open(req_file) as f:
                for line_num, line in enumerate(f, 1):
                    line_stripped = line.strip()
                    logging.debug(f"Line {line_num} in {req_file}: '{line_stripped}'")
                    if line_stripped.startswith(package):
                        found = True
                        logging.debug(f"Found {package} in {req_file}: '{line_stripped}'")
                        # Parse version from line
                        version_match = re.search(r'([><=~! ]+)([0-9a-zA-Z.-]+)', line_stripped)
                        if version_match:
                            line_spec, line_ver = version_match.groups()
                            line_ver = line_ver.strip()
                            logging.debug(f"Parsed line_ver: '{line_ver}', line_spec: '{line_spec}' for constraint '{constraint}'")
                            # Semver compare with packaging if available
                            if HAS_PACKAGING and line_ver:
                                try:
                                    req_ver = pkg_version.parse(line_ver)
                                    if '>=' in constraint:
                                        const_ver_str = constraint.replace('>=', '').strip()
                                        const_ver = pkg_version.parse(const_ver_str)
                                        if req_ver >= const_ver:
                                            logging.debug(f"Match: {req_ver} >= {const_ver}")
                                            return True
                                    elif '==' in constraint:
                                        const_ver_str = constraint.replace('==', '').strip()
                                        if line_ver == const_ver_str:
                                            logging.debug(f"Match: '{line_ver}' == '{const_ver_str}'")
                                            return True
                                    # Add other specifiers
                                except ValueError as ve:
                                    logging.debug(f"Packaging parse error: {ve}, falling back to str")
                            # Fallback str compare
                            if line_ver and '>=' in constraint and line_ver >= constraint.replace('>=', '').strip():
                                logging.debug(f"Str match: '{line_ver}' >= '{constraint.replace('>=', '').strip()}'")
                                return True
                            if line_ver and '==' in constraint and line_ver == constraint.replace('==', '').strip():
                                logging.debug(f"Str match: '{line_ver}' == '{constraint.replace('==', '').strip()}'")
                                return True
                            logging.debug(f"No match for '{line_ver}' against '{constraint}'")
                            return False  # Fails if no match
                        logging.debug(f"No version parsed from line for {package}; assuming fail")
                        return False  # No version, fail for strict
        # If not found, ignore constraint
        if not found:
            logging.debug(f"Package {package} not found in requirements; ignoring constraint '{constraint}'")
            return True
        logging.debug(f"No matching line for {package}; fail constraint '{constraint}'")
        return False

    def generate_report(self):
        """Generate a version management report"""
        report = ["# Version Management Report\n"]
        report.append("\n## Current Versions\n")
        
        for section, versions in self.versions.items():
            report.append(f"\n### {section.title()}\n")
            for pkg, ver in versions.items():
                report.append(f"- {pkg}: {ver}\n")

        report.append("\n## Requirements Files\n")
        for req in self.requirements_files:
            report.append(f"- {req}\n")

        with open("versions/version_report.md", 'w') as f:
            f.writelines(report)
        logging.info("Generated version report")

def main():
    """Main entry point for version management"""
    try:
        manager = VersionManager()
        
        # Update requirements files first (before validation)
        manager.update_requirements()

        # Validate constraints on updated files
        if not manager.validate_constraints():
            logging.error("Constraint validation failed")
            sys.exit(1)

        # Generate report
        manager.generate_report()

        logging.info("Version management completed successfully")
    except Exception as e:
        logging.error(f"Version management failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()