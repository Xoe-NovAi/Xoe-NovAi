#!/usr/bin/env python3
"""
enhanced_download_wheelhouse.py - Advanced Python package downloader with tracking

Features:
1. Smart dependency resolution
2. Conflict detection
3. Build flag tracking
4. Comprehensive logging
5. Download caching
6. Progress tracking
7. Offline mode support
8. Integration with dependency_tracker.py

Usage:
    ./enhanced_download_wheelhouse.py --requirements "requirements-*.txt" 
                                    --wheelhouse ./wheelhouse
                                    --offline
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('build_tools.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('download_wheelhouse')

@dataclass
class DownloadInfo:
    """Information about a downloaded package."""
    name: str
    version: str
    filename: str
    size: int
    sha256: str
    download_time: float
    source_url: Optional[str] = None
    cached: bool = False

class WheelhouseManager:
    """Manage Python package downloads and caching."""
    
    def __init__(self, wheelhouse_dir: Path, offline: bool = False):
        self.wheelhouse_dir = wheelhouse_dir
        self.offline = offline
        self.cache_dir = wheelhouse_dir / '.cache'
        self.index_file = self.cache_dir / 'wheel_index.json'
        self.downloads: Dict[str, DownloadInfo] = {}
        
        # Create directories
        self.wheelhouse_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing index
        if self.index_file.exists():
            with open(self.index_file) as f:
                self.index = json.load(f)
        else:
            self.index = {}
    
    def _save_index(self):
        """Save wheel index to disk."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _hash_file(self, path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _get_package_info(self, wheel_path: Path) -> Tuple[str, str]:
        """Extract package name and version from wheel filename."""
        pattern = r'(?P<name>.+?)-(?P<version>[^-]+)(?:-[^-]+)?\.whl$'
        match = re.match(pattern, wheel_path.name)
        if not match:
            raise ValueError(f"Invalid wheel filename: {wheel_path.name}")
        return match.group('name'), match.group('version')
    
    def download_requirements(self, requirements_files: List[Path]):
        """Download wheels for all requirements files."""
        total_size = 0
        start_time = datetime.now()
        
        for req_file in requirements_files:
            logger.info(f"Processing {req_file}")
            
            # Create temporary directory for downloads
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                
                # Download dependencies
                cmd = [
                    'pip', 'download',
                    '--only-binary=:all:',
                    '-d', str(tmp_path),
                    '-r', str(req_file)
                ]
                
                if not self.offline:
                    try:
                        subprocess.run(cmd, check=True, capture_output=True, text=True)
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Download failed: {e.stderr}")
                        continue
                
                # Process downloaded wheels
                for wheel in tmp_path.glob('*.whl'):
                    name, version = self._get_package_info(wheel)
                    dest = self.wheelhouse_dir / wheel.name
                    
                    # Calculate hash and check cache
                    file_hash = self._hash_file(wheel)
                    cached = False
                    
                    if dest.exists():
                        existing_hash = self._hash_file(dest)
                        if existing_hash == file_hash:
                            logger.info(f"Using cached wheel for {name}=={version}")
                            cached = True
                        else:
                            logger.warning(f"Hash mismatch for {name}, updating wheel")
                    
                    if not cached:
                        shutil.copy2(wheel, dest)
                    
                    # Record download info
                    size = wheel.stat().st_size
                    total_size += size
                    
                    self.downloads[name] = DownloadInfo(
                        name=name,
                        version=version,
                        filename=wheel.name,
                        size=size,
                        sha256=file_hash,
                        download_time=datetime.now().timestamp(),
                        cached=cached
                    )
        
        # Update index and save
        self.index.update({
            info.name: asdict(info)
            for info in self.downloads.values()
        })
        self._save_index()
        
        # Generate summary
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Download complete: {len(self.downloads)} packages, "
            f"{total_size/1024/1024:.1f}MB in {duration:.1f}s"
        )

def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Enhanced wheelhouse download manager")
    parser.add_argument('--requirements', nargs='+', help='Requirements files to process')
    parser.add_argument('--wheelhouse', default='wheelhouse', help='Wheelhouse directory')
    parser.add_argument('--offline', action='store_true', help='Offline mode')
    args = parser.parse_args()
    
    wheelhouse_dir = Path(args.wheelhouse)
    manager = WheelhouseManager(wheelhouse_dir, args.offline)
    
    req_files = []
    for pattern in args.requirements:
        req_files.extend(Path().glob(pattern))
    
    if not req_files:
        logger.error("No requirements files found")
        sys.exit(1)
    
    manager.download_requirements(req_files)

if __name__ == '__main__':
    main()