#!/usr/bin/env python3
"""
Build Process Network Monitor
Purpose: Track and log all network activity during build process
"""

import sys
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Set
import subprocess
import psutil
import requests
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/build_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class BuildMonitor:
    def __init__(self):
        self.start_time = datetime.now()
        self.downloads: Dict[str, Dict] = {}
        self.network_access: List[Dict] = []
        self.errors: List[Dict] = []
        
    def track_pip_downloads(self, process: subprocess.Popen):
        """Track files downloaded by pip"""
        while True:
            line = process.stdout.readline()
            if not line:
                break
                
            if "Downloading" in line:
                url = line.split("from")[1].strip()
                filename = line.split("/")[-1].strip()
                self.downloads[filename] = {
                    "url": url,
                    "time": datetime.now().isoformat(),
                    "source": "pip"
                }
                logging.info(f"Download detected: {filename}")

    def monitor_network(self):
        """Monitor all network connections"""
        try:
            connections = psutil.net_connections()
            for conn in connections:
                if conn.status == "ESTABLISHED":
                    self.network_access.append({
                        "remote_ip": conn.raddr.ip,
                        "remote_port": conn.raddr.port,
                        "pid": conn.pid,
                        "time": datetime.now().isoformat()
                    })
        except Exception as e:
            logging.error(f"Failed to monitor network: {e}")

    def verify_offline_build(self) -> bool:
        """Verify if build is truly offline"""
        external_access = [
            access for access in self.network_access
            if not self._is_local_address(access["remote_ip"])
        ]
        
        if external_access:
            logging.error("Build attempted external network access:")
            for access in external_access:
                logging.error(f"  - {access['remote_ip']}:{access['remote_port']}")
            return False
        return True

    def _is_local_address(self, ip: str) -> bool:
        """Check if an IP address is local"""
        return (
            ip.startswith("127.") or
            ip.startswith("10.") or
            ip.startswith("172.16.") or
            ip.startswith("192.168.")
        )

    def generate_report(self):
        """Generate build monitoring report"""
        report = {
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "downloads": self.downloads,
            "network_access": self.network_access,
            "errors": self.errors,
            "is_offline": self.verify_offline_build()
        }

        # Save detailed JSON report
        with open("logs/build_monitor_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        # Generate markdown summary
        summary = [
            "# Build Process Monitoring Report\n",
            f"\nBuild Start: {report['start_time']}",
            f"\nBuild End: {report['end_time']}",
            f"\nOffline Build: {'Yes' if report['is_offline'] else 'No'}\n",
            "\n## Downloads\n"
        ]

        for filename, details in report['downloads'].items():
            summary.append(f"- {filename} (from {details['url']})\n")

        if report['errors']:
            summary.append("\n## Errors\n")
            for error in report['errors']:
                summary.append(f"- {error}\n")

        with open("logs/build_monitor_report.md", 'w') as f:
            f.writelines(summary)

        logging.info("Generated build monitoring report")

def main():
    """Main entry point for build monitoring"""
    try:
        monitor = BuildMonitor()
        
        # Start network monitoring
        monitor.monitor_network()
        
        # Monitor pip downloads
        process = subprocess.Popen(
            ['pip', 'install', '-r', 'requirements.txt'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        monitor.track_pip_downloads(process)
        
        # Generate final report
        monitor.generate_report()
        
        if not monitor.verify_offline_build():
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Build monitoring failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()