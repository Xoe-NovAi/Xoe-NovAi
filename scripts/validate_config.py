#!/usr/bin/env python3
"""
validate-stack-cat-config.py v0.1.2 (UPDATED)
Validate stack-cat-config.yml before running stack-cat-md.sh

Checks:
- YAML syntax
- Required sections
- Pattern validity
- File existence
- Size estimates
- Pattern conflicts (NEW v0.1.2)
- XNAi directory inclusion (NEW v0.1.2)

Usage:
    python validate-stack-cat-config.py
    python validate-stack-cat-config.py --config custom-config.yml
    python validate-stack-cat-config.py --verbose
    python validate-stack-cat-config.py --estimate-size
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed")
    print("Install with: pip install pyyaml")
    sys.exit(1)

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


class ConfigValidator:
    """Validate stack-cat-config.yml with enhanced checks"""
    
    def __init__(self, config_path: str, verbose: bool = False):
        self.config_path = Path(config_path)
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        self.info = []
        self.config = {}
        
    def log_error(self, msg: str):
        """Log an error"""
        self.errors.append(msg)
        print(f"{RED}❌ ERROR{RESET}: {msg}")
    
    def log_warning(self, msg: str):
        """Log a warning"""
        self.warnings.append(msg)
        print(f"{YELLOW}⚠️  WARNING{RESET}: {msg}")
    
    def log_info(self, msg: str):
        """Log info"""
        self.info.append(msg)
        if self.verbose:
            print(f"{BLUE}ℹ️  INFO{RESET}: {msg}")
    
    def log_success(self, msg: str):
        """Log success"""
        print(f"{GREEN}✅ {msg}{RESET}")
    
    def validate_yaml_syntax(self) -> bool:
        """Validate YAML syntax"""
        print(f"\n{BOLD}1. Validating YAML Syntax{RESET}")
        
        if not self.config_path.exists():
            self.log_error(f"Config file not found: {self.config_path}")
            return False
        
        try:
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)
            
            if self.config is None:
                self.log_error("Config file is empty")
                return False
            
            self.log_success("YAML syntax is valid")
            return True
            
        except yaml.YAMLError as e:
            self.log_error(f"YAML syntax error: {e}")
            return False
    
    def validate_structure(self) -> bool:
        """Validate config structure"""
        print(f"\n{BOLD}2. Validating Structure{RESET}")
        
        required_sections = ['settings', 'include_patterns', 'categories']
        optional_sections = ['exclude_patterns', 'advanced', 'formatting', 'hooks']
        
        valid = True
        
        # Check required sections
        for section in required_sections:
            if section not in self.config:
                self.log_error(f"Missing required section: {section}")
                valid = False
            else:
                self.log_success(f"Found section: {section}")
        
        # Check optional sections
        for section in optional_sections:
            if section in self.config:
                self.log_info(f"Found optional section: {section}")
        
        # Validate section types
        if 'settings' in self.config and not isinstance(self.config['settings'], dict):
            self.log_error("'settings' must be a dictionary")
            valid = False
        
        if 'include_patterns' in self.config and not isinstance(self.config['include_patterns'], list):
            self.log_error("'include_patterns' must be a list")
            valid = False
        
        if 'exclude_patterns' in self.config and not isinstance(self.config['exclude_patterns'], list):
            self.log_error("'exclude_patterns' must be a list")
            valid = False
        
        if 'categories' in self.config and not isinstance(self.config['categories'], dict):
            self.log_error("'categories' must be a dictionary")
            valid = False
        
        return valid
    
    def validate_settings(self) -> bool:
        """Validate settings section"""
        print(f"\n{BOLD}3. Validating Settings{RESET}")
        
        if 'settings' not in self.config:
            return False
        
        settings = self.config['settings']
        valid = True
        
        # Check config_version
        if 'config_version' in settings:
            version = settings['config_version']
            self.log_info(f"Config version: {version}")
            
            if not re.match(r'\d+\.\d+\.\d+', str(version)):
                self.log_warning(f"Version format unusual: {version} (expected X.Y.Z)")
        else:
            self.log_warning("No 'config_version' specified")
        
        # Check split size
        if 'default_split_size' in settings:
            split_size = settings['default_split_size']
            if not isinstance(split_size, int) or split_size <= 0:
                self.log_error(f"Invalid default_split_size: {split_size} (must be positive integer)")
                valid = False
            else:
                self.log_info(f"Default split size: {split_size} bytes ({split_size // 1024}KB)")
        
        # Check max file size
        if 'max_file_size' in settings:
            max_size = settings['max_file_size']
            if not isinstance(max_size, int) or max_size <= 0:
                self.log_error(f"Invalid max_file_size: {max_size}")
                valid = False
            else:
                self.log_info(f"Max file size: {max_size} bytes ({max_size // 1024}KB)")
                
                # NEW v0.1.2: Warn if too high
                if max_size > 10485760:
                    self.log_warning(f"max_file_size is large ({max_size // 1024}KB), may include binary files")
        
        return valid
    
    def validate_patterns(self) -> bool:
        """Validate include/exclude patterns"""
        print(f"\n{BOLD}4. Validating Patterns{RESET}")
        
        valid = True
        
        # Validate include patterns
        if 'include_patterns' in self.config:
            patterns = self.config['include_patterns']
            self.log_info(f"Include patterns: {len(patterns)}")
            
            for i, pattern in enumerate(patterns):
                if not isinstance(pattern, str):
                    self.log_error(f"Include pattern {i} is not a string: {pattern}")
                    valid = False
                elif not pattern.strip():
                    self.log_warning(f"Include pattern {i} is empty")
                else:
                    # Check for common mistakes
                    if pattern.endswith('/'):
                        self.log_warning(f"Pattern ends with '/': {pattern} (use '/**' for recursive)")
                    
                    # NEW v0.1.2: Check for XNAi directories
                    if "./library" in pattern or "./knowledge" in pattern:
                        self.log_success(f"XNAi directory included: {pattern}")
                    
                    if self.verbose:
                        self.log_info(f"  - {pattern}")
        
        # Validate exclude patterns
        if 'exclude_patterns' in self.config:
            patterns = self.config['exclude_patterns']
            self.log_info(f"Exclude patterns: {len(patterns)}")
            
            for i, pattern in enumerate(patterns):
                if not isinstance(pattern, str):
                    self.log_error(f"Exclude pattern {i} is not a string: {pattern}")
                    valid = False
                elif self.verbose and pattern.strip():
                    self.log_info(f"  - {pattern}")
                
                # NEW v0.1.2: Check for critical exclusions
                if "stack-cat-files" in pattern:
                    self.log_success(f"Critical exclusion found: {pattern} (prevents recursion)")
        
        return valid
    
    def validate_categories(self) -> bool:
        """Validate categories"""
        print(f"\n{BOLD}5. Validating Categories{RESET}")
        
        if 'categories' not in self.config:
            return False
        
        categories = self.config['categories']
        valid = True
        
        self.log_info(f"Categories: {len(categories)}")
        
        for cat_name, patterns in categories.items():
            if not isinstance(patterns, list):
                self.log_error(f"Category '{cat_name}' patterns must be a list")
                valid = False
                continue
            
            if not patterns:
                self.log_warning(f"Category '{cat_name}' has no patterns")
            
            # NEW v0.1.2: Check for XNAi-specific categories
            if cat_name in ['library_content', 'knowledge_bases']:
                self.log_success(f"XNAi category found: {cat_name}")
            
            if self.verbose:
                self.log_info(f"  {cat_name}: {len(patterns)} patterns")
        
        # Check for duplicate patterns across categories
        all_patterns = []
        for patterns in categories.values():
            all_patterns.extend(patterns)
        
        if len(all_patterns) != len(set(all_patterns)):
            self.log_warning("Some patterns appear in multiple categories (first match wins)")
        
        return valid
    
    # NEW v0.1.2: Add pattern conflict detection
    def validate_pattern_conflicts(self) -> bool:
        """Detect if same file matches multiple category patterns"""
        print(f"\n{BOLD}6. Detecting Pattern Conflicts{RESET}")
        
        if 'categories' not in self.config:
            return True
        
        from pathlib import Path
        
        file_to_categories = {}
        root_dir = self.config_path.parent.parent
        
        categories = self.config['categories']
        
        # Find files matching multiple patterns
        for category, patterns in categories.items():
            if not isinstance(patterns, list):
                continue
            
            for pattern in patterns:
                pattern_clean = pattern.strip('./')
                
                # Simple glob matching
                try:
                    for file in root_dir.rglob('*'):
                        if not file.is_file():
                            continue
                        
                        rel_path = str(file.relative_to(root_dir))
                        
                        # Check if matches pattern (simplified)
                        if file.match(pattern_clean) or rel_path.endswith(pattern_clean.rstrip('*')):
                            if rel_path not in file_to_categories:
                                file_to_categories[rel_path] = []
                            if category not in file_to_categories[rel_path]:
                                file_to_categories[rel_path].append(category)
                except:
                    continue
        
        # Report conflicts
        conflicts = {f: c for f, c in file_to_categories.items() if len(c) > 1}
        
        if conflicts:
            self.log_warning(f"Found {len(conflicts)} files in multiple categories:")
            for file, cats in list(conflicts.items())[:5]:
                self.log_info(f"  {file}: {', '.join(cats)} (first match wins)")
            return False
        else:
            self.log_success("No pattern conflicts detected")
            return True
    
    def estimate_output_size(self) -> Tuple[int, int]:
        """Estimate output size based on patterns"""
        print(f"\n{BOLD}7. Estimating Output Size{RESET}")
        
        if 'include_patterns' not in self.config:
            return 0, 0
        
        root_dir = self.config_path.parent.parent
        total_files = 0
        total_size = 0
        
        for pattern in self.config['include_patterns']:
            pattern_clean = pattern.strip('./')
            
            try:
                for file in root_dir.rglob('*'):
                    if file.is_file() and (file.match(pattern_clean) or 
                                          str(file.relative_to(root_dir)).endswith(pattern_clean.rstrip('*'))):
                        total_files += 1
                        total_size += file.stat().st_size
            except:
                continue
        
        # Apply exclusions (approximate)
        if 'exclude_patterns' in self.config:
            excluded_files = 0
            for pattern in self.config['exclude_patterns']:
                pattern_clean = pattern.strip('./')
                try:
                    for file in root_dir.rglob('*'):
                        if file.is_file() and file.match(pattern_clean):
                            excluded_files += 1
                except:
                    continue
            
            total_files = max(0, total_files - excluded_files)
        
        self.log_info(f"Estimated files: {total_files}")
        self.log_info(f"Estimated size: {total_size} bytes ({total_size // 1024}KB)")
        
        # Estimate split parts
        if 'settings' in self.config and 'default_split_size' in self.config['settings']:
            split_size = self.config['settings']['default_split_size']
            estimated_parts = max(1, total_size // split_size)
            self.log_info(f"Estimated split parts: {estimated_parts}")
        
        return total_files, total_size
    
    def validate_hooks(self) -> bool:
        """Validate hooks"""
        print(f"\n{BOLD}8. Validating Hooks{RESET}")
        
        if 'hooks' not in self.config:
            self.log_info("No hooks configured")
            return True
        
        hooks = self.config['hooks']
        valid = True
        
        for hook_name in ['pre_process', 'post_process']:
            if hook_name in hooks:
                command = hooks[hook_name]
                if command and not isinstance(command, str):
                    self.log_error(f"Hook '{hook_name}' must be a string")
                    valid = False
                elif command:
                    self.log_info(f"{hook_name}: {command}")
        
        if 'webhook_url' in hooks:
            url = hooks['webhook_url']
            if url and not url.startswith(('http://', 'https://')):
                self.log_warning(f"webhook_url doesn't start with