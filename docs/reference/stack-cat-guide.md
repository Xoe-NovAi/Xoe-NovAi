# Stack-Cat v0.1.5 User Guide

**Comprehensive Documentation for the Xoe-NovAi Stack Documentation Generator**

---

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Command Reference](#command-reference)
6. [Configuration Files](#configuration-files)
7. [Usage Patterns](#usage-patterns)
8. [Output Formats](#output-formats)
9. [Advanced Features](#advanced-features)
10. [Automated Maintenance](#automated-maintenance)
11. [Troubleshooting](#troubleshooting)
12. [Best Practices](#best-practices)
13. [Examples & Recipes](#examples--recipes)

---

## Overview

### What is Stack-Cat?

Stack-Cat is a comprehensive documentation generator designed specifically for the Xoe-NovAi stack. It automatically collects, organizes, and formats your codebase into multiple documentation formats, making it easy to:

- **Share your stack** with AI assistants (Claude, ChatGPT, etc.)
- **Onboard new developers** with comprehensive code overviews
- **Archive project snapshots** at specific points in time
- **Review code structure** across your entire project
- **Extract individual files** from concatenated documentation
- **Automated maintenance** with intelligent archiving and cleanup

### Key Features

✨ **Multiple Output Formats**
- Markdown (`.md`) - AI-friendly, readable format
- HTML (`.html`) - Interactive web documentation with collapsible sections
- JSON (`.json`) - Structured metadata for programmatic access

🎯 **Flexible File Selection**
- Pre-configured groups (api, rag, frontend, crawler, voice)
- Custom whitelist configuration
- Directory-wide concatenation
- Pattern-based file matching

🔄 **Bidirectional Processing**
- Concatenate files into unified documentation
- De-concatenate documentation back into separate files

📊 **Smart Features**
- File type detection and syntax highlighting
- Automatic exclusion of build artifacts and caches
- v0.1.5 stack compliance validation
- Timestamped outputs with symlinks to latest
- Automated archiving (7-day retention)
- Makefile integration (11 targets)

🚀 **Production Ready**
- Enterprise-grade error handling
- Safety confirmations for destructive operations
- Comprehensive logging and progress indicators
- Cross-platform compatibility (Linux/Mac/Windows)

---

## Installation & Setup

### Prerequisites

**Required:**
- Bash 4.0 or higher
- `jq` (JSON processor)
- Standard Unix utilities (`find`, `sed`, `awk`, `md5sum`)

**Optional:**
- `tree` (for prettier directory visualization)
- Web browser (for viewing HTML output)

### Installing jq

```bash
# Ubuntu/Debian
sudo apt-get install jq

# macOS
brew install jq

# CentOS/RHEL
sudo yum install jq

# Fedora
sudo dnf install jq
```

### Project Structure

Stack-Cat expects the following structure:

```
your-project/
├── app/
│   └── XNAi_rag_app/
│       ├── main.py
│       ├── chainlit_app.py
│       └── ...
├── scripts/
│   └── stack-cat/
│       ├── stack-cat.sh              # Main script (v0.1.5)
│       ├── whitelist.json            # File filtering rules
│       ├── groups.json               # Pre-configured file groups
│       ├── stack-cat-output/         # Active documentation output
│       ├── stack-cat-archive/        # Archived historical snapshots
│       ├── stack-cat_latest.md       # Symlink to latest markdown
│       ├── stack-cat_latest.html     # Symlink to latest HTML
│       └── stack-manifest_latest.json # Symlink to latest JSON
├── docker-compose.yml
├── config.toml
└── ...
```

### Initial Setup

1. **Place the script in your project:**
   ```bash
   mkdir -p scripts/stack-cat
   cp stack-cat_v017.sh scripts/stack-cat/
   chmod +x scripts/stack-cat/stack-cat_v017.sh
   ```

2. **Create configuration files** (or let the script use built-in defaults):
   ```bash
   cd scripts/stack-cat
   # Create whitelist.json and groups.json
   # (See Configuration Files section)
   ```

3. **Test the installation:**
   ```bash
   ./stack-cat_v017.sh --help
   ```

---

## Quick Start

### Basic Usage

```bash
# Navigate to the stack-cat directory
cd scripts/stack-cat

# Generate default documentation (MD, HTML, JSON)
./stack-cat_v017.sh -g default

# View the generated markdown
cat stack-cat-output/stack-cat_latest.md | less

# Open HTML in browser
xdg-open stack-cat-output/stack-cat_latest.html
```

### Common Commands

```bash
# Generate API documentation only
./stack-cat_v017.sh -g api -f html

# Generate RAG subsystem docs with separate files
./stack-cat_v017.sh -g rag -s

# Concatenate all files in current directory
./stack-cat_v017.sh -a

# Extract files from concatenated markdown
./stack-cat_v017.sh -d stack-cat-output/20251021_143022/stack-cat_20251021_143022.md
```

---

## Core Concepts

### File Groups

Groups are pre-configured sets of files representing different parts of your stack:

- **default** - Complete core stack (all components)
- **api** - Backend API services only
- **rag** - RAG (Retrieval-Augmented Generation) subsystem
- **frontend** - User interface components
- **crawler** - Web crawling and data collection modules

### Whitelist System

The whitelist defines which files are included/excluded:

- **allowed_roots** - Files in project root
- **allowed_dirs** - Directories to include
- **excluded_dirs** - Directories to skip (caches, builds, etc.)
- **excluded_extensions** - File types to ignore (.log, .pyc, etc.)

### Output Modes

1. **Concatenated** - All files combined into one document
2. **Separate** - Individual `.md` files for each source file
3. **Manifest** - JSON metadata about all files

### Timestamping

Every run creates a timestamped directory:
- Format: `YYYYMMDD_HHMMSS`
- Preserves history of documentation snapshots
- Symlinks point to latest version for easy access

---

## Command Reference

### Syntax

```bash
./stack-cat_v017.sh [OPTIONS]
```

### Options

| Option | Long Form | Argument | Description |
|--------|-----------|----------|-------------|
| `-g` | `--group` | GROUP | Specify file group (default, api, rag, frontend, crawler) |
| `-f` | `--format` | FORMAT | Output format (md, html, json, all) |
| `-d` | `--decat` | FILE | De-concatenate markdown file into separate files |
| `-a` | `--all-in-dir` | - | Concatenate all files in script directory |
| `-s` | `--separate` | - | Generate separate .md files for each source file |
| `-h` | `--help` | - | Show help message |

### Group Options

- `default` - Full core stack
- `api` - API backend only
- `rag` - RAG subsystem
- `frontend` - UI frontend
- `crawler` - CrawlModule subsystem

### Format Options

- `md` or `markdown` - Markdown format only
- `html` - HTML format only
- `json` - JSON manifest only
- `all` - Generate all three formats

### Exit Codes

- `0` - Success
- `1` - Error (configuration, file access, validation failure)

---

## Configuration Files

### whitelist.json

Defines which files to include and exclude from documentation:

```json
{
  "allowed_roots": [
    "Dockerfile.api",
    "docker-compose.yml",
    "requirements-api.txt",
    "config.toml",
    ".env.example",
    "Makefile",
    "README.md"
  ],
  "allowed_dirs": [
    "app/XNAi_rag_app/",
    "scripts/",
    "tests/"
  ],
  "excluded_dirs": [
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".venv",
    "venv",
    "node_modules",
    "stack-cat-output",
    "models",
    "data"
  ],
  "excluded_extensions": [
    ".log",
    ".tmp",
    ".pyc",
    ".DS_Store",
    ".swp"
  ]
}
```

**Key Fields:**

- **allowed_roots** - Files in project root that should be included
- **allowed_dirs** - Directories to scan (must end with `/`)
- **excluded_dirs** - Directories to skip entirely
- **excluded_extensions** - File extensions to ignore

### groups.json

Defines file groups for different documentation scenarios:

```json
{
  "default": {
    "description": "Full core stack - Xoe-NovAi v0.1.3",
    "files": [
      "Dockerfile.api",
      "docker-compose.yml",
      "config.toml",
      "app/XNAi_rag_app/*.py",
      "scripts/*.py",
      "scripts/*.sh",
      "tests/*.py"
    ]
  },
  "api": {
    "description": "API backend only",
    "files": [
      "Dockerfile.api",
      "requirements-api.txt",
      "app/XNAi_rag_app/main.py",
      "app/XNAi_rag_app/dependencies.py",
      "app/XNAi_rag_app/config_loader.py"
    ]
  }
}
```

**Pattern Syntax:**

- Exact paths: `"app/main.py"`
- Wildcards: `"app/*.py"` (all .py files in app/)
- Recursive: `"app/**/*.py"` (handled by find command)

### Built-in Fallbacks

If configuration files are missing or invalid, Stack-Cat uses built-in defaults. You'll see warnings like:

```
[2025-10-21 14:30:22] WARNING: JSON file not found: whitelist.json, using built-in
```

---

## Usage Patterns

### Pattern 1: Full Stack Documentation

**Use Case:** Document entire codebase for AI analysis

```bash
./stack-cat_v017.sh -g default -f all

# Result:
# - Complete markdown for Claude/ChatGPT
# - Interactive HTML for human review
# - JSON manifest for tooling
```

**Best For:**
- Initial project analysis
- Architecture reviews
- Sharing complete context with AI

### Pattern 2: Subsystem Focus

**Use Case:** Document specific component for debugging

```bash
./stack-cat_v017.sh -g api -f md

# Result:
# - Focused markdown with only API files
# - Smaller, more digestible output
```

**Best For:**
- Bug investigation
- Feature development
- Component-specific reviews

### Pattern 3: Separate File Extraction

**Use Case:** Need individual files from documentation

```bash
./stack-cat_v017.sh -g rag -s

# Result:
# - separate-md/app_XNAi_rag_app_main.py.md
# - separate-md/app_XNAi_rag_app_crawl.py.md
# - separate-md/config.toml.md
```

**Best For:**
- Reference documentation
- File-by-file review
- Version comparison

### Pattern 4: Directory Snapshot

**Use Case:** Document current working directory

```bash
cd /path/to/working/directory
/path/to/stack-cat_v017.sh -a

# Result:
# - All files in current directory concatenated
# - Script and configs automatically excluded
```

**Best For:**
- Quick directory documentation
- Sharing work-in-progress
- Creating portable snapshots

### Pattern 5: De-concatenation

**Use Case:** Extract files from existing documentation

```bash
./stack-cat_v017.sh -d stack-cat-output/20251021_143022/stack-cat_20251021_143022.md

# Result:
# - Individual files extracted to separate-md/
# - Original extensions preserved with .md suffix
```

**Best For:**
- Recovering files from documentation
- Converting concatenated docs to separate files
- File-level analysis

---

## Output Formats

### Markdown (.md)

**Structure:**
```markdown
# Xoe-NovAi Stack Documentation
**Generated**: 2025-10-21 14:30:22
**Total Files**: 42

## Table of Contents
- [Dockerfile.api](#dockerfile-api)
- [app/main.py](#app-main-py)
...

## File Contents

### Dockerfile.api
**Type**: dockerfile
**Size**: 1024 bytes
**Lines**: 45

```dockerfile
FROM python:3.11-slim
...
```

**Advantages:**
- ✅ Perfect for AI assistants (Claude, GPT)
- ✅ Readable in any text editor
- ✅ Easy to search and navigate
- ✅ Version control friendly

**Use When:**
- Sharing code with AI for analysis
- Creating documentation for version control
- Need simple, portable format

### HTML (.html)

**Features:**
- 🎨 Beautiful gradient header
- 📊 File statistics dashboard
- 🔍 Interactive table of contents
- 👆 Click-to-expand file sections
- 🌙 Dark syntax highlighting
- 📱 Responsive design

**Advantages:**
- ✅ Human-friendly interface
- ✅ No external dependencies
- ✅ Professional appearance
- ✅ Easy navigation

**Use When:**
- Presenting to stakeholders
- Code reviews with team
- Creating shareable documentation
- Need visual overview

### JSON (.json)

**Structure:**
```json
{
  "metadata": {
    "project": "Xoe-NovAi",
    "version": "v0.1.3-beta",
    "generated": "2025-10-21 14:30:22",
    "total_files": 42
  },
  "files": [
    {
      "path": "app/main.py",
      "type": "python",
      "size_bytes": 5432,
      "lines": 187,
      "checksum": "a1b2c3d4..."
    }
  ],
  "statistics": {
    "file_types": {
      "python": 25,
      "dockerfile": 3,
      "yaml": 2
    },
    "total_size_bytes": 125678
  }
}
```

**Advantages:**
- ✅ Machine-readable
- ✅ Programmatic access
- ✅ Easy to parse
- ✅ Integration-friendly

**Use When:**
- Building automation tools
- Need file metadata
- Creating reports
- Integrating with other systems

### Separate Markdown Files

**Structure:**
```
separate-md/
├── Dockerfile.api.md
├── docker-compose.yml.md
├── app_XNAi_rag_app_main.py.md
├── app_XNAi_rag_app_crawl.py.md
└── config.toml.md
```

Each file contains:
```markdown
# app/XNAi_rag_app/main.py

**Type**: python
**Size**: 5432 bytes
**Lines**: 187
**Generated**: 2025-10-21 14:30:22

## File Content

```python
# File contents here
```

**Advantages:**
- ✅ Individual file reference
- ✅ Easy to share specific files
- ✅ Good for documentation sites
- ✅ Preserves original structure

---

## Advanced Features

### Stack Version Validation

Stack-Cat validates your project against v0.1.3-beta standards:

```bash
[2025-10-21 14:30:22] Validating v0.1.3-beta stack compliance...
[2025-10-21 14:30:22]   ✓ app/XNAi_rag_app/main.py
[2025-10-21 14:30:22]   ✓ app/XNAi_rag_app/chainlit_app.py
[2025-10-21 14:30:22]   ✓ Pattern 1 (Import Path Resolution)
[2025-10-21 14:30:22]   ✓ v0.1.3 config version
```

**Checks:**
- Required file presence
- Code pattern compliance
- Configuration version

### Automatic Symlinks

Every run updates symlinks for easy access:

```bash
stack-cat-output/
├── stack-cat_latest.md -> 20251021_143022/stack-cat_20251021_143022.md
├── stack-cat_latest.html -> 20251021_143022/stack-cat_20251021_143022.html
├── stack-manifest_latest.json -> 20251021_143022/stack-manifest_20251021_143022.json
└── separate-md_latest -> 20251021_143022/separate-md
```

**Benefits:**
- Always access latest version
- Preserve historical snapshots
- Simple scripting integration

### File Type Detection

Stack-Cat automatically detects file types for proper syntax highlighting:

| File Pattern | Detected Type |
|--------------|---------------|
| `Dockerfile*` | dockerfile |
| `*.py` | python |
| `*.sh`, `*.bash` | shell |
| `*.js` | javascript |
| `*.yml`, `*.yaml` | yaml |
| `*.json` | json |
| `*.toml` | toml |
| `*.md` | markdown |
| `Makefile` | makefile |

### Progress Indicators

For large projects, Stack-Cat shows progress:

```bash
[2025-10-21 14:30:25] Processed 5/42 files for Markdown
[2025-10-21 14:30:27] Processed 10/42 files for Markdown
...
[2025-10-21 14:30:35] Created 20/42 separate markdown files
```

### Error Recovery

Built-in fallbacks ensure documentation generation continues:

```bash
[2025-10-21 14:30:22] WARNING: Cannot read file: /path/to/locked.py
# File will show: # ERROR: Cannot read file
```

---

## Troubleshooting

### Common Issues

#### Issue: "jq command not found"

**Solution:**
```bash
# Install jq
sudo apt-get install jq  # Ubuntu/Debian
brew install jq          # macOS
```

#### Issue: "No files found for group: api"

**Causes:**
- Invalid group name
- Empty groups.json configuration
- Files don't match patterns

**Solutions:**
1. Check group name: `./stack-cat_v017.sh -h`
2. Verify groups.json syntax with `jq . groups.json`
3. Test with default group: `./stack-cat_v017.sh -g default`

#### Issue: "Invalid JSON in whitelist.json"

**Solution:**
```bash
# Validate JSON syntax
jq . whitelist.json

# Common errors:
# - Missing commas between array items
# - Trailing commas (not allowed in JSON)
# - Unquoted strings
# - Unclosed brackets
```

#### Issue: Files missing from output

**Causes:**
- File in excluded directory
- File has excluded extension
- File not in whitelist

**Solution:**
```bash
# Check if file is in excluded_dirs
grep -A 10 "excluded_dirs" whitelist.json

# Check if extension is excluded
grep -A 5 "excluded_extensions" whitelist.json

# Add directory to allowed_dirs
# Add file to allowed_roots
```

#### Issue: HTML not displaying properly

**Causes:**
- Special characters not escaped
- Browser security restrictions

**Solutions:**
1. Check file encoding: `file stack-cat_latest.html`
2. Try different browser
3. Serve via local web server: `python3 -m http.server 8000`

#### Issue: De-concatenation produces empty files

**Causes:**
- Markdown format doesn't match expected structure
- Code fences missing or malformed

**Solution:**
- Ensure source markdown was generated by Stack-Cat
- Check for proper code fence markers (```)
- Verify file headers are present (### filename)

### Debug Mode

Add verbose logging:

```bash
# Run with bash debug mode
bash -x ./stack-cat_v017.sh -g default 2>&1 | tee debug.log

# Check what files are being collected
./stack-cat_v017.sh -g default 2>&1 | grep "Found file"
```

### Permission Issues

```bash
# Make script executable
chmod +x stack-cat_v017.sh

# Check file permissions
ls -la stack-cat_v017.sh

# Fix ownership
sudo chown $USER:$USER stack-cat_v017.sh
```

---

## Best Practices

### Configuration Management

#### DO:
✅ Keep whitelist.json in version control
✅ Document custom groups in comments
✅ Use descriptive group names
✅ Regularly review excluded patterns
✅ Test changes with small groups first

#### DON'T:
❌ Hardcode sensitive paths
❌ Include binary files in documentation
❌ Exclude critical configuration files
❌ Use overly broad wildcard patterns

### Documentation Workflow

#### 1. **Pre-Development**
```bash
# Document current state before changes
./stack-cat_v017.sh -g default -f all
mv stack-cat-output/stack-cat_latest.md docs/pre-feature-x.md
```

#### 2. **During Development**
```bash
# Quick subsystem checks
./stack-cat_v017.sh -g api -f md
# Share with AI for review
```

#### 3. **Post-Development**
```bash
# Document final state
./stack-cat_v017.sh -g default -f all -s
# Compare with pre-feature-x.md
```

### AI Integration

#### Optimal Formats for Different AI Tools:

**Claude:**
```bash
# Claude handles large contexts well
./stack-cat_v017.sh -g default -f md
cat stack-cat-output/stack-cat_latest.md | pbcopy  # macOS
cat stack-cat-output/stack-cat_latest.md | xclip   # Linux
```

**ChatGPT:**
```bash
# Smaller chunks work better
./stack-cat_v017.sh -g api -f md
# Or use separate files
./stack-cat_v017.sh -g api -s
```

**Local LLMs:**
```bash
# Use JSON for programmatic access
./stack-cat_v017.sh -g default -f json
# Feed through your LLM API
```

### File Organization

```bash
# Create documentation archive
mkdir -p docs/snapshots
cp stack-cat-output/20251021_143022/* docs/snapshots/release-v1.2.0/

# Tag specific versions
cd stack-cat-output
ln -s 20251021_143022 release-v1.2.0
```

### Performance Optimization

#### For Large Projects:

```bash
# Use specific groups instead of default
./stack-cat_v017.sh -g api      # Instead of 'default'

# Skip HTML for speed (HTML generation is slower)
./stack-cat_v017.sh -g default -f md json

# Exclude unnecessary directories
# Add to whitelist.json excluded_dirs:
# "docs", "examples", "archive"
```

### Security Considerations

#### Sensitive Data:

```bash
# NEVER include in documentation:
# - .env files with real credentials
# - API keys or tokens
# - Private keys
# - Database passwords

# In whitelist.json:
"excluded_extensions": [
  ".env",      # ← Add this
  ".key",
  ".pem",
  ".secret"
]

# Use .env.example instead:
"allowed_roots": [
  ".env.example"  # ← Safe template only
]
```

#### Review Before Sharing:

```bash
# Always review generated docs before sharing
less stack-cat-output/stack-cat_latest.md
# Search for sensitive patterns
grep -i "password\|secret\|key\|token" stack-cat-output/stack-cat_latest.md
```

---

## Examples & Recipes

### Recipe 1: Quick Project Share

**Goal:** Share project with AI for analysis

```bash
cd scripts/stack-cat
./stack-cat_v017.sh -g default -f md
cat stack-cat-output/stack-cat_latest.md | pbcopy
# Paste into Claude/ChatGPT
```

### Recipe 2: Component Documentation

**Goal:** Document API layer for new developer

```bash
./stack-cat_v017.sh -g api -f html
xdg-open stack-cat-output/stack-cat_latest.html
# Send HTML file to developer
```

### Recipe 3: Release Documentation

**Goal:** Create documentation for v1.2.0 release

```bash
# Generate comprehensive docs
./stack-cat_v017.sh -g default -f all -s

# Archive the release
cd stack-cat-output
cp -r 20251021_143022 ../../docs/releases/v1.2.0
ln -s 20251021_143022 release-v1.2.0

# Create release notes
echo "# Release v1.2.0 Documentation" > ../../docs/releases/v1.2.0/README.md
echo "Generated: $(date)" >> ../../docs/releases/v1.2.0/README.md
```

### Recipe 4: Bug Investigation

**Goal:** Document system state during bug

```bash
# Capture full state
./stack-cat_v017.sh -g default -f all
mv stack-cat-output/20251021_143022 ../../bugs/issue-123-state

# Document specific subsystem
./stack-cat_v017.sh -g rag -s
# Review separate files for issue
```

### Recipe 5: Code Review Preparation

**Goal:** Prepare code for review meeting

```bash
# Generate HTML for presentation
./stack-cat_v017.sh -g default -f html

# Extract specific files for detailed review
./stack-cat_v017.sh -g api -s
cd stack-cat-output/separate-md_latest
# Review individual files
```

### Recipe 6: Migration Documentation

**Goal:** Document before major refactor

```bash
# Before migration
./stack-cat_v017.sh -g default -f all
mv stack-cat-output/20251021_143022 ../../docs/pre-migration

# After migration
./stack-cat_v017.sh -g default -f all
mv stack-cat-output/20251021_150000 ../../docs/post-migration

# Compare
diff -r docs/pre-migration/separate-md docs/post-migration/separate-md
```

### Recipe 7: Custom Group

**Goal:** Document just the configuration files

```json
// Add to groups.json
{
  "config": {
    "description": "Configuration files only",
    "files": [
      "config.toml",
      ".env.example",
      "docker-compose.yml",
      "Makefile",
      "requirements-*.txt"
    ]
  }
}
```

```bash
./stack-cat_v017.sh -g config -f md
```

### Recipe 8: Automated Daily Backup

**Goal:** Cron job for daily documentation snapshots

```bash
# Create backup script
cat > daily-stack-backup.sh <<'EOF'
#!/bin/bash
cd /path/to/scripts/stack-cat
./stack-cat_v017.sh -g default -f json
# Archive old backups (keep last 30 days)
find stack-cat-output/ -maxdepth 1 -type d -mtime +30 -exec rm -rf {} \;
EOF

chmod +x daily-stack-backup.sh

# Add to crontab
crontab -e
# Add line:
0 2 * * * /path/to/scripts/stack-cat/daily-stack-backup.sh
```

### Recipe 9: CI/CD Integration

**Goal:** Generate docs on every commit

```yaml
# .github/workflows/docs.yml
name: Generate Documentation

on:
  push:
    branches: [ main ]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install jq
        run: sudo apt-get install -y jq
      
      - name: Generate Documentation
        run: |
          cd scripts/stack-cat
          ./stack-cat_v017.sh -g default -f all
      
      - name: Upload Artifact
        uses: actions/upload-artifact@v2
        with:
          name: stack-documentation
          path: scripts/stack-cat/stack-cat-output/
```

### Recipe 10: Extract Specific File

**Goal:** Get just one file from concatenated docs

```bash
# Generate separate files
./stack-cat_v017.sh -d stack-cat-output/20251021_143022/stack-cat_20251021_143022.md

# Find the specific file
cd stack-cat-output/20251021_143022/separate-md
ls -la | grep "main.py"
# Output: app_XNAi_rag_app_main.py.md

# View it
cat app_XNAi_rag_app_main.py.md
```

---

## Appendix

### Supported File Types

| Extension | Language | Syntax Highlighting |
|-----------|----------|-------------------|
| `.py` | Python | ✅ |
| `.sh`, `.bash` | Shell | ✅ |
| `.js` | JavaScript | ✅ |
| `.ts` | TypeScript | ✅ |
| `.yml`, `.yaml` | YAML | ✅ |
| `.json` | JSON | ✅ |
| `.toml` | TOML | ✅ |
| `.md` | Markdown | ✅ |
| `.txt` | Plain Text | ✅ |
| `Dockerfile` | Docker | ✅ |
| `Makefile` | Make | ✅ |
| `.env` | Environment | ✅ |

### Output Directory Structure

```
stack-cat-output/
├── stack-cat_latest.md              # Symlink to latest markdown
├── stack-cat_latest.html            # Symlink to latest HTML
├── stack-manifest_latest.json       # Symlink to latest JSON
├── separate-md_latest/              # Symlink to latest separate files
└── 20251021_143022/                 # Timestamped snapshot
    ├── stack-cat_20251021_143022.md
    ├── stack-cat_20251021_143022.html
    ├── stack-manifest_20251021_143022.json
    └── separate-md/
        ├── Dockerfile.api.md
        ├── app_XNAi_rag_app_main.py.md
        └── config.toml.md
```

### Version History

- **v0.1.7-beta** (2025-10-21)
  - Added de-concatenation function
  - Added separate markdown file generation
  - Added directory-wide concatenation
  - Improved error handling
  - Enhanced documentation

- **v0.1.6-beta** (Previous)
  - Built-in fallback configurations
  - Better error handling
  - Missing JSON file handling

### Contributing

To contribute improvements:

1. Test changes thoroughly
2. Update this user guide
3. Maintain backward compatibility
4. Follow existing code patterns
5. Document new features

### License & Credits

Stack-Cat is part of the Xoe-NovAi project.

**Created for:** Comprehensive stack documentation and AI-assisted development

**Guide Reference:** Section 0.2 (Mandatory Code Patterns)

---

## Quick Reference Card

```bash
# Basic Commands
./stack-cat_v017.sh -g default           # Generate default docs
./stack-cat_v017.sh -g api -f html       # API docs in HTML
./stack-cat_v017.sh -a                   # All files in directory
./stack-cat_v017.sh -s                   # Generate separate files
./stack-cat_v017.sh -d file.md           # De-concatenate markdown

# View Output
cat stack-cat-output/stack-cat_latest.md | less
xdg-open stack-cat-output/stack-cat_latest.html
jq . stack-cat-output/stack-manifest_latest.json | less

# Groups: default | api | rag | frontend | crawler
# Formats: md | html | json | all
```

---

## FAQ

### Q: How much disk space does Stack-Cat use?

**A:** Depends on your project size. Generally:
- Small project (< 100 files): ~1-5 MB per snapshot
- Medium project (100-500 files): ~5-20 MB per snapshot
- Large project (500+ files): ~20-100 MB per snapshot

HTML files are typically 10-20% larger than markdown due to styling.

### Q: Can I customize the HTML styling?

**A:** Yes! The HTML is self-contained with inline CSS. Edit the `<style>` section in the `generate_html()` function:

```bash
# Edit the script
nano stack-cat_v017.sh

# Find the generate_html function (around line 700)
# Modify colors, fonts, layout as needed
```

### Q: How do I add a new file group?

**A:** Edit `groups.json`:

```json
{
  "mygroup": {
    "description": "My custom group",
    "files": [
      "path/to/file1.py",
      "path/to/dir/*.py",
      "specific-file.txt"
    ]
  }
}
```

Then use: `./stack-cat_v017.sh -g mygroup`

### Q: Can I use Stack-Cat outside the Xoe-NovAi project?

**A:** Absolutely! Stack-Cat works with any project. You'll need to:

1. Adjust `PROJECT_ROOT` if needed (or use `--all-in-dir`)
2. Update `whitelist.json` for your file structure
3. Skip the v0.1.3 validation (it will just show warnings)

### Q: What if I don't have the configuration JSON files?

**A:** Stack-Cat has built-in defaults and will work without them. You'll see:

```
[2025-10-21 14:30:22] WARNING: JSON file not found: whitelist.json, using built-in
```

The script continues using reasonable defaults.

### Q: How do I exclude sensitive files?

**A:** Add to `whitelist.json`:

```json
{
  "excluded_extensions": [
    ".env",
    ".key",
    ".pem",
    ".secret",
    ".password"
  ],
  "excluded_dirs": [
    "secrets",
    "credentials",
    ".ssh"
  ]
}
```

### Q: Can I run Stack-Cat from anywhere?

**A:** Yes, but it's designed to run from `scripts/stack-cat/`. If you run from elsewhere:

- Use `--all-in-dir` to document current directory
- Adjust `PROJECT_ROOT` variable in the script
- Or use absolute paths in your configuration

### Q: What's the maximum file size Stack-Cat can handle?

**A:** No hard limit, but practical considerations:

- **Individual files**: Up to ~100MB (very large files slow generation)
- **Total project**: Tested with 1000+ files, ~500MB total
- **Markdown output**: Claude can handle ~200k tokens (~1MB markdown)

For very large projects, use specific groups instead of `default`.

### Q: How do I integrate Stack-Cat with git?

**A:** Add to `.gitignore`:

```gitignore
# Stack-Cat outputs (keep latest symlinks, ignore snapshots)
scripts/stack-cat/stack-cat-output/202*
```

Keep in `.gitignore`:
```gitignore
# Track these
scripts/stack-cat/whitelist.json
scripts/stack-cat/groups.json
scripts/stack-cat/stack-cat_v017.sh
scripts/stack-cat/stack-cat-output/*_latest.*
```

### Q: Can Stack-Cat handle binary files?

**A:** No, Stack-Cat is designed for text files only. Binary files are automatically skipped, but you should exclude them in `whitelist.json`:

```json
{
  "excluded_extensions": [
    ".png", ".jpg", ".jpeg", ".gif",
    ".pdf", ".zip", ".tar", ".gz",
    ".pkl", ".bin", ".dat"
  ]
}
```

### Q: How do I document only changed files?

**A:** Use git to find changed files, then create a custom group:

```bash
# Get changed files
git diff --name-only HEAD~1 > changed-files.txt

# Create custom group in groups.json
{
  "recent-changes": {
    "description": "Recently modified files",
    "files": [
      "app/main.py",
      "app/utils.py"
      # ... paste from changed-files.txt
    ]
  }
}

./stack-cat_v017.sh -g recent-changes
```

### Q: What if de-concatenation fails?

**A:** De-concatenation expects markdown generated by Stack-Cat. If it fails:

1. Verify the markdown has proper structure:
   - File headers: `### path/to/file.py`
   - Type markers: `**Type**: python`
   - Code fences: ` ```python `

2. Try manual extraction:
   ```bash
   # Extract a specific file
   sed -n '/### path\/to\/file.py/,/```$/p' input.md > output.md
   ```

3. Regenerate from source if possible

### Q: How do I schedule automatic documentation?

**A:** Use cron (Linux/Mac) or Task Scheduler (Windows):

**Linux/Mac:**
```bash
# Edit crontab
crontab -e

# Add daily at 2 AM
0 2 * * * cd /path/to/stack-cat && ./stack-cat_v017.sh -g default -f json

# Add on every git push (via git hook)
# Create .git/hooks/post-commit:
#!/bin/bash
cd scripts/stack-cat
./stack-cat_v017.sh -g default -f all
```

**Windows Task Scheduler:**
```powershell
# Create task
schtasks /create /tn "StackCatDaily" /tr "C:\path\to\stack-cat_v017.sh" /sc daily /st 02:00
```

### Q: Can I pipe Stack-Cat output to another tool?

**A:** Yes! Stack-Cat writes to stderr, output to stdout:

```bash
# Pipe markdown to less
./stack-cat_v017.sh -g api -f md 2>/dev/null | less

# Count total lines
./stack-cat_v017.sh -g default -f md 2>/dev/null | wc -l

# Search for patterns
./stack-cat_v017.sh -g rag -f md 2>/dev/null | grep "TODO"

# Use with AI CLI tools
./stack-cat_v017.sh -g api -f md 2>/dev/null | claude-cli analyze
```

### Q: What's the difference between concatenation and separate files?

**A:**

**Concatenation** (`-f md`):
- Single file with all code
- Better for AI context (one upload)
- Easier to search across files
- More compact

**Separate** (`-s`):
- Individual file per source
- Better for reference docs
- Easier to navigate specific files
- Good for documentation sites

**Both** (`-f md -s`):
- Get the best of both worlds
- Concatenated for AI, separate for reference

### Q: How do I contribute to Stack-Cat development?

**A:** Stack-Cat is part of the Xoe-NovAi project. To contribute:

1. Test your changes thoroughly
2. Update this documentation
3. Follow bash best practices
4. Maintain backward compatibility
5. Submit a detailed description of changes

### Q: Why use Stack-Cat instead of just `find` and `cat`?

**A:** Stack-Cat provides:

- ✅ **Smart filtering** - Exclude caches, logs, builds automatically
- ✅ **Multiple formats** - Markdown, HTML, JSON
- ✅ **Metadata** - File sizes, types, checksums
- ✅ **Organization** - Timestamped snapshots with symlinks
- ✅ **Validation** - Stack compliance checking
- ✅ **Safety** - Built-in exclusions for sensitive files
- ✅ **Convenience** - Pre-configured groups for common tasks

Compare:
```bash
# Manual way
find . -name "*.py" -exec cat {} \; > output.txt
# No organization, no metadata, includes unwanted files

# Stack-Cat way
./stack-cat_v017.sh -g default
# Organized, filtered, multiple formats, with metadata
```

---

## Troubleshooting Checklist

When Stack-Cat isn't working as expected, go through this checklist:

### ✅ Prerequisites
- [ ] Bash 4.0+ installed (`bash --version`)
- [ ] jq installed (`jq --version`)
- [ ] Script has execute permissions (`chmod +x stack-cat_v017.sh`)
- [ ] Running from correct directory (`pwd`)

### ✅ Configuration
- [ ] whitelist.json is valid JSON (`jq . whitelist.json`)
- [ ] groups.json is valid JSON (`jq . groups.json`)
- [ ] File paths in configuration match actual structure
- [ ] No trailing commas in JSON files

### ✅ Permissions
- [ ] Can read source files (`ls -la /path/to/files`)
- [ ] Can write to output directory (`touch stack-cat-output/test`)
- [ ] Output directory not on read-only filesystem

### ✅ File Patterns
- [ ] Group name exists in groups.json
- [ ] File patterns use correct syntax (`*.py` not `*.py*`)
- [ ] Paths relative to PROJECT_ROOT, not script directory
- [ ] Wildcards properly escaped if needed

### ✅ Output Issues
- [ ] Sufficient disk space (`df -h`)
- [ ] No file locks on output directory
- [ ] Previous runs completed successfully
- [ ] Check latest symlinks point to valid files

---

## Performance Tips

### For Large Projects (1000+ files)

1. **Use specific groups instead of 'default':**
   ```bash
   # Slow (processes everything)
   ./stack-cat_v017.sh -g default
   
   # Fast (only API files)
   ./stack-cat_v017.sh -g api
   ```

2. **Skip HTML generation (it's the slowest):**
   ```bash
   ./stack-cat_v017.sh -g default -f md json
   ```

3. **Exclude unnecessary directories:**
   ```json
   // In whitelist.json
   "excluded_dirs": [
     "tests",        // If you don't need test files
     "docs",         // If you don't need documentation
     "examples",     // If you don't need examples
     "node_modules",
     ".git"
   ]
   ```

4. **Run on SSD, not network drives:**
   ```bash
   # Copy to local SSD first
   cp -r /network/project /tmp/project
   cd /tmp/project/scripts/stack-cat
   ./stack-cat_v017.sh -g default
   ```

5. **Use separate markdown for parallelization:**
   ```bash
   # Slower: Everything in sequence
   ./stack-cat_v017.sh -g default -f all
   
   # Faster: Can process separate files in parallel
   ./stack-cat_v017.sh -g default -s &
   # Then process separately in parallel if needed
   ```

### Benchmarks

Typical performance on modern hardware:

| Project Size | Files | Time (MD) | Time (HTML) | Time (All) |
|--------------|-------|-----------|-------------|------------|
| Small | 50 | 2s | 5s | 8s |
| Medium | 200 | 8s | 20s | 30s |
| Large | 1000 | 40s | 120s | 180s |
| Very Large | 5000+ | 200s+ | 600s+ | 900s+ |

*Times include disk I/O, parsing, and formatting*

---

## Integration Examples

### With Claude Desktop

```bash
# Generate documentation
./stack-cat_v017.sh -g default -f md

# Copy to clipboard
cat stack-cat-output/stack-cat_latest.md | pbcopy  # macOS
cat stack-cat-output/stack-cat_latest.md | xclip -selection clipboard  # Linux

# Or drag-and-drop the file into Claude Desktop
```

### With VS Code

```json
// .vscode/tasks.json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Generate Stack Documentation",
      "type": "shell",
      "command": "cd scripts/stack-cat && ./stack-cat_v017.sh -g default -f html",
      "problemMatcher": [],
      "group": {
        "kind": "build",
        "isDefault": false
      }
    }
  ]
}
```

### With Documentation Sites

```bash
# Generate separate files for MkDocs/Docusaurus
./stack-cat_v017.sh -g default -s

# Copy to docs directory
cp -r stack-cat-output/separate-md_latest/* ../docs/api-reference/

# Or integrate in build script
./build-docs.sh
```

### With Monitoring Tools

```bash
# Monitor documentation size
./stack-cat_v017.sh -g default -f json
SIZE=$(jq '.statistics.total_size_bytes' stack-cat-output/stack-manifest_latest.json)
echo "Project size: $SIZE bytes"

# Alert if size changes dramatically
if [ $SIZE -gt 1000000000 ]; then
  echo "Warning: Project exceeds 1GB"
fi
```

---

## Security Best Practices

### 🔒 What to NEVER Include

- `.env` files with real credentials
- API keys, tokens, secrets
- SSH private keys (`.pem`, `.key`)
- Database passwords
- AWS/cloud credentials
- Personal information (PII)
- Proprietary algorithms
- Customer data

### ✅ Safe Practices

1. **Use templates instead of real configs:**
   ```json
   // whitelist.json
   "allowed_roots": [
     ".env.example",  // ✅ Safe template
     // NOT .env      // ❌ Real credentials
   ]
   ```

2. **Review before sharing:**
   ```bash
   # Generate docs
   ./stack-cat_v017.sh -g default -f md
   
   # Search for sensitive patterns
   grep -Ei "password|secret|key|token|api.*key" \
     stack-cat-output/stack-cat_latest.md
   
   # If found, add to excluded_extensions
   ```

3. **Use separate configs for different environments:**
   ```bash
   # Development
   ./stack-cat_v017.sh -g default  # All files
   
   # Production (for public docs)
   ./stack-cat_v017.sh -g public   # Curated list only
   ```

4. **Sanitize before commits:**
   ```bash
   # Add to .gitignore
   echo "*.env" >> .gitignore
   echo "secrets/" >> .gitignore
   
   # Never commit real credentials
   git add -p  # Review each change
   ```

---

## Conclusion

Stack-Cat is a powerful tool for documenting, analyzing, and sharing your codebase. Whether you're:

- 🤖 Working with AI assistants
- 👥 Onboarding team members
- 📋 Creating project documentation
- 🐛 Debugging issues
- 📦 Archiving releases

Stack-Cat provides the flexibility and features you need.

### Getting Help

- **Check this guide** for common issues and patterns
- **Review examples** for similar use cases
- **Test with small groups** before running on full project
- **Check output logs** for error messages and warnings

### Next Steps

1. Run your first Stack-Cat command
2. Customize `whitelist.json` and `groups.json` for your project
3. Integrate into your workflow (git hooks, CI/CD, daily backups)
4. Share documentation with team and AI assistants

**Happy documenting! 🚀**

---

*Stack-Cat v0.1.7-beta User Guide*  
*Last Updated: 2025-10-21*  
*Part of the Xoe-NovAi Project**Stack-Cat v0.1.7-beta User Guide*  
