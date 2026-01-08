# Build Tools & Process Management

This section documents the build tools and process management utilities added in v0.1.3.

## Overview

Three new tools have been added to enhance build management and dependency tracking:

1. **Dependency Tracker** (`dependency_tracker.py`)
2. **Enhanced Download Manager** (`enhanced_download_wheelhouse.py`)
3. **Build Process Visualizer** (`build_visualizer.py`)

## Location

All build tools are located in:
```
scripts/build_tools/
├── dependency_tracker.py    # Smart dependency management
├── enhanced_download_wheelhouse.py    # Advanced package downloads
├── build_visualizer.py    # Build process visualization
└── requirements.txt    # Tool dependencies
```

## 1. Dependency Tracker

### Purpose
- Track package dependencies across all requirements files
- Detect version conflicts
- Record build flags and configurations
- Generate dependency reports and visualizations

### Usage
```bash
# Analyze dependencies
./dependency_tracker.py analyze-deps

# Generate report
./dependency_tracker.py generate-report

# Check for conflicts
./dependency_tracker.py check-conflicts
```

## 2. Enhanced Download Manager

### Purpose
- Smart dependency resolution
- Build flag tracking
- Download caching
- Progress tracking
- Offline mode support

### Usage
```bash
# Download with caching
./enhanced_download_wheelhouse.py \
    --requirements "requirements-*.txt" \
    --wheelhouse ./wheelhouse

# Offline mode
./enhanced_download_wheelhouse.py \
    --requirements "requirements-*.txt" \
    --wheelhouse ./wheelhouse \
    --offline
```

## 3. Build Process Visualizer

### Purpose
- Generate build flow diagrams
- Create resource utilization charts
- Visualize build timeline
- Map component relationships

### Usage
```bash
# Generate build flow diagram
./build_visualizer.py generate-flow

# Create build timeline
./build_visualizer.py generate-timeline
```

## Build Process Overview

The build process follows these stages:

1. **Dependency Resolution** (~5 minutes)
   - Download required packages
   - Check version conflicts
   - Cache wheelhouse

2. **Service Builds** (parallel, ~10-15 minutes total)
   - API Service (10m)
   - UI Service (5m)
   - Crawler Service (4m)
   - Worker Service (3m)

## Resource Requirements

Build process typically requires:

- CPU: 60-80% peak during parallel builds
- Memory: 2-3GB during builds
- Disk: 500MB-1GB temporary space
- Network: 100MB-1GB for initial downloads

## Best Practices

1. **Version Control**
   - Always commit wheelhouse index
   - Track dependency changes
   - Document build flags

2. **Build Optimization**
   - Use cached downloads
   - Enable parallel builds
   - Monitor resource usage

3. **Troubleshooting**
   - Check dependency reports
   - Validate offline builds
   - Review build visualizations

## Common Issues

1. **Version Conflicts**
   ```bash
   # Check for conflicts
   ./dependency_tracker.py check-conflicts
   ```

2. **Build Failures**
   - Review generated reports
   - Check resource utilization
   - Validate dependencies

3. **Cache Issues**
   ```bash
   # Clear and rebuild cache
   rm -rf wheelhouse/.cache
   ./enhanced_download_wheelhouse.py --requirements "requirements-*.txt"
   ```

## Maintenance

Regular tasks to keep builds healthy:

1. **Update Dependencies**
   ```bash
   # Generate current state
   ./dependency_tracker.py generate-report > deps_before.md
   
   # After updates
   ./dependency_tracker.py generate-report > deps_after.md
   
   # Compare changes
   diff deps_before.md deps_after.md
   ```

2. **Cache Management**
   ```bash
   # View cache usage
   du -sh wheelhouse/.cache
   
   # Clean old entries
   find wheelhouse/.cache -type f -mtime +30 -delete
   ```

3. **Build Monitoring**
   ```bash
   # Generate fresh visualizations
   ./build_visualizer.py generate-flow
   ./build_visualizer.py generate-timeline
   ```

## Integration with CI/CD

The build tools support CI/CD integration:

1. **Pre-build Checks**
   ```bash
   # Verify dependencies
   ./dependency_tracker.py check-conflicts
   
   # Validate offline capability
   ./enhanced_download_wheelhouse.py --offline
   ```

2. **Build Artifacts**
   ```bash
   # Generate documentation
   ./build_visualizer.py generate-flow
   ./build_visualizer.py generate-timeline
   ```

3. **Post-build Validation**
   ```bash
   # Check final state
   ./dependency_tracker.py generate-report
   ```