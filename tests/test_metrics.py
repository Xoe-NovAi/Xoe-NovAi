"""
============================================================================
Xoe-NovAi Phase 1 v0.1.2 - Metrics Tests
============================================================================
Purpose: Testing of Prometheus metrics implementation
Guide Reference: Section 5.2 (Memory Metrics)
Last Updated: 2025-10-28

Test Coverage:
  - Memory metrics in bytes and GB
  - Metric deprecation notices
  - Unit conversions
  - Labels and tags

Usage:
  pytest tests/test_metrics.py -v
  pytest tests/test_metrics.py -v --cov
============================================================================
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'app' / 'XNAi_rag_app'))


# ============================================================================
# MEMORY METRICS TESTS
# ============================================================================

@pytest.mark.unit
def test_memory_metrics():
    """Test memory metrics in bytes and GB."""
    from metrics import memory_usage_bytes, memory_usage_gb
    
    # Sample memory value (5 GB in bytes)
    used_bytes = 5 * 1024 * 1024 * 1024
    
    # Set metrics
    memory_usage_bytes.labels(component='system').set(used_bytes)
    memory_usage_gb.labels(component='system').set(used_bytes / (1024**3))
    
    # Get metric values
    bytes_value = memory_usage_bytes.labels(component='system')._value.get()
    gb_value = memory_usage_gb.labels(component='system')._value.get()
    
    # Verify values
    assert bytes_value == used_bytes
    assert abs(gb_value - 5.0) < 0.01  # Allow small float difference

@pytest.mark.unit
def test_memory_metric_labels():
    """Test memory metric labels/components."""
    from metrics import memory_usage_bytes, memory_usage_gb
    
    # Test all supported components
    components = ['system', 'process', 'llm', 'embeddings']
    for component in components:
        # Set a test value
        memory_usage_bytes.labels(component=component).set(1024)
        memory_usage_gb.labels(component=component).set(1.0)
        
        # Verify the metric exists
        assert memory_usage_bytes.labels(component=component)._value.get() == 1024
        assert memory_usage_gb.labels(component=component)._value.get() == 1.0

@pytest.mark.unit
def test_memory_warning_thresholds():
    """Test memory warning threshold conversions."""
    from metrics import check_memory_thresholds
    import os
    
    # Mock config values
    config = {
        'performance': {
            'memory_limit_bytes': 6442450944,  # 6 GB
            'memory_warning_threshold_bytes': 5905580032,  # 5.5 GB
            'memory_critical_threshold_bytes': 6228254720  # 5.8 GB
        }
    }
    
    with patch('metrics.get_config', return_value=config):
        # Test with memory under warning threshold
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.used = 5368709120  # 5 GB
            status, _ = check_memory_thresholds()
            assert status is True
        
        # Test with memory above warning threshold
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.used = 6228254720  # 5.8 GB
            status, _ = check_memory_thresholds()
            assert status is False