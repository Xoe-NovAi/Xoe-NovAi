#!/usr/bin/env python3
# ============================================================================
# Xoe-NovAi Phase 1 v0.1.2 - Logging Configuration Module (FIXED)
# ============================================================================
# Purpose: Structured JSON logging with rotation and multiple outputs
# Guide Reference: Section 5.2 (JSON Structured Logging)
# Last Updated: 2025-10-19 (COMPLETE FIX - was truncated)
# Features:
#   - JSON formatted logs for machine parsing
#   - Rotating file handler (10MB per file, 5 backups)
#   - Console and file output
#   - Context injection (request_id, user_id, session_id)
#   - Performance logging for token generation
#   - Crawler operation logging
# ============================================================================

import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional

# JSON formatter
try:
    from json_log_formatter import JSONFormatter
except ImportError:
    # Fallback if json_log_formatter not available
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            return json.dumps({
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'module': record.module,
                'message': record.getMessage()
            })

# Configuration
try:
    from config_loader import load_config, get_config_value
    CONFIG = load_config()
except Exception as e:
    print(f"Warning: Could not load config: {e}")
    CONFIG = {'metadata': {'stack_version': 'v0.1.2'}, 'performance': {}}

# ============================================================================
# CUSTOM JSON FORMATTER
# ============================================================================

class XNAiJSONFormatter(JSONFormatter):
    """
    Custom JSON formatter for Xoe-NovAi logs.
    
    Guide Reference: Section 5.2 (Custom JSON Formatting)
    """
    
    def json_record(
        self,
        message: str,
        extra: Dict[str, Any],
        record: logging.LogRecord
    ) -> Dict[str, Any]:
        """Create JSON log record."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": message,
        }
        
        # Add stack version
        try:
            log_entry["stack_version"] = get_config_value("metadata.stack_version", "v0.1.2")
        except:
            log_entry["stack_version"] = "v0.1.2"
        
        # Add process info
        log_entry["process_id"] = record.process
        log_entry["thread_id"] = record.thread
        
        # Add extra fields from context
        if extra:
            filtered_extra = {
                k: v for k, v in extra.items()
                if not k.startswith('_') and k not in ['message', 'asctime']
            }
            log_entry.update(filtered_extra)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }
        
        return log_entry

# ============================================================================
# CONTEXT INJECTION
# ============================================================================

class ContextAdapter(logging.LoggerAdapter):
    """Logger adapter for injecting contextual information."""
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add context to log message."""
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = extra
        return msg, kwargs

# ============================================================================
# PERFORMANCE LOGGING
# ============================================================================

class PerformanceLogger:
    """Performance metrics logger."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize performance logger."""
        self.logger = logger
    
    def log_token_generation(
        self,
        tokens: int,
        duration_s: float,
        model: str = "gemma-3-4b"
    ):
        """Log token generation performance."""
        tokens_per_second = tokens / duration_s if duration_s > 0 else 0
        
        self.logger.info(
            "Token generation completed",
            extra={
                "operation": "token_generation",
                "model": model,
                "tokens": tokens,
                "duration_s": round(duration_s, 3),
                "tokens_per_second": round(tokens_per_second, 2),
                "target_min": CONFIG.get('performance', {}).get('token_rate_min', 15),
                "target_max": CONFIG.get('performance', {}).get('token_rate_max', 25),
            }
        )
    
    def log_memory_usage(self, component: str = "system"):
        """Log current memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            process = psutil.Process()
            
            self.logger.info(
                "Memory usage",
                extra={
                    "operation": "memory_check",
                    "component": component,
                    "system_used_gb": round(memory.used / (1024**3), 2),
                    "system_percent": memory.percent,
                    "process_used_gb": round(process.memory_info().rss / (1024**3), 2),
                    "limit_gb": CONFIG.get('performance', {}).get('memory_limit_gb', 6.0),
                }
            )
        except Exception as e:
            self.logger.warning(f"Could not measure memory: {e}")
    
    def log_query_latency(
        self,
        query: str,
        duration_ms: float,
        success: bool = True,
        error: str = None
    ):
        """Log query processing latency."""
        self.logger.info(
            f"Query {'succeeded' if success else 'failed'}",
            extra={
                "operation": "query_processing",
                "query_preview": query[:100] if query else "",
                "duration_ms": round(duration_ms, 2),
                "success": success,
                "error": error,
                "target_ms": CONFIG.get('performance', {}).get('latency_target_ms', 1000),
            }
        )
    
    def log_crawl_operation(
        self,
        source: str,
        items: int,
        duration_s: float,
        success: bool = True,
        error: str = None
    ):
        """Log crawler operation."""
        items_per_hour = (items / duration_s * 3600) if duration_s > 0 else 0
        
        self.logger.info(
            f"Crawl {'completed' if success else 'failed'}: {source}",
            extra={
                "operation": "crawl",
                "source": source,
                "items": items,
                "duration_s": round(duration_s, 2),
                "items_per_hour": round(items_per_hour, 1),
                "success": success,
                "error": error,
                "target_rate": CONFIG.get('performance', {}).get('crawl_rate_target', 50),
            }
        )

# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

def setup_file_handler(
    log_file: str,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    level: int = logging.INFO
) -> RotatingFileHandler:
    """
    Create rotating file handler.
    
    CRITICAL: This MUST handle the case where the directory doesn't exist
    (created at build time) but we still need to verify it's writable.
    """
    # Ensure log directory exists (CRITICAL - this was failing before)
    log_path = Path(log_file)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        print(f"ERROR: Cannot create logs directory: {e}")
        print(f"  Path: {log_path.parent}")
        print(f"  Current user: {os.getuid()}")
        print(f"  Directory ownership:")
        import subprocess
        try:
            subprocess.run(['ls', '-ld', str(log_path.parent)])
        except:
            pass
        raise
    
    # Create handler
    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    
    handler.setLevel(level)
    handler.setFormatter(XNAiJSONFormatter())
    
    return handler

def setup_console_handler(
    level: int = logging.INFO,
    use_json: bool = True
) -> logging.StreamHandler:
    """Create console handler."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    if use_json:
        handler.setFormatter(XNAiJSONFormatter())
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
    
    return handler

def setup_logging(
    log_level: str = None,
    log_file: str = None,
    console_enabled: bool = True,
    file_enabled: bool = True,
    json_format: bool = True
):
    """
    Configure logging for entire application.
    
    This is the main entrypoint for configuring logging.
    """
    # Get configuration
    if log_level is None:
        try:
            log_level = get_config_value('logging.level', 'INFO')
        except:
            log_level = 'INFO'
    
    if log_file is None:
        try:
            log_file = get_config_value(
                'logging.file_path',
                '/app/XNAi_rag_app/logs/xnai.log'
            )
        except:
            log_file = '/app/XNAi_rag_app/logs/xnai.log'
    
    # Parse log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Add console handler
    if console_enabled:
        try:
            console_handler = setup_console_handler(
                level=numeric_level,
                use_json=json_format
            )
            root_logger.addHandler(console_handler)
        except Exception as e:
            print(f"ERROR: Failed to setup console handler: {e}")
    
    # Add file handler
    if file_enabled:
        try:
            max_size_mb = get_config_value('logging.max_size_mb', 10)
        except:
            max_size_mb = 10
        
        try:
            backup_count = get_config_value('logging.backup_count', 5)
        except:
            backup_count = 5
        
        try:
            file_handler = setup_file_handler(
                log_file=log_file,
                max_bytes=max_size_mb * 1024 * 1024,
                backup_count=backup_count,
                level=numeric_level
            )
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"ERROR: Failed to setup file handler: {e}")
            print(f"  Log file: {log_file}")
            print(f"  Will continue with console logging only")
            file_enabled = False
    
    # Log initialization
    try:
        root_logger.info(
            "Logging configured",
            extra={
                "log_level": log_level,
                "log_file": log_file if file_enabled else None,
                "console_enabled": console_enabled,
                "file_enabled": file_enabled,
                "json_format": json_format,
            }
        )
    except Exception as e:
        print(f"Warning: Could not log initialization: {e}")

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_logger(
    name: str,
    context: Dict[str, Any] = None
) -> logging.Logger:
    """Get configured logger with optional context."""
    logger = logging.getLogger(name)
    
    if context:
        return ContextAdapter(logger, context)
    
    return logger

def log_startup_info():
    """Log application startup information."""
    logger = logging.getLogger('xnai.startup')
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
    except:
        memory = None
        cpu_count = None
    
    # Stack info
    logger.info(
        "Xoe-NovAi starting",
        extra={
            "stack_version": CONFIG.get('metadata', {}).get('stack_version', 'v0.1.2'),
            "codename": CONFIG.get('metadata', {}).get('codename', 'unknown'),
            "phase": CONFIG.get('project', {}).get('phase', 1),
        }
    )
    
    # System info
    if memory and cpu_count:
        logger.info(
            "System information",
            extra={
                "cpu_count": cpu_count,
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
            }
        )

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test logging configuration."""
    print("=" * 70)
    print("Xoe-NovAi Logging Configuration - Test Suite v0.1.2")
    print("=" * 70)
    print()
    
    # Setup logging
    print("Setting up logging...")
    try:
        setup_logging(log_level='INFO', json_format=True)
        print("✓ Logging configured\n")
    except Exception as e:
        print(f"✗ Logging setup failed: {e}\n")
        sys.exit(1)
    
    # Test basic logging
    print("Test 1: Basic logging")
    logger = get_logger(__name__)
    logger.info("Info message")
    logger.warning("Warning message")
    print("✓ Basic logging test complete\n")
    
    # Test context injection
    print("Test 2: Context injection")
    context_logger = get_logger(
        __name__,
        context={'request_id': 'test-123'}
    )
    context_logger.info("Message with context")
    print("✓ Context injection test complete\n")
    
    # Test performance logging
    print("Test 3: Performance logging")
    perf = PerformanceLogger(logger)
    perf.log_token_generation(tokens=100, duration_s=5.0)
    perf.log_memory_usage(component="test")
    print("✓ Performance logging test complete\n")
    
    print("=" * 70)
    print("All logging tests passed!")
    print("=" * 70)