#!/usr/bin/env python3
# ============================================================================
# Xoe-NovAi Phase 1 v0.1.2 - Logging Configuration Module
# ============================================================================
# Purpose: Structured JSON logging with rotation and multiple outputs
# Guide Reference: Section 5.2 (JSON Structured Logging)
# Last Updated: 2025-10-13
# Features:
#   - JSON formatted logs for machine parsing
#   - Rotating file handler (10MB per file, 5 backups)
#   - Console and file output
#   - Context injection (request_id, user_id, session_id)
#   - Performance logging for token generation
#   - NEW v0.1.2: Crawler operation logging
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
from json_log_formatter import JSONFormatter

# Configuration
from config_loader import load_config, get_config_value

CONFIG = load_config()

# ============================================================================
# CUSTOM JSON FORMATTER
# ============================================================================

class XNAiJSONFormatter(JSONFormatter):
    """
    Custom JSON formatter for Xoe-NovAi logs.
    
    Guide Reference: Section 5.2 (Custom JSON Formatting)
    
    Output format:
    {
        "timestamp": "2025-10-13T12:34:56.789Z",
        "level": "INFO",
        "module": "main",
        "function": "query_endpoint",
        "message": "Query processed successfully",
        "request_id": "abc123",
        "duration_ms": 245.6,
        "memory_gb": 4.2,
        "stack_version": "v0.1.2"
    }
    """
    
    def json_record(
        self,
        message: str,
        extra: Dict[str, Any],
        record: logging.LogRecord
    ) -> Dict[str, Any]:
        """
        Create JSON log record.
        
        Args:
            message: Log message
            extra: Extra fields from logger
            record: LogRecord object
            
        Returns:
            Dict for JSON serialization
        """
        # Base fields
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": message,
        }
        
        # Add stack version
        log_entry["stack_version"] = get_config_value("metadata.stack_version", "v0.1.2")
        
        # Add process info
        log_entry["process_id"] = record.process
        log_entry["thread_id"] = record.thread
        
        # Add extra fields from context
        if extra:
            # Filter out internal fields
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
                "traceback": self.formatException(record.exc_info)
            }
        
        return log_entry

# ============================================================================
# CONTEXT INJECTION
# ============================================================================

class ContextAdapter(logging.LoggerAdapter):
    """
    Logger adapter for injecting contextual information.
    
    Guide Reference: Section 5.2 (Context Injection)
    
    This allows attaching request_id, user_id, session_id to all logs
    within a request context.
    
    Example:
        >>> logger = ContextAdapter(logging.getLogger(__name__), {'request_id': '123'})
        >>> logger.info("Processing request")
        # Outputs: {..., "request_id": "123", "message": "Processing request"}
    """
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Add context to log message.
        
        Args:
            msg: Log message
            kwargs: Keyword arguments
            
        Returns:
            Tuple of (message, kwargs with extra)
        """
        # Merge context into extra
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = extra
        
        return msg, kwargs

# ============================================================================
# PERFORMANCE LOGGING
# ============================================================================

class PerformanceLogger:
    """
    Performance metrics logger.
    
    Guide Reference: Section 5.2 (Performance Logging)
    
    This logs token generation rate, memory usage, latency, and
    NEW v0.1.2: crawler operations for monitoring and debugging.
    
    Example:
        >>> perf = PerformanceLogger(logger)
        >>> with perf.measure("query_processing"):
        ...     # Process query
        ...     pass
        # Logs: {..., "operation": "query_processing", "duration_ms": 123.4}
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize performance logger.
        
        Args:
            logger: Base logger to use
        """
        self.logger = logger
    
    def log_token_generation(
        self,
        tokens: int,
        duration_s: float,
        model: str = "gemma-3-4b"
    ):
        """
        Log token generation performance.
        
        Args:
            tokens: Number of tokens generated
            duration_s: Time taken in seconds
            model: Model name
        """
        tokens_per_second = tokens / duration_s if duration_s > 0 else 0
        
        self.logger.info(
            f"Query {'succeeded' if success else 'failed'}",
            extra={
                "operation": "query_processing",
                "query_preview": query[:100],
                "duration_ms": round(duration_ms, 2),
                "success": success,
                "error": error,
                "target_ms": CONFIG['performance']['latency_target_ms'],
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
        """
        Log crawler operation (NEW v0.1.2).
        
        Guide Reference: Section 9 (CrawlModule Logging)
        
        Args:
            source: Crawl source (gutenberg, arxiv, etc.)
            items: Number of items curated
            duration_s: Time taken in seconds
            success: Whether operation succeeded
            error: Error message if failed
        """
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
                "target_rate": CONFIG['performance'].get('crawl_rate_target', 50),
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
    
    Guide Reference: Section 5.2 (File Logging)
    
    Args:
        log_file: Path to log file
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
        level: Logging level
        
    Returns:
        Configured RotatingFileHandler
    """
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    """
    Create console handler.
    
    Guide Reference: Section 5.2 (Console Logging)
    
    Args:
        level: Logging level
        use_json: Use JSON format (True) or plain text (False)
        
    Returns:
        Configured StreamHandler
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    if use_json:
        handler.setFormatter(XNAiJSONFormatter())
    else:
        # Plain text format for human readability
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
    
    Guide Reference: Section 5.2 (Logging Setup)
    
    This is the main entrypoint for configuring logging. Call once at
    application startup.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (default: from config)
        console_enabled: Enable console logging
        file_enabled: Enable file logging
        json_format: Use JSON format (vs plain text)
        
    Example:
        >>> setup_logging()
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
    """
    # Get configuration
    if log_level is None:
        log_level = get_config_value('logging.level', 'INFO')
    
    if log_file is None:
        log_file = get_config_value(
            'logging.file_path',
            '/app/XNAi_rag_app/logs/xnai.log'
        )
    
    # Parse log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Add console handler
    if console_enabled:
        console_handler = setup_console_handler(
            level=numeric_level,
            use_json=json_format
        )
        root_logger.addHandler(console_handler)
    
    # Add file handler
    if file_enabled:
        max_size_mb = get_config_value('logging.max_size_mb', 10)
        backup_count = get_config_value('logging.backup_count', 5)
        
        file_handler = setup_file_handler(
            log_file=log_file,
            max_bytes=max_size_mb * 1024 * 1024,
            backup_count=backup_count,
            level=numeric_level
        )
        root_logger.addHandler(file_handler)
    
    # Log initialization
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

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_logger(
    name: str,
    context: Dict[str, Any] = None
) -> logging.Logger:
    """
    Get configured logger with optional context.
    
    Guide Reference: Section 5.2 (Logger Creation)
    
    Args:
        name: Logger name (usually __name__)
        context: Context dict (request_id, user_id, etc.)
        
    Returns:
        Configured logger or ContextAdapter
        
    Example:
        >>> logger = get_logger(__name__, {'request_id': '123'})
        >>> logger.info("Processing request")
    """
    logger = logging.getLogger(name)
    
    if context:
        return ContextAdapter(logger, context)
    
    return logger

def log_startup_info():
    """
    Log application startup information.
    
    Guide Reference: Section 5.2 (Startup Logging)
    
    This logs critical configuration and system info at startup.
    """
    import psutil
    
    logger = logging.getLogger('xnai.startup')
    
    # Stack info
    logger.info(
        "Xoe-NovAi starting",
        extra={
            "stack_version": CONFIG['metadata']['stack_version'],
            "codename": CONFIG['metadata']['codename'],
            "phase": CONFIG['project']['phase'],
            "architecture": CONFIG['metadata']['architecture'],
        }
    )
    
    # System info
    memory = psutil.virtual_memory()
    logger.info(
        "System information",
        extra={
            "cpu_count": psutil.cpu_count(),
            "cpu_threads": CONFIG['performance']['cpu_threads'],
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
        }
    )
    
    # Configuration
    logger.info(
        "Configuration loaded",
        extra={
            "memory_limit_gb": CONFIG['performance']['memory_limit_gb'],
            "token_rate_target": CONFIG['performance']['token_rate_target'],
            "f16_kv_enabled": CONFIG['performance']['f16_kv_enabled'],
            "telemetry_enabled": CONFIG['project']['telemetry_enabled'],
        }
    )

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test logging configuration.
    
    Usage: python3 logging_config.py
    
    This validates the logging module and generates test logs.
    """
    print("=" * 70)
    print("Xoe-NovAi Logging Configuration - Test Suite v0.1.2")
    print("=" * 70)
    print()
    
    # Setup logging
    print("Setting up logging...")
    setup_logging(log_level='INFO', json_format=True)
    print("✓ Logging configured\n")
    
    # Test basic logging
    print("Test 1: Basic logging")
    logger = get_logger(__name__)
    logger.debug("Debug message (should not appear)")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    print("✓ Basic logging test complete\n")
    
    # Test context injection
    print("Test 2: Context injection")
    context_logger = get_logger(
        __name__,
        context={'request_id': 'test-123', 'user_id': 'user-456'}
    )
    context_logger.info("Message with context")
    print("✓ Context injection test complete\n")
    
    # Test performance logging
    print("Test 3: Performance logging")
    perf = PerformanceLogger(logger)
    perf.log_token_generation(tokens=100, duration_s=5.0)
    perf.log_memory_usage(component="test")
    perf.log_query_latency(query="test query", duration_ms=123.4, success=True)
    print("✓ Performance logging test complete\n")
    
    # Test crawler logging (NEW v0.1.2)
    print("Test 4: Crawler logging (NEW v0.1.2)")
    perf.log_crawl_operation(
        source="gutenberg",
        items=25,
        duration_s=1800,  # 30 minutes
        success=True
    )
    print("✓ Crawler logging test complete\n")
    
    # Test exception logging
    print("Test 5: Exception logging")
    try:
        raise ValueError("Test exception")
    except Exception:
        logger.exception("Exception occurred during test")
    print("✓ Exception logging test complete\n")
    
    # Log startup info
    print("Test 6: Startup info")
    log_startup_info()
    print("✓ Startup info test complete\n")
    
    # Check log file
    log_file = get_config_value('logging.file_path', '/app/XNAi_rag_app/logs/xnai.log')
    if Path(log_file).exists():
        print(f"✓ Log file created: {log_file}")
        print(f"  Size: {Path(log_file).stat().st_size} bytes")
    else:
        print(f"⚠  Log file not found: {log_file}")
    
    print()
    print("=" * 70)
    print("All logging tests passed!")
    print("=" * 70)
    print()
    print("Integration: Import with 'from logging_config import setup_logging'")
    print("Usage: Call setup_logging() once at application startup")(
            "Token generation completed",
            extra={
                "operation": "token_generation",
                "model": model,
                "tokens": tokens,
                "duration_s": round(duration_s, 3),
                "tokens_per_second": round(tokens_per_second, 2),
                "target_min": CONFIG['performance']['token_rate_min'],
                "target_max": CONFIG['performance']['token_rate_max'],
            }
        )
    
    def log_memory_usage(self, component: str = "system"):
        """
        Log current memory usage.
        
        Args:
            component: Component name (e.g., 'llm', 'embeddings', 'system')
        """
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
                "limit_gb": CONFIG['performance']['memory_limit_gb'],
            }
        )
    
    def log_query_latency(
        self,
        query: str,
        duration_ms: float,
        success: bool = True,
        error: str = None
    ):
        """
        Log query processing latency.
        
        Args:
            query: Query text (truncated to 100 chars)
            duration_ms: Latency in milliseconds
            success: Whether query succeeded
            error: Error message if failed
        """
        self.logger.info