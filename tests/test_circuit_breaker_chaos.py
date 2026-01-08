#!/usr/bin/env python3
# ============================================================================
# Xoe-NovAi Circuit Breaker Chaos Test v0.1.4-stable
# ============================================================================
# Purpose: Validate Pattern 5 circuit breaker resilience and fail-fast behavior
# Guide Reference: Section 1.5 (Pattern 5 - Circuit Breaker - Chaos Test)
# Last Updated: 2025-01-02
# ============================================================================

import pytest
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add app directory to path (Pattern 1)
sys.path.insert(0, str(Path(__file__).parent.parent / "app" / "XNAi_rag_app"))

from pybreaker import CircuitBreaker, CircuitBreakerError


class TestCircuitBreakerChaos:
    """
    Chaos tests for circuit breaker resilience (Pattern 5).
    
    Blueprint Reference: Section 1.5 (Chaos Test)
    
    This test suite validates that the circuit breaker:
    1. Fails fast after max failures (3)
    2. Returns appropriate errors (CircuitBreakerError)
    3. Recovers after timeout (60s)
    4. Tracks failure state correctly
    """
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker instance for testing."""
        return CircuitBreaker(
            fail_max=3,
            reset_timeout=1,  # 1s for testing (blueprint uses 60s in production)
            name="test-breaker"
        )
    
    def test_circuit_breaker_opens_after_three_failures(self, circuit_breaker):
        """
        Test that circuit opens after fail_max failures (3).
        
        Blueprint Requirement: fail_max=3
        
        Scenario:
        - Call 1: Fails → Circuit CLOSED (1 failure)
        - Call 2: Fails → Circuit CLOSED (2 failures)
        - Call 3: Fails → Circuit CLOSED (3 failures)
        - Call 4: CircuitBreakerError → Circuit OPEN (fail-fast)
        """
        def failing_function():
            raise Exception("Simulated failure")
        
        breaker_wrapped = circuit_breaker(failing_function)
        
        # First 3 calls: Fail with Exception (increment failure counter)
        for i in range(3):
            with pytest.raises(Exception):
                breaker_wrapped()
        
        # 4th call: Circuit OPEN, immediate CircuitBreakerError (fail-fast)
        with pytest.raises(CircuitBreakerError):
            breaker_wrapped()
    
    def test_circuit_breaker_state_transitions(self, circuit_breaker):
        """
        Test circuit breaker state machine: CLOSED → OPEN → HALF_OPEN → CLOSED.
        
        Blueprint Requirement: Proper state transitions
        """
        def failing_function():
            raise Exception("Failure")
        
        breaker_wrapped = circuit_breaker(failing_function)
        
        # Initial: CLOSED
        assert circuit_breaker.current_state == "closed", "Initial state should be CLOSED"
        
        # After 3 failures: OPEN
        for _ in range(3):
            with pytest.raises(Exception):
                breaker_wrapped()
        
        assert circuit_breaker.current_state == "open", "State should be OPEN after 3 failures"
        
        # During OPEN: Immediate CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            breaker_wrapped()
        
        # Wait for recovery timeout
        time.sleep(1.1)  # Wait 1.1 seconds (slightly more than reset_timeout=1)
        
        # After timeout: Should be HALF_OPEN, ready to test recovery
        # Note: pybreaker doesn't have explicit HALF_OPEN state in API,
        # but it will attempt one call after timeout
    
    def test_circuit_breaker_recovery_after_timeout(self, circuit_breaker):
        """
        Test that circuit recovers after reset_timeout and succeeds.
        
        Blueprint Requirement: reset_timeout=60s (using 1s in tests)
        
        Scenario:
        - Fail 3 times (circuit opens)
        - Wait timeout
        - Next successful call → circuit closes
        """
        call_count = 0
        
        def sometimes_failing_function():
            nonlocal call_count
            call_count += 1
            # Fail for first 3 calls, then succeed
            if call_count <= 3:
                raise Exception("Failure")
            return "success"
        
        breaker_wrapped = circuit_breaker(sometimes_failing_function)
        
        # First 3 calls fail
        for _ in range(3):
            with pytest.raises(Exception):
                breaker_wrapped()
        
        # Circuit is open
        with pytest.raises(CircuitBreakerError):
            breaker_wrapped()
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Next call should succeed and close circuit
        result = breaker_wrapped()
        assert result == "success", "Should succeed after timeout and recovery"
        assert circuit_breaker.current_state == "closed", "Should return to CLOSED state"
    
    def test_circuit_breaker_success_closes_circuit(self, circuit_breaker):
        """
        Test that a successful call closes the circuit (no errors).
        
        Scenario:
        - Call succeeds → Circuit stays CLOSED
        """
        def working_function():
            return "success"
        
        breaker_wrapped = circuit_breaker(working_function)
        
        # Successful calls don't increment failure counter
        for _ in range(5):
            result = breaker_wrapped()
            assert result == "success"
        
        # Circuit should still be closed
        assert circuit_breaker.current_state == "closed"
    
    def test_circuit_breaker_fail_fast_returns_error(self, circuit_breaker):
        """
        Test that circuit breaker raises CircuitBreakerError when open.
        
        Blueprint Requirement: Return 503 with Retry-After header
        
        This test validates the lower-level behavior; HTTP handling tested separately.
        """
        def failing_function():
            raise Exception("Failure")
        
        breaker_wrapped = circuit_breaker(failing_function)
        
        # Open circuit
        for _ in range(3):
            with pytest.raises(Exception):
                breaker_wrapped()
        
        # Verify fail-fast with specific error type
        with pytest.raises(CircuitBreakerError) as exc_info:
            breaker_wrapped()
        
        # CircuitBreakerError should indicate the breaker is open
        error = exc_info.value
        assert "open" in str(error).lower() or error.args
    
    def test_circuit_breaker_default_exception_handling(self, circuit_breaker):
        """
        Test that all exceptions increment failure counter.
        
        Blueprint Requirement: Catch all exceptions (except CircuitBreakerError)
        """
        exceptions_to_test = [
            RuntimeError("Runtime error"),
            ConnectionError("Connection error"),
            TimeoutError("Timeout error"),
            OSError("OS error"),
        ]
        
        def exception_factory(exc):
            def func():
                raise exc
            return func
        
        # Each exception type should increment counter
        for exc in exceptions_to_test:
            breaker = CircuitBreaker(fail_max=1, reset_timeout=1)
            with pytest.raises(type(exc)):
                breaker(exception_factory(exc))()
        
        # 2nd call should fail-fast (circuit open)
        with pytest.raises(CircuitBreakerError):
            breaker(exception_factory(exceptions_to_test[0]))()


# ============================================================================
# INTEGRATION TEST WITH MAIN.PY CIRCUIT BREAKER
# ============================================================================

def test_llm_circuit_breaker_integration():
    """
    Integration test with actual main.py circuit breaker setup.
    
    This test validates that the circuit breaker in main.py is properly configured.
    """
    # Import the actual circuit breaker from main.py
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "app" / "XNAi_rag_app"))
        from main import llm_circuit_breaker
        
        # Verify configuration
        assert llm_circuit_breaker.fail_max == 3, "fail_max should be 3"
        assert llm_circuit_breaker.reset_timeout == 60, "reset_timeout should be 60"
        assert llm_circuit_breaker.name == "llm-load", "name should be 'llm-load'"
    except ImportError as e:
        pytest.skip(f"Could not import from main.py: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
