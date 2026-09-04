"""Tool guard: failure injection, failure detection, retry classification.

Every tool call is wrapped so that experiments can deterministically inject
failures at a controlled rate, and so that both crash failures (exceptions)
and silent failures (malformed output) are detected and classified.

Deterministic seeding: the (run_seed, tool, call_index) tuple drives a PRNG so
experiments are exactly reproducible.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Any


class ToolError(Exception):
    def __init__(self, message: str, classification: str = "fatal"):
        super().__init__(message)
        self.classification = classification  # "transient" | "fatal"


@dataclass
class GuardStats:
    calls: int = 0
    injected: int = 0
    detected: int = 0
    retries: int = 0
    escalated: int = 0
    history: list = field(default_factory=list)


class ToolGuard:
    """Wraps a tool callable.

    failure_rate: probability per *call attempt* of an injected fault.
    fault_type: "crash" (raise TimeoutError, transient), "malformed"
        (return schema-invalid output; silent failure), or "mixed".
    validator: callable(result) -> bool; False means the tool output is
        malformed (silent failure detected).
    max_retries: retry attempts for transient faults (crash) before
        escalation; malformed output is always retried once, then escalated.
    """

    def __init__(self, tool_fn: Callable, name: str, validator: Callable[[Any], bool] | None = None,
                 failure_rate: float = 0.0, fault_type: str = "mixed",
                 max_retries: int = 2, seed: int = 0, audit=None):
        self.fn = tool_fn
        self.name = name
        self.validator = validator
        self.failure_rate = failure_rate
        self.fault_type = fault_type
        self.max_retries = max_retries
        self.rng = random.Random(seed)
        self.seed = seed
        self.stats = GuardStats()
        self.audit = audit

    def _draw_fault(self) -> str | None:
        if self.failure_rate <= 0:
            return None
        if self.rng.random() < self.failure_rate:
            if self.fault_type == "mixed":
                return "malformed" if self.rng.random() < 0.5 else "crash"
            return self.fault_type
        return None

    def call(self, *args, **kwargs):
        """Execute with failure detection. Returns result or raises ToolError.

        All attempts, injected faults, and detected anomalies are recorded in
        stats (and the audit log, if attached).
        """
        attempt = 0
        while True:
            attempt += 1
            self.stats.calls += 1
            fault = self._draw_fault()
            try:
                if fault == "crash":
                    self.stats.injected += 1
                    if self.audit:
                        self.audit.tool_failure(self.name, {"attempt": attempt},
                                                "injected timeout", "transient", attempt, injected=True)
                    raise TimeoutError(f"simulated upstream timeout in {self.name}")
                result = self.fn(*args, **kwargs)
                if fault == "malformed":
                    self.stats.injected += 1
                    result = _corrupt(result)
                if self.validator is not None and not self.validator(result):
                    self.stats.detected += 1
                    if self.audit:
                        self.audit.tool_failure(self.name, {"attempt": attempt},
                                                "malformed output detected", "transient", attempt, injected=(fault == "malformed"))
                    if attempt <= self.max_retries:
                        self.stats.retries += 1
                        continue
                    self.stats.escalated += 1
                    raise ToolError(f"{self.name}: malformed output after retries", "fatal")
                if self.audit:
                    self.audit.tool_call(self.name, {"attempt": attempt}, status="ok")
                return result
            except TimeoutError as e:
                if attempt <= self.max_retries:
                    self.stats.retries += 1
                    continue
                self.stats.escalated += 1
                raise ToolError(f"{self.name}: {e}", "fatal")
            except ToolError:
                raise
            except Exception as e:  # genuine unexpected fault: classify conservatively
                self.stats.detected += 1
                if self.audit:
                    self.audit.tool_failure(self.name, {"attempt": attempt}, repr(e), "transient", attempt)
                if attempt <= self.max_retries:
                    self.stats.retries += 1
                    continue
                self.stats.escalated += 1
                raise ToolError(f"{self.name}: unexpected fault {e!r}", "fatal")


def _corrupt(result):
    """Return a schema-invalid version of the result (simulates silent bug)."""
    if isinstance(result, list):
        return [r for r in result if r is None] or [None]
    if isinstance(result, dict):
        return {k: None for k in result}
    return None
