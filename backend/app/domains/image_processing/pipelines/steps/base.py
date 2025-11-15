"""
Pipeline Step Abstractions.

Purpose:
    Provide a pluggable, composable step interface for the image processing pipeline.
    This enables future extension (e.g. custom steps, conditional execution, metrics collection)
    without modifying the central orchestrator (service.py).

Key Concepts:
    ProcessingContext  - Shared mutable state passed between steps (defined in service.py).
    PipelineStep       - Abstract base class defining lifecycle & execution contract.
    StepResult         - Structured optional output (metadata, timings, diagnostics).
    StepRegistry       - Lightweight registry for named step lookups & future dynamic loading.

Design Principles:
    - Steps are async (to allow future concurrency / I/O bound operations).
    - Each step declares the ProcessingStep enum value(s) it implements.
    - Clear separation of step metadata (name, description) from execution logic.
    - Non-failing philosophy: steps should catch & log internal errors and
      surface them in StepResult.errors (unless failure must abort pipeline).
    - Timings captured for performance profiling.

Usage Example (creating a new step):
    class MyCustomStep(PipelineStep):
        name = "my_custom"
        description = "Demonstration custom step"
        provides = [ProcessingStep.TEXT_DETECTION]

        async def run(self, ctx: ProcessingContext) -> StepResult:
            start = time.perf_counter()
            try:
                # ... mutate ctx or compute derived artifacts ...
                return StepResult.success(duration=time.perf_counter() - start, metadata={"info": "ok"})
            except Exception as e:
                logger.exception("[MyCustomStep] Failed")
                return StepResult.failure(duration=time.perf_counter() - start, errors=[str(e)])

Future Enhancements (not implemented now to keep refactor scope stable):
    - Dependency graph between steps (topological ordering).
    - Conditional execution based on ctx state.
    - Metrics exporter hook (Prometheus / OpenTelemetry).
    - Cancellation / timeout handling.
    - Automatic retry wrapper for transient failures.

"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Callable, Any

from app.api.schemas.pipeline import ProcessingStep
from app.domains.image_processing.service import ProcessingContext

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """
    Represents the outcome of executing a pipeline step.

    Attributes:
        success: Whether the step completed successfully.
        duration: Execution duration in seconds.
        metadata: Arbitrary step-specific data (e.g., counts, dimensions).
        warnings: Non-fatal issues encountered.
        errors: Fatal or captured exceptions (if any).
    """
    success: bool
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @classmethod
    def success(cls, duration: float, metadata: Dict[str, Any] | None = None, warnings: List[str] | None = None) -> "StepResult":
        return cls(
            success=True,
            duration=duration,
            metadata=metadata or {},
            warnings=warnings or [],
            errors=[],
        )

    @classmethod
    def failure(cls, duration: float, errors: List[str], metadata: Dict[str, Any] | None = None, warnings: List[str] | None = None) -> "StepResult":
        return cls(
            success=False,
            duration=duration,
            metadata=metadata or {},
            warnings=warnings or [],
            errors=errors,
        )


class PipelineStep:
    """
    Abstract pipeline step contract.

    Subclasses MUST:
        - Set 'name' (unique within registry)
        - Optionally set 'description'
        - Optionally set 'provides' to one or more ProcessingStep enum values
        - Implement async run(ctx) -> StepResult

    Subclasses MAY override:
        - prepare(ctx): Pre-execution hook
        - cleanup(ctx, result): Post-execution hook
        - should_run(ctx): Conditional execution (default always True)
    """
    name: str = "unnamed_step"
    description: str = "No description provided."
    provides: List[ProcessingStep] = []

    async def run(self, ctx: ProcessingContext) -> StepResult:  # pragma: no cover - abstract
        raise NotImplementedError("PipelineStep subclasses must implement run()")

    async def execute(self, ctx: ProcessingContext) -> StepResult:
        """
        Execute the step lifecycle: prepare -> run -> cleanup.

        Catches exceptions from run() and wraps them into a failure StepResult.
        """
        if not self.should_run(ctx):
            logger.info("[Step:%s] Skipping (should_run returned False)", self.name)
            return StepResult.success(duration=0.0, metadata={"skipped": True})

        self.prepare(ctx)
        start = time.perf_counter()
        try:
            result = await self.run(ctx)
        except Exception as e:  # noqa: BLE001
            logger.exception("[Step:%s] Execution failed", self.name)
            result = StepResult.failure(
                duration=time.perf_counter() - start,
                errors=[str(e)],
            )
        finally:
            try:
                self.cleanup(ctx, result)
            except Exception as cleanup_err:  # noqa: BLE001
                logger.warning("[Step:%s] Cleanup error: %s", self.name, cleanup_err)
                if result.success:
                    # Convert success to partial failure with cleanup warning
                    result.warnings.append(f"Cleanup error: {cleanup_err}")

        logger.info(
            "[Step:%s] Completed (success=%s, duration=%.3fs)",
            self.name,
            result.success,
            result.duration,
        )
        return result

    def prepare(self, ctx: ProcessingContext) -> None:
        """Hook for pre-run preparation (default: no-op)."""
        return None

    def cleanup(self, ctx: ProcessingContext, result: StepResult) -> None:
        """Hook for post-run cleanup (default: no-op)."""
        return None

    def should_run(self, ctx: ProcessingContext) -> bool:
        """Predicate to determine conditional execution (default: always True)."""
        return True


class StepRegistry:
    """
    Lightweight registry for pipeline steps by name.

    Supports:
        - Explicit registration
        - Retrieval by name
        - Bulk instantiation
        - Future dynamic loading (entrypoints / config-driven)

    Thread-safety: Not implemented (single-threaded FastAPI expected).
    """
    def __init__(self):
        self._registry: Dict[str, Type[PipelineStep]] = {}

    def register(self, step_cls: Type[PipelineStep]) -> None:
        if not issubclass(step_cls, PipelineStep):
            raise TypeError("Registered class must inherit from PipelineStep")
        name = step_cls.name
        if not name:
            raise ValueError("PipelineStep subclass must define a non-empty 'name'")
        if name in self._registry:
            logger.warning("[StepRegistry] Overwriting existing step: %s", name)
        self._registry[name] = step_cls
        logger.debug("[StepRegistry] Registered step: %s", name)

    def create(self, name: str) -> PipelineStep:
        if name not in self._registry:
            raise KeyError(f"Step '{name}' not found in registry")
        return self._registry[name]()

    def list_steps(self) -> List[str]:
        return list(self._registry.keys())

    def get(self, name: str) -> Optional[Type[PipelineStep]]:
        return self._registry.get(name)

    def create_all(self) -> List[PipelineStep]:
        return [cls() for cls in self._registry.values()]


# Global default registry instance (optional usage)
default_step_registry = StepRegistry()


def register_step(step_cls: Type[PipelineStep]) -> Type[PipelineStep]:
    """
    Decorator to auto-register a PipelineStep subclass in default registry.

    Example:
        @register_step
        class MyStep(PipelineStep):
            name = "my_step"
            async def run(self, ctx): ...

    Returns:
        The original class (for decorator chaining).
    """
    default_step_registry.register(step_cls)
    return step_cls


__all__ = [
    "PipelineStep",
    "ProcessingContext",
    "StepResult",
    "StepRegistry",
    "default_step_registry",
    "register_step",
]
