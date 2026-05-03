"""Eval registry — discovers and registers evaluators."""

from dataclasses import dataclass
from typing import Callable
from agent_eval.core.types import Trace, Label, EvalResult


@dataclass
class Evaluator:
    name: str
    layer: str  # "pre_tool", "at_tool", "end_to_end"
    metric: str
    fn: Callable[[Trace, Label], EvalResult]


_registry: dict[str, Evaluator] = {}


def register(name: str, layer: str, metric: str):
    """Decorator to register an eval function."""

    def decorator(fn: Callable[[Trace, Label], EvalResult]):
        _registry[name] = Evaluator(
            name=name, layer=layer, metric=metric, fn=fn
        )
        return fn

    return decorator


def get_all() -> list[Evaluator]:
    return list(_registry.values())


def get_by_layer(layer: str) -> list[Evaluator]:
    return [e for e in _registry.values() if e.layer == layer]
