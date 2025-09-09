"""
utils/adapter.py

Small adapter utilities to normalize inputs (e.g. TradingSignal dataclass/objects)
into plain dictionaries for loggers, order manager, and notifier code paths.
"""

from typing import Any, Dict
import dataclasses


def signal_to_dict(signal: Any) -> Dict[str, Any]:
    """
    Normalize a trading signal object into a plain dict.

    Supports:
    - dict -> shallow copy
    - dataclass -> dataclasses.asdict
    - objects with to_dict() -> call it
    - generic objects -> probe common attributes and __dict__

    Returns an empty dict for None input.
    """
    if signal is None:
        return {}

    # If already a dict, return a shallow copy
    if isinstance(signal, dict):
        return dict(signal)

    # Dataclass support
    try:
        if dataclasses.is_dataclass(signal):
            result = dataclasses.asdict(signal)
            # Convert enums to their values/names for JSON serialization
            for key, value in result.items():
                if hasattr(value, "value"):
                    result[key] = value.value
                elif hasattr(value, "name"):
                    result[key] = value.name.lower()
            return result
    except (AttributeError, TypeError, ValueError):
        # If dataclass conversion fails for known reasons, fall back to other methods.
        pass

    # to_dict() protocol
    try:
        to_dict = getattr(signal, "to_dict", None)
        if callable(to_dict):
            return to_dict()
    except (TypeError, AttributeError, ValueError):
        # to_dict() may raise if implementation is buggy; fall back to probing attributes.
        pass

    out: Dict[str, Any] = {}

    # Probe common signal attributes
    common_attrs = (
        "id",
        "symbol",
        "order_type",
        "side",
        "amount",
        "price",
        "params",
        "stop_loss",
        "take_profit",
        "trailing_stop",
        "timestamp",
    )

    for a in common_attrs:
        # Use getattr with default to avoid KeyError; properties may raise AttributeError,
        # so catch AttributeError at the minimal scope.
        try:
            if hasattr(signal, a):
                val = getattr(signal, a)
            else:
                continue
        except AttributeError:
            # If property access raises, skip that attribute.
            continue

        # Normalize enums (e.g., OrderType) to value or name
        try:
            if hasattr(val, "value"):
                val = val.value
            elif hasattr(val, "name"):
                # some enums expose .name; convert to lower-case string for readability
                val = val.name.lower()
        except AttributeError:
            # If enum-like object doesn't expose expected attributes, ignore and continue.
            pass

        out[a] = val

    # As a fallback, include any public attributes from __dict__ (non-callable)
    try:
        obj_dict = getattr(signal, "__dict__", None)
        if isinstance(obj_dict, dict):
            for k, v in obj_dict.items():
                if k not in out and not k.startswith("_"):
                    out[k] = v
    except AttributeError:
        # If accessing __dict__ raises, give up on this fallback.
        pass

    return out
