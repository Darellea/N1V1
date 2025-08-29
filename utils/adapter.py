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
            return dataclasses.asdict(signal)
    except Exception:
        # dataclasses may not be importable in some contexts; continue
        pass

    # to_dict() protocol
    try:
        if hasattr(signal, "to_dict") and callable(getattr(signal, "to_dict")):
            return signal.to_dict()
    except Exception:
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
        try:
            if hasattr(signal, a):
                val = getattr(signal, a)
                # Normalize enums (e.g., OrderType) to value or name
                try:
                    if hasattr(val, "value"):
                        val = val.value
                    elif hasattr(val, "name"):
                        val = val.name.lower()
                except Exception:
                    pass
                out[a] = val
        except Exception:
            continue

    # As a fallback, include any public attributes from __dict__
    try:
        if hasattr(signal, "__dict__"):
            for k, v in signal.__dict__.items():
                if k not in out:
                    out[k] = v
    except Exception:
        pass

    return out
