# strategies/__init__.py
import importlib
import os

from strategies.base_strategy import BaseStrategy

# Dynamic strategy discovery
STRATEGY_MAP = {}

# Iterate through all files in the strategies directory
for filename in os.listdir(os.path.dirname(__file__)):
    if filename.endswith("_strategy.py") and filename != "__init__.py":
        module_name = filename[:-3]  # Remove .py extension
        try:
            # Import the module
            module = importlib.import_module(f"strategies.{module_name}")
            # Find all classes in the module that inherit from BaseStrategy
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseStrategy)
                    and attr != BaseStrategy
                ):
                    STRATEGY_MAP[attr.__name__] = attr
        except ImportError:
            # Log or handle import errors if needed
            pass
