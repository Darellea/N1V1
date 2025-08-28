# strategies/__init__.py
from strategies.base_strategy import BaseStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.ema_cross_strategy import EMACrossStrategy

# Define strategy mapping
STRATEGY_MAP = {
    "RSIStrategy": RSIStrategy,
    "EMACrossStrategy": EMACrossStrategy,
}
