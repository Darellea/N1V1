"""
optimization/rl_optimizer.py

Reinforcement Learning implementation for strategy selection optimization.
Uses Q-learning to learn optimal strategy selection based on market conditions.
"""

from typing import Dict, List, Any, Optional, Tuple
import ast
import logging
import time
import random
import json
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import numpy as np

from .base_optimizer import BaseOptimizer


@dataclass
class MarketState:
    """Represents the current market state for RL."""

    trend_strength: float  # -1 to 1 (negative = downtrend, positive = uptrend)
    volatility_regime: str  # 'low', 'medium', 'high'
    volume_regime: str     # 'low', 'normal', 'high'
    momentum: float        # -1 to 1 (negative = bearish, positive = bullish)

    def to_tuple(self) -> Tuple:
        """Convert state to hashable tuple for Q-table."""
        return (
            round(self.trend_strength, 2),
            self.volatility_regime,
            self.volume_regime,
            round(self.momentum, 2)
        )

    @classmethod
    def from_data(cls, data: pd.DataFrame) -> 'MarketState':
        """
        Create market state from OHLCV data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            MarketState instance
        """
        if len(data) < 20:
            return cls(0.0, 'medium', 'normal', 0.0)

        # Calculate trend strength using moving averages
        sma_short = data['close'].rolling(10).mean()
        sma_long = data['close'].rolling(20).mean()
        trend_strength = (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
        trend_strength = max(-1.0, min(1.0, trend_strength))

        # Determine volatility regime
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility

        if volatility < 0.3:  # 30%
            vol_regime = 'low'
        elif volatility > 0.7:  # 70%
            vol_regime = 'high'
        else:
            vol_regime = 'medium'

        # Determine volume regime
        avg_volume = data['volume'].mean()
        current_volume = data['volume'].iloc[-1]

        if current_volume < avg_volume * 0.7:
            volume_regime = 'low'
        elif current_volume > avg_volume * 1.3:
            volume_regime = 'high'
        else:
            volume_regime = 'normal'

        # Calculate momentum
        momentum = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10]
        momentum = max(-1.0, min(1.0, momentum))

        return cls(trend_strength, vol_regime, volume_regime, momentum)


class RLOptimizer(BaseOptimizer):
    """
    Reinforcement Learning optimizer for strategy selection.

    This optimizer:
    1. Defines market states based on trend, volatility, and volume
    2. Uses strategies as actions in the RL environment
    3. Learns Q-values for state-action pairs
    4. Maximizes cumulative reward (PnL, Sharpe ratio, etc.)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RL Optimizer.

        Args:
            config: Configuration dictionary containing:
                - alpha: Learning rate
                - gamma: Discount factor
                - epsilon: Exploration rate
                - episodes: Number of training episodes
                - max_steps_per_episode: Maximum steps per episode
                - reward_function: How to calculate rewards
        """
        super().__init__(config)

        # RL specific configuration
        self.alpha = config.get('alpha', 0.1)  # Learning rate
        self.gamma = config.get('gamma', 0.95)  # Discount factor
        self.epsilon = config.get('epsilon', 0.1)  # Exploration rate
        self.episodes = config.get('episodes', 100)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 50)
        self.reward_function = config.get('reward_function', 'sharpe_ratio')

        # Q-learning state
        self.q_table: Dict[Tuple, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.strategy_actions: List[str] = []

        # Training state
        self.current_episode = 0
        self.total_steps = 0

    def optimize(self, strategy_class, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run reinforcement learning optimization.

        Args:
            strategy_class: Strategy class to optimize (not used directly)
            data: Historical data for optimization

        Returns:
            Best strategy selection policy
        """
        start_time = time.time()

        self.logger.info("Starting Reinforcement Learning Optimization")
        self.logger.info(f"Episodes: {self.episodes}")
        self.logger.info(f"Alpha: {self.alpha}, Gamma: {self.gamma}, Epsilon: {self.epsilon}")

        # Initialize available strategies
        self._initialize_strategy_actions()

        # Training loop
        for episode in range(self.episodes):
            self.current_episode = episode + 1

            self.logger.info(f"Episode {episode + 1}/{self.episodes}")

            # Run one episode
            episode_reward = self._run_episode(data)

            # Log episode results
            self.logger.info(f"Episode {episode + 1} completed with total reward: {episode_reward:.4f}")

            # Decay exploration rate
            self.epsilon = max(0.01, self.epsilon * 0.995)

        # Finalize optimization
        optimization_time = time.time() - start_time
        self.config['optimization_time'] = optimization_time

        self.logger.info(f"RL Optimization completed in {optimization_time:.2f}s")
        self.logger.info(f"Total episodes: {self.episodes}")
        self.logger.info(f"Total steps: {self.total_steps}")

        # Return the learned policy
        return self.get_learned_policy()

    def _initialize_strategy_actions(self) -> None:
        """
        Initialize available strategy actions.
        In a real implementation, this would be configurable.
        """
        # For demonstration, we'll use predefined strategy types
        self.strategy_actions = [
            'trend_following',
            'mean_reversion',
            'breakout',
            'scalping',
            'swing'
        ]

        self.logger.info(f"Initialized {len(self.strategy_actions)} strategy actions: {self.strategy_actions}")

    def _run_episode(self, data: pd.DataFrame) -> float:
        """
        Run a single training episode.

        Args:
            data: Historical data for the episode

        Returns:
            Total reward for the episode
        """
        total_reward = 0.0
        steps = 0

        # Start from a random position in the data
        current_idx = random.randint(100, len(data) - self.max_steps_per_episode - 100)

        while steps < self.max_steps_per_episode and current_idx < len(data) - 50:
            # Get current market state
            window_data = data.iloc[current_idx:current_idx + 50]
            current_state = MarketState.from_data(window_data)

            # Choose action (strategy) using epsilon-greedy policy
            action = self._choose_action(current_state)

            # Execute action and get reward
            reward, next_idx = self._execute_action(action, data, current_idx)

            # Get next state
            if next_idx < len(data) - 50:
                next_window = data.iloc[next_idx:next_idx + 50]
                next_state = MarketState.from_data(next_window)
            else:
                next_state = None

            # Update Q-table
            self._update_q_table(current_state, action, reward, next_state)

            # Move to next state
            current_idx = next_idx
            total_reward += reward
            steps += 1
            self.total_steps += 1

        return total_reward

    def _choose_action(self, state: MarketState) -> str:
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: Current market state

        Returns:
            Selected action (strategy name)
        """
        state_tuple = state.to_tuple()

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(self.strategy_actions)

        # Exploitation - choose best action for current state
        if state_tuple in self.q_table:
            action_values = self.q_table[state_tuple]
            if action_values:
                return max(action_values, key=action_values.get)

        # If no Q-values exist, choose randomly
        return random.choice(self.strategy_actions)

    def _execute_action(self, action: str, data: pd.DataFrame, current_idx: int) -> Tuple[float, int]:
        """
        Execute an action and calculate reward.

        Args:
            action: Strategy action to execute
            data: Historical data
            current_idx: Current position in data

        Returns:
            Tuple of (reward, next_index)
        """
        # Define action-specific parameters
        strategy_params = self._get_strategy_params_for_action(action)

        # Simulate strategy execution on next segment
        segment_length = 20  # Look ahead 20 periods
        next_idx = min(current_idx + segment_length, len(data) - 1)

        segment_data = data.iloc[current_idx:next_idx]

        # Calculate reward based on simulated performance
        reward = self._calculate_action_reward(action, segment_data, strategy_params)

        return reward, next_idx

    def _get_strategy_params_for_action(self, action: str) -> Dict[str, Any]:
        """
        Get strategy parameters for a given action.

        Args:
            action: Strategy action name

        Returns:
            Strategy parameters dictionary
        """
        # Action-specific parameter mappings
        action_params = {
            'trend_following': {
                'rsi_period': 14,
                'ema_fast': 9,
                'ema_slow': 21,
                'adx_threshold': 25
            },
            'mean_reversion': {
                'rsi_period': 14,
                'bollinger_period': 20,
                'bollinger_std': 2.0,
                'oversold': 30,
                'overbought': 70
            },
            'breakout': {
                'lookback_period': 20,
                'breakout_threshold': 0.02,
                'volume_multiplier': 1.5
            },
            'scalping': {
                'fast_period': 5,
                'slow_period': 10,
                'stop_loss_pct': 0.005,
                'take_profit_pct': 0.01
            },
            'swing': {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            }
        }

        return action_params.get(action, {})

    def _calculate_action_reward(self, action: str, data: pd.DataFrame,
                               params: Dict[str, Any]) -> float:
        """
        Calculate reward for executing an action.

        Args:
            action: Strategy action executed
            data: Data segment for evaluation
            params: Strategy parameters

        Returns:
            Reward value
        """
        if len(data) < 10:
            return 0.0

        try:
            # Simple reward calculation based on price movement and action type
            start_price = data['close'].iloc[0]
            end_price = data['close'].iloc[-1]
            price_return = (end_price - start_price) / start_price

            # Action-specific reward modifiers
            action_modifier = {
                'trend_following': 1.0,  # Standard reward
                'mean_reversion': 1.2,  # Bonus for mean reversion in ranging markets
                'breakout': 1.5,        # Bonus for breakout strategies
                'scalping': 0.8,        # Penalty for frequent trading
                'swing': 1.1           # Slight bonus for swing trading
            }.get(action, 1.0)

            # Volatility adjustment
            volatility = data['close'].pct_change().std()
            vol_modifier = 1.0 - (volatility * 2)  # Reduce reward in high volatility

            # Volume confirmation
            avg_volume = data['volume'].mean()
            volume_modifier = 1.0 if avg_volume > data['volume'].quantile(0.5) else 0.8

            base_reward = price_return * action_modifier * vol_modifier * volume_modifier

            # Add Sharpe-like component
            returns = data['close'].pct_change().dropna()
            if len(returns) > 1 and returns.std() > 0:
                sharpe_component = returns.mean() / returns.std() * np.sqrt(252)
                base_reward += sharpe_component * 0.1  # Small weight

            return base_reward

        except Exception as e:
            self.logger.debug(f"Reward calculation failed: {str(e)}")
            return 0.0

    def _update_q_table(self, state: MarketState, action: str,
                       reward: float, next_state: Optional[MarketState]) -> None:
        """
        Update Q-table using Q-learning update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state (None if terminal)
        """
        state_tuple = state.to_tuple()

        # Get current Q-value
        current_q = self.q_table[state_tuple][action]

        # Calculate max Q-value for next state
        if next_state is not None:
            next_state_tuple = next_state.to_tuple()
            next_q_values = self.q_table[next_state_tuple]
            max_next_q = max(next_q_values.values()) if next_q_values else 0.0
        else:
            max_next_q = 0.0

        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

        # Update Q-table
        self.q_table[state_tuple][action] = new_q

    def get_learned_policy(self) -> Dict[str, Any]:
        """
        Get the learned policy from Q-table.

        Returns:
            Policy dictionary mapping states to best actions
        """
        policy = {}

        for state_tuple, actions in self.q_table.items():
            if actions:
                best_action = max(actions, key=actions.get)
                best_q_value = actions[best_action]

                policy[str(state_tuple)] = {
                    'best_action': best_action,
                    'q_value': best_q_value,
                    'all_actions': dict(actions)
                }

        return {
            'policy': policy,
            'total_states_learned': len(policy),
            'total_state_action_pairs': sum(len(actions) for actions in self.q_table.values()),
            'strategy_actions': self.strategy_actions
        }

    def predict_action(self, market_data: pd.DataFrame) -> str:
        """
        Predict the best action for current market conditions.

        Args:
            market_data: Current market data

        Returns:
            Recommended strategy action
        """
        current_state = MarketState.from_data(market_data)
        state_tuple = current_state.to_tuple()

        # Get Q-values for current state
        if state_tuple in self.q_table:
            action_values = self.q_table[state_tuple]
            if action_values:
                return max(action_values, key=action_values.get)

        # Fallback to random action if state not seen
        return random.choice(self.strategy_actions)

    def save_policy(self, filepath: str) -> None:
        """
        Save the learned policy to file.

        Args:
            filepath: Path to save policy
        """
        # Convert tuple keys to strings for JSON serialization
        q_table_serializable = {}
        for state_tuple, actions in self.q_table.items():
            state_str = str(state_tuple)
            q_table_serializable[state_str] = dict(actions)

        policy_data = {
            'q_table': q_table_serializable,
            'strategy_actions': self.strategy_actions,
            'config': self.config,
            'total_episodes': self.episodes,
            'total_steps': self.total_steps
        }

        with open(filepath, 'w') as f:
            json.dump(policy_data, f, indent=2)

        self.logger.info(f"Policy saved to {filepath}")

    def load_policy(self, filepath: str) -> None:
        """
        Load a learned policy from file.

        Args:
            filepath: Path to load policy from
        """
        with open(filepath, 'r') as f:
            policy_data = json.load(f)

        # Restore Q-table
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state_str, actions in policy_data.get('q_table', {}).items():
            # Convert string keys back to tuples if needed
            try:
                state_tuple = ast.literal_eval(state_str) if isinstance(state_str, str) else state_str
            except (ValueError, SyntaxError):
                state_tuple = state_str
            self.q_table[state_tuple] = defaultdict(float, actions)

        self.strategy_actions = policy_data.get('strategy_actions', [])
        self.logger.info(f"Policy loaded from {filepath}")

    def get_rl_summary(self) -> Dict[str, Any]:
        """
        Get summary of RL optimization process.

        Returns:
            Summary dictionary
        """
        summary = self.get_optimization_summary()
        summary.update({
            'episodes': self.episodes,
            'total_steps': self.total_steps,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'final_epsilon': self.epsilon,
            'states_learned': len(self.q_table),
            'strategy_actions': self.strategy_actions,
            'q_table_size': sum(len(actions) for actions in self.q_table.values())
        })

        return summary
