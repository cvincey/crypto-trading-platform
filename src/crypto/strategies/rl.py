"""Reinforcement Learning trading strategies."""

import logging
from typing import Any

import numpy as np
import pandas as pd

from crypto.core.types import Signal
from crypto.indicators.base import indicator_registry
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)

# Check if RL dependencies are available
RL_AVAILABLE = False
gym = None
spaces = None
DQN = None
PPO = None
A2C = None
DummyVecEnv = None
TradingEnv = None

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_AVAILABLE = True
except ImportError:
    logger.warning(
        "RL dependencies not installed. Install with: "
        "pip install stable-baselines3 gymnasium"
    )


def _create_trading_env_class():
    """Factory function to create TradingEnv class when dependencies are available."""
    if not RL_AVAILABLE:
        return None
    
    class TradingEnv(gym.Env):
        """
        Custom Gym environment for trading.
        
        Observation: Feature vector (technical indicators)
        Action: 0 = HOLD, 1 = BUY, 2 = SELL
        Reward: Based on portfolio return or Sharpe ratio
        """

        def __init__(
            self,
            features: np.ndarray,
            prices: np.ndarray,
            reward_type: str = "returns",
            initial_balance: float = 10000.0,
        ):
            super().__init__()
            
            self.features = features
            self.prices = prices
            self.reward_type = reward_type
            self.initial_balance = initial_balance
            
            self.n_steps = len(features)
            self.current_step = 0
            
            # Define action and observation space
            self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(features.shape[1] + 2,),  # features + position + balance ratio
                dtype=np.float32
            )
            
            # Trading state
            self.balance = initial_balance
            self.position = 0.0  # 0 = no position, 1 = long
            self.entry_price = 0.0
            self.returns = []

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.current_step = 0
            self.balance = self.initial_balance
            self.position = 0.0
            self.entry_price = 0.0
            self.returns = []
            return self._get_observation(), {}

        def _get_observation(self):
            """Get current observation."""
            features = self.features[self.current_step]
            # Add position and balance ratio to observation
            balance_ratio = self.balance / self.initial_balance
            obs = np.concatenate([features, [self.position, balance_ratio]])
            return obs.astype(np.float32)

        def step(self, action):
            """Execute trading action."""
            current_price = self.prices[self.current_step]
            reward = 0.0
            
            # Execute action
            if action == 1:  # BUY
                if self.position == 0:
                    self.position = 1.0
                    self.entry_price = current_price
            elif action == 2:  # SELL
                if self.position > 0:
                    # Calculate return
                    trade_return = (current_price - self.entry_price) / self.entry_price
                    self.balance *= (1 + trade_return)
                    self.returns.append(trade_return)
                    self.position = 0.0
                    
                    if self.reward_type == "returns":
                        reward = trade_return * 100  # Scale for learning
                    elif self.reward_type == "sharpe":
                        if len(self.returns) > 1:
                            reward = np.mean(self.returns) / (np.std(self.returns) + 1e-8)
                        else:
                            reward = trade_return
            
            # Move to next step
            self.current_step += 1
            terminated = self.current_step >= self.n_steps - 1
            truncated = False
            
            # Update unrealized P&L in observation
            if self.position > 0:
                unrealized = (current_price - self.entry_price) / self.entry_price
                # Small reward for holding profitable position
                if self.reward_type == "returns":
                    reward += unrealized * 0.1
            
            obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
            
            return obs, reward, terminated, truncated, {}
    
    return TradingEnv


# Create the class if dependencies are available
TradingEnv = _create_trading_env_class()


@strategy_registry.register(
    "rl_dqn",
    description="Reinforcement Learning DQN Strategy",
)
class RLDQNStrategy(BaseStrategy):
    """
    Reinforcement Learning DQN Strategy.
    
    Uses Deep Q-Network to learn trading policy from price data.
    Trains on historical data and generates signals based on learned policy.
    
    Config params:
        algorithm: RL algorithm (DQN, PPO, A2C)
        learning_rate: Learning rate for the optimizer
        gamma: Discount factor for future rewards
        buffer_size: Replay buffer size (for DQN)
        train_episodes: Number of training episodes
        features: List of feature names to use as state
        reward_type: "returns", "sharpe", or "sortino"
    
    Note: Requires stable-baselines3 and gymnasium packages.
    """

    name = "rl_dqn"

    def _setup(
        self,
        algorithm: str = "DQN",
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        train_episodes: int = 100,
        features: list[str] | None = None,
        reward_type: str = "returns",
        **kwargs,
    ) -> None:
        if not RL_AVAILABLE:
            logger.error("RL dependencies not available. Strategy will return HOLD signals.")
            self._model = None
            self.features = []
            self.algorithm_name = algorithm
            return
            
        self.algorithm_name = algorithm
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.train_episodes = train_episodes
        self.features = features or ["sma_20", "rsi_14", "macd", "atr"]
        self.reward_type = reward_type
        
        self._model = None
        self._is_trained = False
        self._features_mean = None
        self._features_std = None

    def _compute_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute feature matrix from candles."""
        features_df = pd.DataFrame(index=candles.index)

        for feature in self.features:
            try:
                if feature.startswith("sma_"):
                    period = int(feature.split("_")[1])
                    features_df[feature] = indicator_registry.compute("sma", candles, period=period)
                elif feature.startswith("rsi_"):
                    period = int(feature.split("_")[1])
                    features_df[feature] = indicator_registry.compute("rsi", candles, period=period)
                elif feature == "macd":
                    macd_df = indicator_registry.compute("macd", candles)
                    features_df["macd"] = macd_df["macd"]
                    features_df["macd_signal"] = macd_df["signal"]
                elif feature == "atr":
                    features_df[feature] = indicator_registry.compute("atr", candles)
                elif feature == "adx":
                    features_df[feature] = indicator_registry.compute("adx", candles)
                else:
                    features_df[feature] = indicator_registry.compute(feature, candles)
            except Exception as e:
                logger.warning(f"Could not compute feature {feature}: {e}")

        return features_df

    def _train(self, candles: pd.DataFrame) -> None:
        """Train the RL model."""
        if not RL_AVAILABLE or TradingEnv is None:
            return
            
        features_df = self._compute_features(candles)
        valid_idx = features_df.dropna().index
        
        if len(valid_idx) < 100:
            logger.warning("Insufficient data for RL training")
            return

        # Prepare data
        features_array = features_df.loc[valid_idx].values
        prices_array = candles.loc[valid_idx, "close"].values

        # Normalize features
        features_mean = features_array.mean(axis=0)
        features_std = features_array.std(axis=0) + 1e-8
        features_normalized = (features_array - features_mean) / features_std
        
        self._features_mean = features_mean
        self._features_std = features_std

        # Create environment
        env = TradingEnv(
            features=features_normalized,
            prices=prices_array,
            reward_type=self.reward_type,
        )
        env = DummyVecEnv([lambda: env])

        # Create model
        algorithms = {"DQN": DQN, "PPO": PPO, "A2C": A2C}
        algorithm_class = algorithms.get(self.algorithm_name)
        if algorithm_class is None:
            logger.error(f"Unknown algorithm: {self.algorithm_name}")
            return

        if self.algorithm_name == "DQN":
            self._model = algorithm_class(
                "MlpPolicy",
                env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                buffer_size=self.buffer_size,
                verbose=0,
            )
        else:
            self._model = algorithm_class(
                "MlpPolicy",
                env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                verbose=0,
            )

        # Train
        total_timesteps = len(valid_idx) * self.train_episodes
        logger.info(f"Training RL model for {total_timesteps} timesteps...")
        self._model.learn(total_timesteps=total_timesteps)
        self._is_trained = True
        logger.info("RL model training complete")

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        if not RL_AVAILABLE:
            logger.warning("RL not available, returning HOLD signals")
            return signals

        # Train if not already trained
        if not self._is_trained:
            self._train(candles)
            
        if self._model is None:
            return signals

        # Compute features
        features_df = self._compute_features(candles)
        valid_idx = features_df.dropna().index

        if len(valid_idx) == 0:
            return signals

        # Normalize features
        features_array = features_df.loc[valid_idx].values
        features_normalized = (features_array - self._features_mean) / self._features_std

        # Get actions from model
        position = 0.0
        balance_ratio = 1.0
        
        for i, idx in enumerate(valid_idx):
            obs = np.concatenate([
                features_normalized[i],
                [position, balance_ratio]
            ]).astype(np.float32)
            
            action, _ = self._model.predict(obs, deterministic=True)
            action = int(action)
            
            if action == 1 and position == 0:  # BUY
                signals.loc[idx] = Signal.BUY
                position = 1.0
            elif action == 2 and position > 0:  # SELL
                signals.loc[idx] = Signal.SELL
                position = 0.0

        return self.apply_filters(signals, candles)

    def get_parameters(self) -> dict[str, Any]:
        """Get parameters including RL config."""
        params = super().get_parameters()
        params["algorithm"] = self.algorithm_name
        params["rl_available"] = RL_AVAILABLE
        return params
