"""Core Kronos model implementation for time series prediction.

This module provides the main Kronos model class used for stock price
prediction based on historical OHLCV data.
"""

import numpy as np
from typing import Optional, Tuple, Union


class KronosConfig:
    """Configuration class for the Kronos model."""

    def __init__(
        self,
        context_length: int = 64,
        prediction_length: int = 10,
        num_samples: int = 100,
        device: str = "cpu",
        model_name: str = "kronos-bolt-small",
    ):
        """
        Args:
            context_length: Number of historical time steps to use as input.
            prediction_length: Number of future time steps to predict.
            num_samples: Number of sample paths for probabilistic forecasting.
            device: Computation device ('cpu' or 'cuda').
            model_name: Pretrained model variant to load.
        """
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.device = device
        self.model_name = model_name


class KronosPredictor:
    """Wrapper around the Kronos forecasting pipeline.

    Provides a simplified interface for loading a pretrained Kronos model
    and generating price predictions from historical stock data.
    """

    def __init__(self, config: Optional[KronosConfig] = None):
        """
        Args:
            config: Model configuration. Defaults to KronosConfig() if None.
        """
        self.config = config or KronosConfig()
        self._pipeline = None

    def load(self) -> "KronosPredictor":
        """Load the pretrained Kronos pipeline.

        Returns:
            self, to allow method chaining.

        Raises:
            ImportError: If the `kronos-forecasting` package is not installed.
        """
        try:
            from kronos_forecasting import KronosPipeline  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "kronos-forecasting is required. "
                "Install it with: pip install kronos-forecasting"
            ) from exc

        self._pipeline = KronosPipeline.from_pretrained(
            self.config.model_name,
            device_map=self.config.device,
        )
        return self

    @property
    def is_loaded(self) -> bool:
        """Return True if the underlying pipeline has been loaded."""
        return self._pipeline is not None

    def predict(
        self,
        context: np.ndarray,
        prediction_length: Optional[int] = None,
        num_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate probabilistic forecasts from a 1-D context array.

        Args:
            context: 1-D numpy array of historical close prices (float32/64).
            prediction_length: Steps to predict. Falls back to config value.
            num_samples: Sample paths to draw. Falls back to config value.

        Returns:
            Tuple of (low, median, high) arrays, each of shape
            (prediction_length,), representing the 10th, 50th, and 90th
            percentiles of the forecast distribution.

        Raises:
            RuntimeError: If the model has not been loaded via `.load()`.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model is not loaded. Call .load() before .predict()."
            )

        pred_len = prediction_length or self.config.prediction_length
        n_samples = num_samples or self.config.num_samples

        import torch  # type: ignore

        ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        forecast = self._pipeline.predict(
            ctx_tensor,
            prediction_length=pred_len,
            num_samples=n_samples,
        )  # shape: (1, num_samples, pred_len)

        samples: np.ndarray = forecast[0].numpy()  # (num_samples, pred_len)

        low = np.percentile(samples, 10, axis=0)
        median = np.percentile(samples, 50, axis=0)
        high = np.percentile(samples, 90, axis=0)

        return low, median, high

    def predict_from_dataframe(
        self,
        df,
        close_col: str = "close",
        prediction_length: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convenience wrapper that accepts a pandas DataFrame.

        Args:
            df: DataFrame containing historical OHLCV data.
            close_col: Column name for closing prices.
            prediction_length: Steps to predict. Falls back to config value.

        Returns:
            Tuple of (low, median, high) forecast arrays.
        """
        context = df[close_col].values.astype(np.float32)
        # Use only the last `context_length` bars
        context = context[-self.config.context_length :]
        return self.predict(context, prediction_length=prediction_length)
