#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Tests for SuperchargedMMM class."""

import numpy as np
import pandas as pd
import pytest

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation, SuperchargedMMM

seed: int = sum(map(ord, "supercharged_mmm"))
rng: np.random.Generator = np.random.default_rng(seed=seed)


@pytest.fixture(scope="module")
def toy_X() -> pd.DataFrame:
    """Generate toy data for testing."""
    date_data: pd.DatetimeIndex = pd.date_range(
        start="2019-06-01", end="2021-12-31", freq="W-MON"
    )
    n: int = date_data.size

    return pd.DataFrame(
        data={
            "date": date_data,
            "channel_1": rng.integers(low=0, high=400, size=n),
            "channel_2": rng.integers(low=0, high=50, size=n),
        }
    )


@pytest.fixture(scope="module")
def toy_y(toy_X: pd.DataFrame) -> pd.Series:
    """Generate toy target data for testing."""
    return pd.Series(data=rng.integers(low=0, high=100, size=toy_X.shape[0]))


class TestSuperchargedMMM:
    """Test class for SuperchargedMMM."""

    def test_init_with_monthly_seasonality(self):
        """Test initialization with monthly seasonality."""
        model = SuperchargedMMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            monthly_seasonality=1,
        )
        
        assert model.monthly_seasonality == 1
        assert hasattr(model, "monthly_fourier")
        assert model.weekly_seasonality is None

    def test_init_with_weekly_seasonality(self):
        """Test initialization with weekly seasonality."""
        model = SuperchargedMMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            weekly_seasonality=1,
        )
        
        assert model.weekly_seasonality == 1
        assert hasattr(model, "weekly_fourier")
        assert model.monthly_seasonality is None

    def test_init_with_all_seasonalities(self):
        """Test initialization with yearly, monthly, and weekly seasonality."""
        model = SuperchargedMMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            yearly_seasonality=2,
            monthly_seasonality=1,
            weekly_seasonality=1,
        )
        
        assert model.yearly_seasonality == 2
        assert model.monthly_seasonality == 1
        assert model.weekly_seasonality == 1
        assert hasattr(model, "yearly_fourier")
        assert hasattr(model, "monthly_fourier")
        assert hasattr(model, "weekly_fourier")

    def test_model_building_with_monthly_seasonality(self, toy_X, toy_y):
        """Test model building with monthly seasonality."""
        model = SuperchargedMMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            monthly_seasonality=1,
        )
        
        model.build_model(toy_X, toy_y)
        
        assert model.model is not None
        assert "monthly_seasonality_contribution" in model.model.named_vars

    def test_model_building_with_weekly_seasonality(self, toy_X, toy_y):
        """Test model building with weekly seasonality."""
        model = SuperchargedMMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            weekly_seasonality=1,
        )
        
        model.build_model(toy_X, toy_y)
        
        assert model.model is not None
        assert "weekly_seasonality_contribution" in model.model.named_vars

    def test_model_building_with_all_seasonalities(self, toy_X, toy_y):
        """Test model building with all seasonality types."""
        model = SuperchargedMMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            yearly_seasonality=2,
            monthly_seasonality=1,
            weekly_seasonality=1,
        )
        
        model.build_model(toy_X, toy_y)
        
        assert model.model is not None
        assert "yearly_seasonality_contribution" in model.model.named_vars
        assert "monthly_seasonality_contribution" in model.model.named_vars
        assert "weekly_seasonality_contribution" in model.model.named_vars
        assert "yearly_seasonality_contribution_original_scale" in model.model.named_vars
        assert "monthly_seasonality_contribution_original_scale" in model.model.named_vars
        assert "weekly_seasonality_contribution_original_scale" in model.model.named_vars

    def test_default_model_config(self):
        """Test that default model config includes monthly and weekly priors."""
        model = SuperchargedMMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            monthly_seasonality=1,
            weekly_seasonality=1,
        )
        
        config = model.default_model_config
        assert "gamma_monthly_fourier" in config
        assert "gamma_weekly_fourier" in config

    def test_plot_components_contributions_method_exists(self):
        """Test that plot_components_contributions method exists and is callable."""
        model = SuperchargedMMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            monthly_seasonality=1,
            weekly_seasonality=1,
        )
        
        assert hasattr(model, "plot_components_contributions")
        assert callable(model.plot_components_contributions)

    def test_inherits_mmm_methods(self):
        """Test that SuperchargedMMM inherits key MMM methods."""
        model = SuperchargedMMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            monthly_seasonality=1,
        )
        
        # Check that key MMM methods are accessible
        mmm_methods = [
            "plot_direct_contribution_curves",
            "sample_posterior_predictive",
            "optimize_budget",
            "get_channel_contributions_forward_pass",
            "compute_channel_contribution_forward_pass",
        ]
        
        for method_name in mmm_methods:
            assert hasattr(model, method_name), f"Missing method: {method_name}"
            assert callable(getattr(model, method_name)), f"Method not callable: {method_name}"
