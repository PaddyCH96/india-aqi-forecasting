"""Basic smoke tests for lib/charts.py chart functions."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TestChartFunctions:
    @staticmethod
    def _make_df():
        return pd.DataFrame({
            "ds": pd.date_range("2023-01-01", periods=100, freq="D"),
            "y": np.random.default_rng(42).uniform(50, 200, 100),
        })

    @staticmethod
    def _make_forecast():
        return pd.DataFrame({
            "ds": pd.date_range("2023-01-01", periods=200, freq="D"),
            "yhat": np.random.default_rng(42).uniform(50, 200, 200),
            "yhat_lower": np.random.default_rng(42).uniform(30, 180, 200),
            "yhat_upper": np.random.default_rng(42).uniform(70, 220, 200),
        })

    def test_plot_history_runs(self):
        from lib.charts import plot_history
        fig, ax = plt.subplots()
        plot_history(ax, self._make_df(), "TestCity")
        plt.close(fig)

    def test_plot_history_with_covid(self):
        from lib.charts import plot_history
        fig, ax = plt.subplots()
        plot_history(ax, self._make_df(), "TestCity", show_covid=True)
        plt.close(fig)

    def test_plot_forecast_runs(self):
        from lib.charts import plot_forecast
        fig, ax = plt.subplots()
        plot_forecast(ax, self._make_df(), self._make_forecast(), "TestCity")
        plt.close(fig)

    def test_plot_monthly_breakdown_runs(self):
        from lib.charts import plot_monthly_breakdown
        fig, ax = plt.subplots()
        forecast = self._make_forecast()
        forecast["ds"] = pd.date_range("2030-01-01", periods=200, freq="D")
        plot_monthly_breakdown(ax, forecast, 2030)
        plt.close(fig)

    def test_plot_monthly_breakdown_empty_year(self):
        from lib.charts import plot_monthly_breakdown
        fig, ax = plt.subplots()
        plot_monthly_breakdown(ax, self._make_forecast(), 2099)
        plt.close(fig)

    def test_plot_validation_runs(self):
        from lib.charts import plot_validation
        fig, ax = plt.subplots()
        df = self._make_df().rename(columns={"y": "yhat"})
        df["y"] = df["yhat"] * np.random.default_rng(42).uniform(0.8, 1.2, 100)
        plot_validation(ax, df, 15.6, "TestCity")
        plt.close(fig)

    def test_plot_scatter_accuracy_runs(self):
        from lib.charts import plot_scatter_accuracy
        fig, ax = plt.subplots()
        df = self._make_df().rename(columns={"y": "yhat"})
        df["y"] = df["yhat"] * np.random.default_rng(42).uniform(0.8, 1.2, 100)
        plot_scatter_accuracy(ax, df, 15.6)
        plt.close(fig)

    def test_plot_comparison_barh_runs(self):
        from lib.charts import plot_comparison_barh
        fig, ax = plt.subplots()
        df = pd.DataFrame({
            "City": ["Delhi", "Mumbai", "Hyderabad"],
            "MAPE (%)": [12.3, 15.6, 18.9],
        })
        plot_comparison_barh(ax, df)
        plt.close(fig)
