"""
Callbacks for Driver Analysis page (Page 2)
"""

from dash import Input, Output, html
import dash_bootstrap_components as dbc
import pandas as pd
from app.utils.plot_helpers import plot_driver_importance, plot_baseline_vs_drivers_breakdown
from app.utils.data_loader import get_driver_columns


def register_callbacks(app, df):
    """Register all callbacks for the driver analysis page."""

    @app.callback(
        Output("driver-importance-chart", "figure"),
        Input("url", "pathname"),
    )
    def update_driver_importance(pathname):
        """Update driver importance chart."""
        return plot_driver_importance(df, top_n=12)

    @app.callback(
        Output("baseline-vs-drivers-chart", "figure"),
        Input("breakdown-groupby-dropdown", "value"),
    )
    def update_baseline_vs_drivers(groupby_col):
        """Update baseline vs drivers breakdown based on groupby selection."""
        return plot_baseline_vs_drivers_breakdown(df, groupby_col=groupby_col)

    @app.callback(
        Output("driver-stats-table", "children"),
        Input("url", "pathname"),
    )
    def update_driver_stats_table(pathname):
        """Create table showing statistics for each driver."""
        driver_cols = get_driver_columns(df)["original"]

        stats_data = []
        for col in driver_cols:
            feature_name = col.replace("driver_", "").replace("_original", "")
            stats_data.append(
                {
                    "Driver": feature_name,
                    "Mean": df[col].mean(),
                    "Std": df[col].std(),
                    "Min": df[col].min(),
                    "Max": df[col].max(),
                    "% Positive": (df[col] > 0).sum() / len(df) * 100,
                    "% Negative": (df[col] < 0).sum() / len(df) * 100,
                    "Avg Absolute": df[col].abs().mean(),
                }
            )

        stats_df = pd.DataFrame(stats_data).sort_values("Avg Absolute", ascending=False)

        # Create a nice table
        table = dbc.Table.from_dataframe(
            stats_df.round(2),
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            size="sm",
        )

        return table
