"""
Callbacks for Driver Analysis page (Page 2)
"""

from dash import Input, Output, html
import dash_bootstrap_components as dbc
import pandas as pd
from app.utils.plot_helpers import (
    plot_driver_importance,
    plot_baseline_vs_drivers_breakdown,
)
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

    # =========================================================================
    # STACKED DECOMPOSITION CALLBACKS
    # =========================================================================

    # Decomp: Populate region options
    @app.callback(
        Output("decomp-filter-region", "options"),
        Input("url", "pathname"),
    )
    def populate_decomp_region_options(pathname):
        return [{"label": r, "value": r} for r in sorted(df["region"].unique())]

    # Decomp: Cascading filter for state
    @app.callback(
        Output("decomp-filter-state", "options"),
        Input("decomp-filter-region", "value"),
    )
    def populate_decomp_state_options(region_filter):
        if region_filter:
            filtered_df = df[df["region"].isin(region_filter)]
            states = sorted(filtered_df["state"].unique())
        else:
            states = sorted(df["state"].unique())
        return [{"label": s, "value": s} for s in states]

    # Decomp: Cascading filter for segment
    @app.callback(
        Output("decomp-filter-segment", "options"),
        [Input("decomp-filter-region", "value"), Input("decomp-filter-state", "value")],
    )
    def populate_decomp_segment_options(region_filter, state_filter):
        filtered_df = df.copy()
        if region_filter:
            filtered_df = filtered_df[filtered_df["region"].isin(region_filter)]
        if state_filter:
            filtered_df = filtered_df[filtered_df["state"].isin(state_filter)]
        segments = sorted(filtered_df["segment"].unique())
        return [{"label": s, "value": s} for s in segments]

    # Decomp: Cascading filter for brand
    @app.callback(
        Output("decomp-filter-brand", "options"),
        [
            Input("decomp-filter-region", "value"),
            Input("decomp-filter-state", "value"),
            Input("decomp-filter-segment", "value"),
        ],
    )
    def populate_decomp_brand_options(region_filter, state_filter, segment_filter):
        filtered_df = df.copy()
        if region_filter:
            filtered_df = filtered_df[filtered_df["region"].isin(region_filter)]
        if state_filter:
            filtered_df = filtered_df[filtered_df["state"].isin(state_filter)]
        if segment_filter:
            filtered_df = filtered_df[filtered_df["segment"].isin(segment_filter)]
        brands = sorted(filtered_df["brand"].unique())
        return [{"label": b, "value": b} for b in brands]

    # Decomp: Cascading filter for pack
    @app.callback(
        Output("decomp-filter-pack", "options"),
        [
            Input("decomp-filter-region", "value"),
            Input("decomp-filter-state", "value"),
            Input("decomp-filter-segment", "value"),
            Input("decomp-filter-brand", "value"),
        ],
    )
    def populate_decomp_pack_options(
        region_filter, state_filter, segment_filter, brand_filter
    ):
        filtered_df = df.copy()
        if region_filter:
            filtered_df = filtered_df[filtered_df["region"].isin(region_filter)]
        if state_filter:
            filtered_df = filtered_df[filtered_df["state"].isin(state_filter)]
        if segment_filter:
            filtered_df = filtered_df[filtered_df["segment"].isin(segment_filter)]
        if brand_filter:
            filtered_df = filtered_df[filtered_df["brand"].isin(brand_filter)]
        packs = sorted(filtered_df["pack"].unique())
        return [{"label": p, "value": p} for p in packs]

    # Update the stacked decomposition chart
    @app.callback(
        Output("stacked-decomposition-chart", "figure"),
        [
            Input("decomp-filter-region", "value"),
            Input("decomp-filter-state", "value"),
            Input("decomp-filter-segment", "value"),
            Input("decomp-filter-brand", "value"),
            Input("decomp-filter-pack", "value"),
        ],
    )
    def update_stacked_decomposition(
        region_filter, state_filter, segment_filter, brand_filter, pack_filter
    ):
        """Update stacked decomposition chart based on filters."""
        from app.utils.plot_helpers import plot_stacked_decomposition

        # Start with full dataset
        filtered_df = df.copy()

        # Apply filters
        if region_filter:
            filtered_df = filtered_df[filtered_df["region"].isin(region_filter)]
        if state_filter:
            filtered_df = filtered_df[filtered_df["state"].isin(state_filter)]
        if segment_filter:
            filtered_df = filtered_df[filtered_df["segment"].isin(segment_filter)]
        if brand_filter:
            filtered_df = filtered_df[filtered_df["brand"].isin(brand_filter)]
        if pack_filter:
            filtered_df = filtered_df[filtered_df["pack"].isin(pack_filter)]

        if len(filtered_df) == 0:
            return go.Figure().update_layout(
                title="No data matching filters",
                xaxis={"visible": False},
                yaxis={"visible": False},
            )

        return plot_stacked_decomposition(filtered_df)
