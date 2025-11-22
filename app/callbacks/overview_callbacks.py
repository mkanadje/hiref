"""
Callbacks for Overview page (Page 1)
"""

from dash import Input, Output
import plotly.graph_objects as go
import numpy as np
from app.utils.plot_helpers import (
    plot_actual_vs_predicted,
    plot_error_distribution_over_time,
)


def register_callbacks(app, df):
    """Register all callbacks for the overview page."""

    @app.callback(
        [
            Output("total-records-card", "children"),
            Output("test-mae-card", "children"),
            Output("test-mape-card", "children"),
            Output("avg-driver-pct-card", "children"),
        ],
        Input("url", "pathname"),  # Dummy input to trigger on page load
    )
    def update_summary_cards(pathname):
        """Update the summary metric cards."""
        total_records = f"{len(df):,}"

        # Test set metrics
        test_df = df[df["dataset_split"] == "test"]
        test_mae = test_df["abs_error"].mean()

        # MAPE (excluding low sales)
        mape_mask = test_df["actual_sales"] >= 50
        if mape_mask.sum() > 0:
            test_mape = test_df[mape_mask]["abs_pct_error"].mean()
            test_mape_str = f"{test_mape:.1f}%"
        else:
            test_mape_str = "N/A"

        # Average driver percentage
        total_magnitude = (
            df["baseline_original"].abs()
            + df["total_driver_contribution_original"].abs()
        )
        driver_pct = (
            df["total_driver_contribution_original"].abs() / total_magnitude * 100
        ).mean()

        return (
            total_records,
            f"{test_mae:.1f}",
            test_mape_str,
            f"{driver_pct:.1f}%",
        )

    @app.callback(
        Output("metrics-comparison-chart", "figure"),
        Input("url", "pathname"),
    )
    def update_metrics_comparison(pathname):
        """Create bar chart comparing metrics across dataset splits."""
        metrics_data = []

        for split in ["train", "val", "test"]:
            split_df = df[df["dataset_split"] == split]
            if len(split_df) == 0:
                continue

            mae = split_df["abs_error"].mean()
            rmse = np.sqrt((split_df["error"] ** 2).mean())

            metrics_data.append({"split": split, "MAE": mae, "RMSE": rmse})

        import pandas as pd

        metrics_df = pd.DataFrame(metrics_data)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name="MAE",
                x=metrics_df["split"],
                y=metrics_df["MAE"],
                marker_color="#1f77b4",
            )
        )

        fig.add_trace(
            go.Bar(
                name="RMSE",
                x=metrics_df["split"],
                y=metrics_df["RMSE"],
                marker_color="#ff7f0e",
            )
        )

        fig.update_layout(
            barmode="group",
            xaxis_title="Dataset Split",
            yaxis_title="Error",
            height=350,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        return fig

    @app.callback(
        Output("error-over-time-chart", "figure"),
        Input("url", "pathname"),
    )
    def update_error_over_time(pathname):
        """Create line chart of error over time."""
        return plot_error_distribution_over_time(df, metric="abs_error")

    # Populate region options (top-level, always shows all)
    @app.callback(
        Output("filter-region", "options"),
        Input("url", "pathname"),
    )
    def populate_region_options(pathname):
        """Populate region dropdown with all regions."""
        return [{"label": r, "value": r} for r in sorted(df["region"].unique())]

    # Cascading filter: State options based on selected regions
    @app.callback(
        Output("filter-state", "options"),
        Input("filter-region", "value"),
    )
    def populate_state_options(region_filter):
        """Populate state dropdown based on selected regions."""
        if region_filter:
            # Filter states to only those in selected regions
            filtered_df = df[df["region"].isin(region_filter)]
            states = sorted(filtered_df["state"].unique())
        else:
            # No region selected, show all states
            states = sorted(df["state"].unique())
        return [{"label": s, "value": s} for s in states]

    # Cascading filter: Segment options based on selected regions/states
    @app.callback(
        Output("filter-segment", "options"),
        [Input("filter-region", "value"), Input("filter-state", "value")],
    )
    def populate_segment_options(region_filter, state_filter):
        """Populate segment dropdown based on selected regions/states."""
        filtered_df = df.copy()
        if region_filter:
            filtered_df = filtered_df[filtered_df["region"].isin(region_filter)]
        if state_filter:
            filtered_df = filtered_df[filtered_df["state"].isin(state_filter)]
        segments = sorted(filtered_df["segment"].unique())
        return [{"label": s, "value": s} for s in segments]

    # Cascading filter: Brand options based on upstream filters
    @app.callback(
        Output("filter-brand", "options"),
        [
            Input("filter-region", "value"),
            Input("filter-state", "value"),
            Input("filter-segment", "value"),
        ],
    )
    def populate_brand_options(region_filter, state_filter, segment_filter):
        """Populate brand dropdown based on selected regions/states/segments."""
        filtered_df = df.copy()
        if region_filter:
            filtered_df = filtered_df[filtered_df["region"].isin(region_filter)]
        if state_filter:
            filtered_df = filtered_df[filtered_df["state"].isin(state_filter)]
        if segment_filter:
            filtered_df = filtered_df[filtered_df["segment"].isin(segment_filter)]
        brands = sorted(filtered_df["brand"].unique())
        return [{"label": b, "value": b} for b in brands]

    # Cascading filter: Pack options based on upstream filters
    @app.callback(
        Output("filter-pack", "options"),
        [
            Input("filter-region", "value"),
            Input("filter-state", "value"),
            Input("filter-segment", "value"),
            Input("filter-brand", "value"),
        ],
    )
    def populate_pack_options(
        region_filter, state_filter, segment_filter, brand_filter
    ):
        """Populate pack dropdown based on all upstream filters."""
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

    @app.callback(
        Output("actual-vs-predicted-chart", "figure"),
        [
            Input("filter-region", "value"),
            Input("filter-state", "value"),
            Input("filter-segment", "value"),
            Input("filter-brand", "value"),
            Input("filter-pack", "value"),
            Input("filter-dataset", "value"),
        ],
    )
    def update_actual_vs_predicted(
        region_filter,
        state_filter,
        segment_filter,
        brand_filter,
        pack_filter,
        dataset_filter,
    ):
        """Create scatter plot of actual vs predicted with filters applied."""
        # Start with full dataset
        filtered_df = df.copy()

        # Apply filters if selected
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
        if dataset_filter:
            filtered_df = filtered_df[filtered_df["dataset_split"].isin(dataset_filter)]

        # If no data after filtering, return empty figure
        if len(filtered_df) == 0:
            return go.Figure().update_layout(
                title="No data matching filters",
                xaxis={"visible": False},
                yaxis={"visible": False},
            )

        return plot_actual_vs_predicted(filtered_df, sample_size=10000)

    # =========================================================================
    # TIMELINE FILTERS (Separate cascading filters for timeline chart)
    # =========================================================================

    # Timeline: Populate region options
    @app.callback(
        Output("timeline-filter-region", "options"),
        Input("url", "pathname"),
    )
    def populate_timeline_region_options(pathname):
        """Populate timeline region dropdown with all regions."""
        return [{"label": r, "value": r} for r in sorted(df["region"].unique())]

    # Timeline: Cascading filter for state
    @app.callback(
        Output("timeline-filter-state", "options"),
        Input("timeline-filter-region", "value"),
    )
    def populate_timeline_state_options(region_filter):
        """Populate timeline state dropdown based on selected regions."""
        if region_filter:
            filtered_df = df[df["region"].isin(region_filter)]
            states = sorted(filtered_df["state"].unique())
        else:
            states = sorted(df["state"].unique())
        return [{"label": s, "value": s} for s in states]

    # Timeline: Cascading filter for segment
    @app.callback(
        Output("timeline-filter-segment", "options"),
        [
            Input("timeline-filter-region", "value"),
            Input("timeline-filter-state", "value"),
        ],
    )
    def populate_timeline_segment_options(region_filter, state_filter):
        """Populate timeline segment dropdown based on selected regions/states."""
        filtered_df = df.copy()
        if region_filter:
            filtered_df = filtered_df[filtered_df["region"].isin(region_filter)]
        if state_filter:
            filtered_df = filtered_df[filtered_df["state"].isin(state_filter)]
        segments = sorted(filtered_df["segment"].unique())
        return [{"label": s, "value": s} for s in segments]

    # Timeline: Cascading filter for brand
    @app.callback(
        Output("timeline-filter-brand", "options"),
        [
            Input("timeline-filter-region", "value"),
            Input("timeline-filter-state", "value"),
            Input("timeline-filter-segment", "value"),
        ],
    )
    def populate_timeline_brand_options(region_filter, state_filter, segment_filter):
        """Populate timeline brand dropdown based on upstream filters."""
        filtered_df = df.copy()
        if region_filter:
            filtered_df = filtered_df[filtered_df["region"].isin(region_filter)]
        if state_filter:
            filtered_df = filtered_df[filtered_df["state"].isin(state_filter)]
        if segment_filter:
            filtered_df = filtered_df[filtered_df["segment"].isin(segment_filter)]
        brands = sorted(filtered_df["brand"].unique())
        return [{"label": b, "value": b} for b in brands]

    # Timeline: Cascading filter for pack
    @app.callback(
        Output("timeline-filter-pack", "options"),
        [
            Input("timeline-filter-region", "value"),
            Input("timeline-filter-state", "value"),
            Input("timeline-filter-segment", "value"),
            Input("timeline-filter-brand", "value"),
        ],
    )
    def populate_timeline_pack_options(
        region_filter, state_filter, segment_filter, brand_filter
    ):
        """Populate timeline pack dropdown based on all upstream filters."""
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

    # Timeline chart with separate filters
    @app.callback(
        Output("timeline-chart", "figure"),
        [
            Input("timeline-filter-region", "value"),
            Input("timeline-filter-state", "value"),
            Input("timeline-filter-segment", "value"),
            Input("timeline-filter-brand", "value"),
            Input("timeline-filter-pack", "value"),
            Input("timeline-filter-dataset", "value"),
        ],
    )
    def update_timeline_chart(
        region_filter,
        state_filter,
        segment_filter,
        brand_filter,
        pack_filter,
        dataset_filter,
    ):
        """Create timeline chart showing actual vs predicted over time with same filters."""
        # Start with full dataset
        filtered_df = df.copy()

        # Apply filters if selected (same logic as scatter plot)
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
        if dataset_filter:
            filtered_df = filtered_df[filtered_df["dataset_split"].isin(dataset_filter)]

        # If no data after filtering, return empty figure
        if len(filtered_df) == 0:
            return go.Figure().update_layout(
                title="No data matching filters",
                xaxis={"visible": False},
                yaxis={"visible": False},
            )

        # Aggregate by date and dataset_split
        import pandas as pd

        timeline_data = (
            filtered_df.groupby(["date", "dataset_split"])
            .agg({"actual_sales": "sum", "prediction": "sum"})
            .reset_index()
        )

        # Create figure
        fig = go.Figure()

        # Color mapping for dataset splits
        color_map = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}

        # Add actual sales lines for each dataset split
        for split in ["train", "val", "test"]:
            split_data = timeline_data[timeline_data["dataset_split"] == split]
            if len(split_data) > 0:
                # Actual sales (solid line)
                fig.add_trace(
                    go.Scatter(
                        x=split_data["date"],
                        y=split_data["actual_sales"],
                        mode="lines",
                        name=f"Actual ({split})",
                        line=dict(color=color_map[split], width=2),
                        legendgroup=split,
                    )
                )
                # Predicted sales (dashed line)
                fig.add_trace(
                    go.Scatter(
                        x=split_data["date"],
                        y=split_data["prediction"],
                        mode="lines",
                        name=f"Predicted ({split})",
                        line=dict(color=color_map[split], width=2, dash="dash"),
                        legendgroup=split,
                    )
                )

        fig.update_layout(
            title="Actual vs Predicted Sales Over Time",
            xaxis_title="Date",
            yaxis_title="Total Sales",
            height=500,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        return fig

    # =========================================================================
    # DYNAMIC METRICS TABLE CALLBACKS
    # =========================================================================

    # Metrics: Populate region options
    @app.callback(
        Output("metrics-filter-region", "options"),
        Input("url", "pathname"),
    )
    def populate_metrics_region_options(pathname):
        return [{"label": r, "value": r} for r in sorted(df["region"].unique())]

    # Metrics: Cascading filter for state
    @app.callback(
        Output("metrics-filter-state", "options"),
        Input("metrics-filter-region", "value"),
    )
    def populate_metrics_state_options(region_filter):
        if region_filter:
            filtered_df = df[df["region"].isin(region_filter)]
            states = sorted(filtered_df["state"].unique())
        else:
            states = sorted(df["state"].unique())
        return [{"label": s, "value": s} for s in states]

    # Metrics: Cascading filter for segment
    @app.callback(
        Output("metrics-filter-segment", "options"),
        [
            Input("metrics-filter-region", "value"),
            Input("metrics-filter-state", "value"),
        ],
    )
    def populate_metrics_segment_options(region_filter, state_filter):
        filtered_df = df.copy()
        if region_filter:
            filtered_df = filtered_df[filtered_df["region"].isin(region_filter)]
        if state_filter:
            filtered_df = filtered_df[filtered_df["state"].isin(state_filter)]
        segments = sorted(filtered_df["segment"].unique())
        return [{"label": s, "value": s} for s in segments]

    # Metrics: Cascading filter for brand
    @app.callback(
        Output("metrics-filter-brand", "options"),
        [
            Input("metrics-filter-region", "value"),
            Input("metrics-filter-state", "value"),
            Input("metrics-filter-segment", "value"),
        ],
    )
    def populate_metrics_brand_options(region_filter, state_filter, segment_filter):
        filtered_df = df.copy()
        if region_filter:
            filtered_df = filtered_df[filtered_df["region"].isin(region_filter)]
        if state_filter:
            filtered_df = filtered_df[filtered_df["state"].isin(state_filter)]
        if segment_filter:
            filtered_df = filtered_df[filtered_df["segment"].isin(segment_filter)]
        brands = sorted(filtered_df["brand"].unique())
        return [{"label": b, "value": b} for b in brands]

    # Metrics: Cascading filter for pack
    @app.callback(
        Output("metrics-filter-pack", "options"),
        [
            Input("metrics-filter-region", "value"),
            Input("metrics-filter-state", "value"),
            Input("metrics-filter-segment", "value"),
            Input("metrics-filter-brand", "value"),
        ],
    )
    def populate_metrics_pack_options(
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

    # Update the dynamic metrics table
    @app.callback(
        Output("dynamic-metrics-table", "children"),
        [
            Input("metrics-filter-region", "value"),
            Input("metrics-filter-state", "value"),
            Input("metrics-filter-segment", "value"),
            Input("metrics-filter-brand", "value"),
            Input("metrics-filter-pack", "value"),
        ],
    )
    def update_dynamic_metrics_table(
        region_filter, state_filter, segment_filter, brand_filter, pack_filter
    ):
        """Update metrics table based on filters."""
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
            return html.Div(
                "No data available for selected filters",
                className="text-center text-muted my-4",
            )

        # Determine aggregation level
        # We aggregate by date and dataset_split to get total sales for the selected scope per time period
        # This effectively "rolls up" the data to the filtered level
        agg_df = (
            filtered_df.groupby(["date", "dataset_split"])
            .agg({"actual_sales": "sum", "prediction": "sum"})
            .reset_index()
        )

        # Calculate metrics for each split
        metrics_data = []
        from sklearn.metrics import r2_score

        for split in ["train", "val", "test"]:
            split_data = agg_df[agg_df["dataset_split"] == split]

            if len(split_data) == 0:
                continue

            actuals = split_data["actual_sales"]
            preds = split_data["prediction"]

            # Calculate metrics
            mae = (actuals - preds).abs().mean()

            # MAPE (handle zeros)
            mask = actuals >= 1  # Avoid division by zero or very small numbers
            if mask.sum() > 0:
                mape = (
                    (actuals[mask] - preds[mask]).abs() / actuals[mask]
                ).mean() * 100
            else:
                mape = np.nan

            bias = (preds - actuals).mean()

            if len(actuals) > 1:
                r2 = r2_score(actuals, preds)
            else:
                r2 = np.nan

            metrics_data.append(
                {
                    "Dataset": split.upper(),
                    "Records (Aggregated)": len(split_data),
                    "Total Actual Sales": actuals.sum(),
                    "MAE": mae,
                    "MAPE (%)": mape,
                    "Bias": bias,
                    "R²": r2,
                }
            )

        if not metrics_data:
            return html.Div(
                "Insufficient data to calculate metrics",
                className="text-center text-muted my-4",
            )

        import pandas as pd
        import dash_bootstrap_components as dbc

        metrics_df = pd.DataFrame(metrics_data)

        # Format the table
        table = dbc.Table.from_dataframe(
            metrics_df.round(2),
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            className="mb-0",
        )

        return table
