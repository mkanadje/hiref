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

    # =========================================================================
    # ERROR ANALYSIS TABLE CALLBACKS
    # =========================================================================

    # Error Table: Update all filter options based on current selections
    @app.callback(
        [
            Output("error-table-filter-region", "options"),
            Output("error-table-filter-state", "options"),
            Output("error-table-filter-segment", "options"),
            Output("error-table-filter-brand", "options"),
            Output("error-table-filter-pack", "options"),
        ],
        [
            Input("error-table-filter-region", "value"),
            Input("error-table-filter-state", "value"),
            Input("error-table-filter-segment", "value"),
            Input("error-table-filter-brand", "value"),
            Input("error-table-filter-pack", "value"),
        ],
    )
    def update_error_filter_options(region_val, state_val, segment_val, brand_val, pack_val):
        """Update all filter options based on current selections."""
        # Start with full dataset
        filtered_df = df.copy()

        # Apply filters to get valid combinations
        if region_val:
            filtered_df = filtered_df[filtered_df["region"] == region_val]
        if state_val:
            filtered_df = filtered_df[filtered_df["state"] == state_val]
        if segment_val:
            filtered_df = filtered_df[filtered_df["segment"] == segment_val]
        if brand_val:
            filtered_df = filtered_df[filtered_df["brand"] == brand_val]
        if pack_val:
            filtered_df = filtered_df[filtered_df["pack"] == pack_val]

        # Get available options for each filter based on filtered data
        region_options = [{"label": r, "value": r} for r in sorted(filtered_df["region"].unique())]
        state_options = [{"label": s, "value": s} for s in sorted(filtered_df["state"].unique())]
        segment_options = [{"label": s, "value": s} for s in sorted(filtered_df["segment"].unique())]
        brand_options = [{"label": b, "value": b} for b in sorted(filtered_df["brand"].unique())]
        pack_options = [{"label": p, "value": p} for p in sorted(filtered_df["pack"].unique())]

        return region_options, state_options, segment_options, brand_options, pack_options

    # Reset button callback
    @app.callback(
        [
            Output("error-table-filter-region", "value"),
            Output("error-table-filter-state", "value"),
            Output("error-table-filter-segment", "value"),
            Output("error-table-filter-brand", "value"),
            Output("error-table-filter-pack", "value"),
        ],
        Input("error-table-reset-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_error_filters(n_clicks):
        """Reset all error table filters."""
        return None, None, None, None, None

    # Update the error analysis table
    @app.callback(
        Output("error-analysis-table", "children"),
        [
            Input("error-table-filter-region", "value"),
            Input("error-table-filter-state", "value"),
            Input("error-table-filter-segment", "value"),
            Input("error-table-filter-brand", "value"),
            Input("error-table-filter-pack", "value"),
        ],
    )
    def update_error_analysis_table(region_filter, state_filter, segment_filter, brand_filter, pack_filter):
        """Update error analysis table based on filters."""
        import pandas as pd
        import dash_bootstrap_components as dbc
        from sklearn.metrics import r2_score

        # Start with full dataset
        filtered_df = df.copy()

        # Apply filters (single-select, so direct comparison)
        if region_filter:
            filtered_df = filtered_df[filtered_df["region"] == region_filter]
        if state_filter:
            filtered_df = filtered_df[filtered_df["state"] == state_filter]
        if segment_filter:
            filtered_df = filtered_df[filtered_df["segment"] == segment_filter]
        if brand_filter:
            filtered_df = filtered_df[filtered_df["brand"] == brand_filter]
        if pack_filter:
            filtered_df = filtered_df[filtered_df["pack"] == pack_filter]

        if len(filtered_df) == 0:
            return html.Div(
                "No data available for selected filters",
                className="text-center text-muted my-4",
            )

        # Calculate metrics for each split
        metrics_data = []

        for split in ["train", "val", "test"]:
            split_df = filtered_df[filtered_df["dataset_split"] == split]

            if len(split_df) == 0:
                continue

            actuals = split_df["actual_sales"].values
            preds = split_df["prediction"].values
            errors = split_df["error"].values

            # Calculate comprehensive error metrics
            mae = np.abs(errors).mean()
            rmse = np.sqrt((errors ** 2).mean())

            # MAPE (excluding very small values to avoid division issues)
            mask = actuals >= 50
            if mask.sum() > 0:
                mape = (np.abs(errors[mask]) / actuals[mask]).mean() * 100
                mape_coverage = mask.sum() / len(actuals) * 100
            else:
                mape = np.nan
                mape_coverage = 0

            # Bias (mean error)
            bias = errors.mean()
            bias_pct = (bias / actuals.mean()) * 100 if actuals.mean() > 0 else np.nan

            # R² Score
            if len(actuals) > 1 and actuals.std() > 0:
                r2 = r2_score(actuals, preds)
            else:
                r2 = np.nan

            # Mean Absolute Error as % of mean actual
            mae_pct = (mae / actuals.mean()) * 100 if actuals.mean() > 0 else np.nan

            metrics_data.append({
                "Dataset": split.upper(),
                "Records": f"{len(split_df):,}",
                "MAE": f"{mae:.2f}",
                "MAE (%)": f"{mae_pct:.2f}%" if not np.isnan(mae_pct) else "N/A",
                "RMSE": f"{rmse:.2f}",
                "MAPE": f"{mape:.2f}%" if not np.isnan(mape) else "N/A",
                "MAPE Coverage": f"{mape_coverage:.1f}%" if not np.isnan(mape) else "N/A",
                "Bias": f"{bias:+.2f}",
                "Bias (%)": f"{bias_pct:+.2f}%" if not np.isnan(bias_pct) else "N/A",
                "R²": f"{r2:.4f}" if not np.isnan(r2) else "N/A",
            })

        if not metrics_data:
            return html.Div(
                "Insufficient data to calculate metrics",
                className="text-center text-muted my-4",
            )

        metrics_df = pd.DataFrame(metrics_data)

        # Create formatted table
        table = dbc.Table.from_dataframe(
            metrics_df,
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            className="mb-0 text-center",
            style={"fontSize": "0.9rem"},
        )

        return table
