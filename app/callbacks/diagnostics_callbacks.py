"""
Callbacks for Diagnostics page (Page 4)
"""

from dash import Input, Output, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from app.utils.plot_helpers import plot_contribution_distributions, plot_feature_correlation_heatmap
from app.utils.data_loader import get_driver_columns


def register_callbacks(app, df):
    """Register all callbacks for the diagnostics page."""

    @app.callback(
        Output("contribution-distributions-chart", "figure"),
        Input("url", "pathname"),
    )
    def update_contribution_distributions(pathname):
        """Update contribution distribution histograms."""
        return plot_contribution_distributions(df)

    @app.callback(
        Output("negative-contrib-summary", "children"),
        Input("url", "pathname"),
    )
    def update_negative_contrib_summary(pathname):
        """Summarize negative driver contributions."""
        driver_cols = get_driver_columns(df)["original"]

        # Count records with negative total driver contribution
        neg_total_count = (df["total_driver_contribution_original"] < 0).sum()
        neg_total_pct = neg_total_count / len(df) * 100

        # Count which individual drivers are most often negative
        neg_driver_counts = {}
        for col in driver_cols:
            feature_name = col.replace("driver_", "").replace("_original", "")
            neg_count = (df[col] < 0).sum()
            neg_pct = neg_count / len(df) * 100
            neg_driver_counts[feature_name] = {"count": neg_count, "pct": neg_pct}

        # Sort by percentage
        sorted_drivers = sorted(neg_driver_counts.items(), key=lambda x: x[1]["pct"], reverse=True)

        # Create summary
        summary = [
            html.H6(f"Records with negative total driver contribution: {neg_total_count:,} ({neg_total_pct:.1f}%)"),
            html.Hr(),
            html.H6("Individual Drivers with Negative Contributions:", className="mt-3"),
            html.Ul(
                [
                    html.Li(f"{driver}: {info['count']:,} records ({info['pct']:.1f}%)")
                    for driver, info in sorted_drivers[:10]
                ]
            ),
        ]

        if neg_total_pct > 50:
            summary.insert(
                0,
                dbc.Alert(
                    "Warning: More than 50% of records have negative total driver contribution. This suggests the model may be over-relying on baseline (embeddings) and learning incorrect driver relationships.",
                    color="danger",
                    className="mb-3",
                ),
            )

        return summary

    @app.callback(
        Output("driver-sign-chart", "figure"),
        Input("url", "pathname"),
    )
    def update_driver_sign_chart(pathname):
        """Create stacked bar chart showing positive vs negative contributions."""
        driver_cols = get_driver_columns(df)["original"]

        sign_data = []
        for col in driver_cols:
            feature_name = col.replace("driver_", "").replace("_original", "")
            pos_pct = (df[col] > 0).sum() / len(df) * 100
            neg_pct = (df[col] < 0).sum() / len(df) * 100
            zero_pct = (df[col] == 0).sum() / len(df) * 100

            sign_data.append(
                {
                    "driver": feature_name,
                    "positive": pos_pct,
                    "negative": neg_pct,
                    "zero": zero_pct,
                }
            )

        sign_df = pd.DataFrame(sign_data).sort_values("negative", ascending=False)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name="Positive",
                y=sign_df["driver"],
                x=sign_df["positive"],
                orientation="h",
                marker_color="#2ca02c",
            )
        )

        fig.add_trace(
            go.Bar(
                name="Negative",
                y=sign_df["driver"],
                x=sign_df["negative"],
                orientation="h",
                marker_color="#d62728",
            )
        )

        fig.add_trace(
            go.Bar(
                name="Zero",
                y=sign_df["driver"],
                x=sign_df["zero"],
                orientation="h",
                marker_color="#7f7f7f",
            )
        )

        fig.update_layout(
            barmode="stack",
            xaxis_title="Percentage of Records",
            yaxis_title="Driver",
            height=400,
            xaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return fig

    @app.callback(
        Output("feature-correlation-heatmap", "figure"),
        Input("url", "pathname"),
    )
    def update_feature_correlation_heatmap(pathname):
        """Update feature correlation heatmap."""
        return plot_feature_correlation_heatmap(df)

    @app.callback(
        Output("sample-problematic-records", "children"),
        Input("url", "pathname"),
    )
    def update_sample_problematic_records(pathname):
        """Show sample records with high baseline and negative drivers."""
        # Filter records where baseline is high and total driver contrib is negative
        problematic = df[
            (df["baseline_original"] > df["baseline_original"].quantile(0.75))
            & (df["total_driver_contribution_original"] < 0)
        ].copy()

        if len(problematic) == 0:
            return html.P("No problematic records found.", className="text-muted")

        # Sample 10 records
        sample = problematic.sample(min(10, len(problematic)), random_state=42)

        # Select columns to display
        display_cols = [
            "date",
            "sku_key",
            "actual_sales",
            "prediction",
            "baseline_original",
            "total_driver_contribution_original",
        ]

        sample_display = sample[display_cols].round(2)

        table = dbc.Table.from_dataframe(
            sample_display,
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            size="sm",
        )

        return [
            html.P(f"Found {len(problematic):,} problematic records. Showing 10 samples:", className="mb-2"),
            table,
        ]
