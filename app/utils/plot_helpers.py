"""
Reusable plotting functions for the dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def create_metric_card(title, value, subtitle="", color="primary"):
    """
    Create a metric card for display.

    Args:
        title: Card title
        value: Main value to display
        subtitle: Optional subtitle
        color: Bootstrap color class

    Returns:
        dict: Card configuration
    """
    return {"title": title, "value": value, "subtitle": subtitle, "color": color}


def plot_actual_vs_predicted(df, sample_size=10000):
    """
    Create scatter plot of actual vs predicted sales.

    Args:
        df: Results dataframe
        sample_size: Max number of points to plot (for performance)

    Returns:
        plotly.graph_objects.Figure
    """
    # Sample for performance if dataset is large
    if len(df) > sample_size:
        plot_df = df.sample(sample_size, random_state=42)
    else:
        plot_df = df

    fig = px.scatter(
        plot_df,
        x="actual_sales",
        y="prediction",
        color="dataset_split",
        color_discrete_map={"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"},
        opacity=0.5,
        labels={"actual_sales": "Actual Sales", "prediction": "Predicted Sales"},
        title="Actual vs Predicted Sales",
        hover_data=["sku_key", "date", "error"],
    )

    # Add 45-degree reference line
    max_val = max(plot_df["actual_sales"].max(), plot_df["prediction"].max())
    min_val = min(plot_df["actual_sales"].min(), plot_df["prediction"].min())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color="red", dash="dash", width=2),
        )
    )

    fig.update_layout(height=500, showlegend=True)

    return fig


def plot_driver_importance(df, top_n=12):
    """
    Create horizontal bar chart of driver importance.

    Args:
        df: Results dataframe
        top_n: Number of top drivers to show

    Returns:
        plotly.graph_objects.Figure
    """
    driver_cols = [
        col
        for col in df.columns
        if col.startswith("driver_") and col.endswith("_original")
    ]

    # Calculate average absolute contribution
    avg_abs_contrib = df[driver_cols].abs().mean().sort_values(ascending=True)

    # Take top N
    avg_abs_contrib = avg_abs_contrib.tail(top_n)

    # Clean feature names
    feature_names = [
        col.replace("driver_", "").replace("_original", "")
        for col in avg_abs_contrib.index
    ]

    fig = go.Figure(
        go.Bar(
            x=avg_abs_contrib.values,
            y=feature_names,
            orientation="h",
            marker=dict(color=avg_abs_contrib.values, colorscale="Viridis"),
        )
    )

    fig.update_layout(
        title="Average Absolute Driver Contribution (Original Scale)",
        xaxis_title="Average Absolute Contribution",
        yaxis_title="Driver",
        height=450,
        showlegend=False,
    )

    return fig


def plot_baseline_vs_drivers_breakdown(df, groupby_col="segment"):
    """
    Create stacked bar chart showing baseline vs driver contribution %.

    Args:
        df: Results dataframe
        groupby_col: Column to group by (segment, region, etc.)

    Returns:
        plotly.graph_objects.Figure
    """
    # Calculate percentages for each group
    grouped = (
        df.groupby(groupby_col)
        .apply(
            lambda x: pd.Series(
                {
                    "baseline_pct": (
                        x["baseline_original"].abs().sum()
                        / (
                            x["baseline_original"].abs().sum()
                            + x["total_driver_contribution_original"].abs().sum()
                        )
                    )
                    * 100,
                    "driver_pct": (
                        x["total_driver_contribution_original"].abs().sum()
                        / (
                            x["baseline_original"].abs().sum()
                            + x["total_driver_contribution_original"].abs().sum()
                        )
                    )
                    * 100,
                }
            )
        )
        .reset_index()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Baseline %",
            x=grouped[groupby_col],
            y=grouped["baseline_pct"],
            marker_color="#1f77b4",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Drivers %",
            x=grouped[groupby_col],
            y=grouped["driver_pct"],
            marker_color="#ff7f0e",
        )
    )

    fig.update_layout(
        barmode="stack",
        title=f"Baseline vs Driver Contribution % by {groupby_col.title()}",
        xaxis_title=groupby_col.title(),
        yaxis_title="Contribution %",
        height=450,
        yaxis=dict(range=[0, 100]),
    )

    return fig


def plot_error_distribution_over_time(df, metric="abs_error"):
    """
    Create line chart of error metrics over time.

    Args:
        df: Results dataframe
        metric: Which error metric to plot

    Returns:
        plotly.graph_objects.Figure
    """
    # Aggregate by month and dataset split
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    monthly_error = df.groupby(["month", "dataset_split"])[metric].mean().reset_index()

    fig = px.line(
        monthly_error,
        x="month",
        y=metric,
        color="dataset_split",
        color_discrete_map={"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"},
        labels={"month": "Month", metric: metric.replace("_", " ").title()},
        title=f"{metric.replace('_', ' ').title()} Over Time",
    )

    fig.update_layout(height=400, showlegend=True)

    return fig


def plot_contribution_distributions(df):
    """
    Create histogram of baseline and driver contribution distributions.

    Args:
        df: Results dataframe

    Returns:
        plotly.graph_objects.Figure (with subplots)
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Baseline Distribution",
            "Total Driver Contribution Distribution",
        ),
    )

    # Baseline histogram
    fig.add_trace(
        go.Histogram(
            x=df["baseline_original"],
            name="Baseline",
            marker_color="#1f77b4",
            nbinsx=50,
        ),
        row=1,
        col=1,
    )

    # Driver contribution histogram
    fig.add_trace(
        go.Histogram(
            x=df["total_driver_contribution_original"],
            name="Driver Contrib",
            marker_color="#ff7f0e",
            nbinsx=50,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Baseline (Original)", row=1, col=1)
    fig.update_xaxes(title_text="Total Driver Contribution (Original)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    fig.update_layout(
        height=450, showlegend=False, title_text="Contribution Distributions"
    )

    return fig


def plot_feature_correlation_heatmap(df):
    """
    Create correlation heatmap between features and actual sales.

    Args:
        df: Results dataframe

    Returns:
        plotly.graph_objects.Figure
    """
    # Get feature columns (not driver_ columns, just the original features)
    feature_cols = [
        "temperature",
        "precipitation",
        "price",
        "tv_spend",
        "digital_spend",
        "trade_spend",
        "discount_pct",
        "distribution",
        "gdp_index",
        "unemployment_rate",
        "consumer_confidence",
    ]

    # Filter to only columns that exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    feature_cols_with_sales = feature_cols + ["actual_sales"]

    # Calculate correlation matrix
    corr_matrix = df[feature_cols_with_sales].corr()

    # Extract correlations with actual_sales
    sales_corr = corr_matrix["actual_sales"].drop("actual_sales")

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )

    fig.update_layout(
        title="Feature Correlation Matrix",
        height=650,
        xaxis=dict(tickangle=-45),
    )

    return fig
