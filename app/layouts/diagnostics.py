"""
Page 4: Diagnostics (Debug Issues)
"""

import dash_bootstrap_components as dbc
from dash import html, dcc


def create_layout():
    """Create the diagnostics page layout."""

    return dbc.Container(
        [
            dbc.Row([html.H2("Model Diagnostics", className="mt-4 mb-4")]),
            # Contribution distribution analysis
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Contribution Distributions", className="card-title"),
                                            html.P(
                                                "Analyze the distribution of baseline and driver contributions to identify issues.",
                                                className="text-muted",
                                            ),
                                            dcc.Graph(id="contribution-distributions-chart"),
                                        ]
                                    )
                                ],
                                className="mb-4",
                            )
                        ],
                        width=12,
                    )
                ],
                className="mb-4",
            ),
            # Negative contribution investigation
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Negative Driver Contributions Investigation", className="card-title"),
                                            html.Div(id="negative-contrib-summary"),
                                        ]
                                    )
                                ],
                                className="mb-4",
                            )
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Driver Sign Analysis", className="card-title"),
                                            html.P(
                                                "Check which drivers are predominantly positive vs negative.",
                                                className="text-muted",
                                            ),
                                            dcc.Graph(id="driver-sign-chart"),
                                        ]
                                    )
                                ],
                                className="mb-4",
                            )
                        ],
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            # Feature correlation heatmap
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Feature-Target Correlation Matrix", className="card-title"),
                                            html.P(
                                                "Validate that features have expected relationships with actual sales.",
                                                className="text-muted",
                                            ),
                                            dcc.Graph(id="feature-correlation-heatmap"),
                                        ]
                                    )
                                ],
                                className="mb-4",
                            )
                        ],
                        width=12,
                    )
                ],
                className="mb-4",
            ),
            # Sample records with issues
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Sample Records with High Baseline / Negative Drivers", className="card-title"),
                                            html.Div(id="sample-problematic-records"),
                                        ]
                                    )
                                ],
                                className="mb-4",
                            )
                        ],
                        width=12,
                    )
                ],
                className="mb-4",
            ),
        ],
        fluid=True,
    )
