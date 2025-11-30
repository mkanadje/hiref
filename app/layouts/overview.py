"""
Page 1: Overview & Model Performance
"""

import dash_bootstrap_components as dbc
from dash import html, dcc


def create_layout():
    """Create the overview page layout."""

    return dbc.Container(
        [
            dbc.Row([html.H2("Model Performance Overview", className="mt-4 mb-4")]),
            # Summary cards row
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H6(
                                            "Total Records",
                                            className="card-subtitle mb-2 text-muted",
                                        ),
                                        html.H3(
                                            id="total-records-card",
                                            className="card-title",
                                        ),
                                    ]
                                )
                            ],
                            className="mb-4",
                            color="primary",
                            outline=True,
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H6(
                                            "Test MAE",
                                            className="card-subtitle mb-2 text-muted",
                                        ),
                                        html.H3(
                                            id="test-mae-card", className="card-title"
                                        ),
                                    ]
                                )
                            ],
                            className="mb-4",
                            color="info",
                            outline=True,
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H6(
                                            "Test MAPE",
                                            className="card-subtitle mb-2 text-muted",
                                        ),
                                        html.H3(
                                            id="test-mape-card", className="card-title"
                                        ),
                                    ]
                                )
                            ],
                            className="mb-4",
                            color="warning",
                            outline=True,
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H6(
                                            "Avg Driver %",
                                            className="card-subtitle mb-2 text-muted",
                                        ),
                                        html.H3(
                                            id="avg-driver-pct-card",
                                            className="card-title",
                                        ),
                                    ]
                                )
                            ],
                            className="mb-4",
                            color="success",
                            outline=True,
                        ),
                        width=3,
                    ),
                ],
                className="mb-4",
            ),
            # Performance metrics comparison
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Performance Metrics by Dataset Split",
                                                className="card-title",
                                            ),
                                            dcc.Graph(id="metrics-comparison-chart"),
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
                                            html.H5(
                                                "Error Over Time",
                                                className="card-title",
                                            ),
                                            dcc.Graph(id="error-over-time-chart"),
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
            # Actual vs Predicted scatter with filters
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Actual vs Predicted Sales",
                                                className="card-title",
                                            ),
                                            # Filters row
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Region:"),
                                                            dcc.Dropdown(
                                                                id="filter-region",
                                                                placeholder="All Regions",
                                                                multi=True,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        width=2,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("State:"),
                                                            dcc.Dropdown(
                                                                id="filter-state",
                                                                placeholder="All States",
                                                                multi=True,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        width=2,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Segment:"),
                                                            dcc.Dropdown(
                                                                id="filter-segment",
                                                                placeholder="All Segments",
                                                                multi=True,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        width=2,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Brand:"),
                                                            dcc.Dropdown(
                                                                id="filter-brand",
                                                                placeholder="All Brands",
                                                                multi=True,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        width=2,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Pack Size:"),
                                                            dcc.Dropdown(
                                                                id="filter-pack",
                                                                placeholder="All Packs",
                                                                multi=True,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        width=2,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Dataset:"),
                                                            dcc.Dropdown(
                                                                id="filter-dataset",
                                                                options=[
                                                                    {
                                                                        "label": "Train",
                                                                        "value": "train",
                                                                    },
                                                                    {
                                                                        "label": "Val",
                                                                        "value": "val",
                                                                    },
                                                                    {
                                                                        "label": "Test",
                                                                        "value": "test",
                                                                    },
                                                                ],
                                                                placeholder="All Datasets",
                                                                multi=True,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        width=2,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            dcc.Graph(id="actual-vs-predicted-chart"),
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
            # Error Analysis Table with filters
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Error Analysis by Dataset Split",
                                                className="card-title",
                                            ),
                                            # Hierarchical filters
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Region:"),
                                                            dcc.Dropdown(
                                                                id="error-table-filter-region",
                                                                placeholder="All Regions",
                                                                multi=False,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        width=2,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("State:"),
                                                            dcc.Dropdown(
                                                                id="error-table-filter-state",
                                                                placeholder="All States",
                                                                multi=False,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        width=2,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Segment:"),
                                                            dcc.Dropdown(
                                                                id="error-table-filter-segment",
                                                                placeholder="All Segments",
                                                                multi=False,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        width=2,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Brand:"),
                                                            dcc.Dropdown(
                                                                id="error-table-filter-brand",
                                                                placeholder="All Brands",
                                                                multi=False,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        width=2,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Pack Size:"),
                                                            dcc.Dropdown(
                                                                id="error-table-filter-pack",
                                                                placeholder="All Packs",
                                                                multi=False,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        width=2,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Br(),
                                                            dbc.Button(
                                                                "Reset Filters",
                                                                id="error-table-reset-button",
                                                                color="secondary",
                                                                size="sm",
                                                                className="mt-2",
                                                            ),
                                                        ],
                                                        width=2,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Error metrics table
                                            html.Div(id="error-analysis-table"),
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
