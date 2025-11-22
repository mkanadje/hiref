"""
Page 2: Driver Contribution Analysis
"""

import dash_bootstrap_components as dbc
from dash import html, dcc


def create_layout():
    """Create the driver analysis page layout."""

    return dbc.Container(
        [
            dbc.Row([html.H2("Driver Contribution Analysis", className="mt-4 mb-4")]),
            # Driver importance
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Driver Importance (Average Absolute Contribution)",
                                                className="card-title",
                                            ),
                                            dcc.Graph(id="driver-importance-chart"),
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
            # Stacked Decomposition Analysis
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Stacked Decomposition Analysis",
                                                className="card-title",
                                            ),
                                            # Filters row
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Region:"),
                                                            dcc.Dropdown(
                                                                id="decomp-filter-region",
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
                                                                id="decomp-filter-state",
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
                                                                id="decomp-filter-segment",
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
                                                                id="decomp-filter-brand",
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
                                                                id="decomp-filter-pack",
                                                                placeholder="All Packs",
                                                                multi=True,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        width=2,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            dcc.Graph(id="stacked-decomposition-chart"),
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
            # Baseline vs Drivers breakdown
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Baseline vs Driver Contribution %",
                                                className="card-title",
                                            ),
                                            dbc.Label("Group By:"),
                                            dcc.Dropdown(
                                                id="breakdown-groupby-dropdown",
                                                options=[
                                                    {
                                                        "label": "Segment",
                                                        "value": "segment",
                                                    },
                                                    {
                                                        "label": "Region",
                                                        "value": "region",
                                                    },
                                                    {
                                                        "label": "Pack Size",
                                                        "value": "pack",
                                                    },
                                                    {
                                                        "label": "Dataset Split",
                                                        "value": "dataset_split",
                                                    },
                                                ],
                                                value="segment",
                                                clearable=False,
                                                className="mb-3",
                                            ),
                                            dcc.Graph(id="baseline-vs-drivers-chart"),
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
            # Individual driver statistics
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "Individual Driver Statistics",
                                                className="card-title",
                                            ),
                                            html.Div(id="driver-stats-table"),
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
