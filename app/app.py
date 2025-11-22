"""
Main Dash application for Hierarchical Model Results Analysis.

Run this file to launch the dashboard:
    python app/app.py

Then navigate to http://localhost:8050 in your browser.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Import layouts
from app.layouts import overview, drivers, diagnostics

# Import callbacks
from app.callbacks import overview_callbacks, driver_callbacks, diagnostics_callbacks

# Import data loader
from app.utils.data_loader import load_results_data


# =============================================================================
# Initialize App
# =============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)

app.title = "Hierarchical Model Dashboard"

# =============================================================================
# Load Data
# =============================================================================

print("Loading results data...")
df = load_results_data()
print(f"Loaded {len(df):,} records")

# =============================================================================
# App Layout
# =============================================================================

app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        # Header
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Overview", href="/", active="exact", className="nav-link")),
                dbc.NavItem(dbc.NavLink("Drivers", href="/drivers", active="exact", className="nav-link")),
                dbc.NavItem(dbc.NavLink("Diagnostics", href="/diagnostics", active="exact", className="nav-link")),
            ],
            brand="📊 Hierarchical Model Dashboard",
            brand_href="/",
            color="primary",
            dark=True,
            className="mb-4",
            sticky="top",
        ),
        # Page content
        html.Div(id="page-content"),
    ],
    fluid=True,
    className="px-0",
)


# =============================================================================
# Page Navigation Callback
# =============================================================================


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    """Route to the appropriate page based on URL."""
    if pathname == "/drivers":
        return drivers.create_layout()
    elif pathname == "/diagnostics":
        return diagnostics.create_layout()
    else:  # Default to overview
        return overview.create_layout()


# =============================================================================
# Register All Callbacks
# =============================================================================

overview_callbacks.register_callbacks(app, df)
driver_callbacks.register_callbacks(app, df)
diagnostics_callbacks.register_callbacks(app, df)


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Starting Hierarchical Model Dashboard...")
    print("Navigate to: http://localhost:8050")
    print("=" * 80 + "\n")

    app.run(debug=True, host="0.0.0.0", port=8050)
