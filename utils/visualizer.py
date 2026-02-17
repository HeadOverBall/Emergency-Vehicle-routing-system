import plotly.graph_objs as go

def create_base_figure(G, positions, route, incident_edge, chosen_signal=None):
    """
    Builds the static base figure:
    - Road network
    - Node labels
    - Incident edge
    - Traffic signals (B and E)
    """

    SIGNAL_NODES = ["B", "E"]

    # -------------------------------------------
    # Build road edges
    # -------------------------------------------
    edge_x = []
    edge_y = []

    for u, v in G.edges():
        x0, y0 = positions[u]
        x1, y1 = positions[v]

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    # -------------------------------------------
    # Create base figure
    # -------------------------------------------
    fig = go.Figure()

    # Road lines
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=2, color="#6EC1FF"),
        name="Road"
    ))

    # -------------------------------------------
    # Node labels
    # -------------------------------------------
    for node, (x, y) in positions.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            text=[node],
            mode="text",
            textposition="top center",
            showlegend=False
        ))

    # -------------------------------------------
    # Incident edge highlight
    # -------------------------------------------
    if incident_edge:
        u, v = incident_edge
        x0, y0 = positions[u]
        x1, y1 = positions[v]

        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(color="red", width=10),
            name="Incident Edge"
        ))

    # -------------------------------------------
    # INITIAL SIGNAL STATES (ALL RED)
    # These will be overridden by animation
    # -------------------------------------------
    for sig in SIGNAL_NODES:
        sx, sy = positions[sig]
        fig.add_trace(go.Scatter(
            x=[sx], y=[sy],
            mode="markers",
            marker=dict(size=22, color="red"),
            name=f"Signal_{sig}_Red"
        ))

    # -------------------------------------------
    # Final appearance settings
    # -------------------------------------------
    fig.update_layout(
        title="EV Routing Simulation",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=600,
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="white")
    )

    return fig