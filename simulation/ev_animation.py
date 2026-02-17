import plotly.graph_objects as go

def animate_ev(fig, route, positions, placeholder,
               clearance_km, chosen_signal, scale_factor):

    clearance_dist_m = clearance_km * 1000
    traveled_m = 0
    prev_x, prev_y = positions[route[0]]

    SIGNAL_NODES = ["B", "E"]

    # ---------------------------------------
    # Base traces (roads, labels, incidents)
    # ---------------------------------------
    base_traces = list(fig.data)

    frames = []  # all animation frames

    # ---------------------------------------
    # Build animation frames
    # ---------------------------------------
    for i in range(1, len(route)):
        start = route[i - 1]
        end = route[i]

        x0, y0 = positions[start]
        x1, y1 = positions[end]

        steps = 30

        for t in range(steps):

            xt = x0 + (x1 - x0) * (t / steps)
            yt = y0 + (y1 - y0) * (t / steps)

            dx = xt - prev_x
            dy = yt - prev_y
            traveled_m += ((dx**2 + dy**2)**0.5) * scale_factor
            prev_x, prev_y = xt, yt

            # -------- SIGNAL LOGIC --------
            frame_traces = list(base_traces)

            if traveled_m >= clearance_dist_m:
                # chosen signal → green
                for sig in SIGNAL_NODES:
                    sx, sy = positions[sig]
                    color = "green" if sig == chosen_signal else "red"
                    frame_traces.append(
                        go.Scatter(x=[sx], y=[sy],
                                   mode="markers",
                                   marker=dict(size=20, color=color),
                                   name=f"Signal_{sig}")
                    )
            else:
                # all red
                for sig in SIGNAL_NODES:
                    sx, sy = positions[sig]
                    frame_traces.append(
                        go.Scatter(x=[sx], y=[sy],
                                   mode="markers",
                                   marker=dict(size=20, color="red"),
                                   name=f"Signal_{sig}")
                    )

            # -------- EV marker --------
            frame_traces.append(
                go.Scatter(x=[xt], y=[yt],
                           mode="markers",
                           marker=dict(size=16, color="cyan"),
                           name="EV")
            )

            # Add frame
            frames.append(go.Frame(data=frame_traces))

    # ---------------------------------------
    # Build final animation figure
    # ---------------------------------------
    anim_fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )

    anim_fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=600,
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {"label": "▶ Play", "method": "animate", "args": [None]},
                    {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 10000}, "mode": "immediate"}]}
                ],
            }
        ]
    )

    placeholder.plotly_chart(anim_fig, use_container_width=True)

