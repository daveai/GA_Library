def plot_performance_chart(df):
    import plotly.graph_objects as go
    import pandas as pd
    import plotly.express as px
    from plotly.subplots import make_subplots

    x = df["Generation"]
    x_rev = x[::-1]
    y1 = df["Fitness_Mean"]
    y1_upper = df["Fitness_Lower"]
    y1_lower = df["Fitness_Upper"]

    # line
    trace1 = go.Scatter(
        x=x, y=y1, line=dict(color="rgb(0,100,80)"), mode="lines", name="Fair",
    )

    trace2 = go.Scatter(
        x=x,
        y=y1_upper,
        fill="tozerox",
        fillcolor="rgba(0,100,80,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
        name="Fair",
    )

    trace3 = go.Scatter(
        x=x,
        y=y1_lower,
        fill="tozerox",
        fillcolor="rgba(0,100,80,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
        name="Fair",
    )

    data = [trace1]

    layout = go.Layout(
        paper_bgcolor="rgb(255,255,255)",
        plot_bgcolor="rgb(229,229,229)",
        xaxis=dict(
            gridcolor="rgb(255,255,255)",
            range=[1, 10],
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor="rgb(127,127,127)",
            ticks="outside",
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgb(255,255,255)",
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor="rgb(127,127,127)",
            ticks="outside",
            zeroline=False,
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    fig.show()
