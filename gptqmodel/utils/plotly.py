import plotly.graph_objects as go


def create_plotly(
        x,
        y,
        xaxis_title,
        yaxis_title,
):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers'))
    fig.update_layout(
        title='',
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title='Legend'
    )
    return fig
