# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import plotly.graph_objects as go


def create_plotly(
        x: object,
        y: object,
        xaxis_title: object,
        yaxis_title: object,
) -> object:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers'))
    fig.update_layout(
        title='',
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title='Legend'
    )
    return fig
