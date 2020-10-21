import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np
import math
np.random.seed(1)

def stream_plot():
    x = np.linspace(-3,3,200)
    y = np.linspace(-3,3,200)
    X,Y = np.meshgrid(x, y)
    dy = X**2 - Y**2
    dx = np.log((Y**2 - Y + 1)/3)



    figure = ff.create_streamline(x, y, dx, dy, arrow_scale = .2,
    density = 1.5, name = "Лінії потоку")

    figure.add_trace(go.Scatter(x=[2, -0.2], y=[2, 1],
                            mode='lines+markers',
                            line=dict(color='green',width=3)))

    figure.add_trace(go.Scatter(x=[2, 1.4], y=[2, 1],
                            mode='lines+markers',
                            line=dict(color='green',width=3)))



    figure.update_layout(
        title=r"$\text{Лінії потоку системи диференційних рівнянь}$",
        font=dict(
        family="Arial",
        size=20,
        color="black"
        )
    )
    figure['layout'].update(width=500, height=500, autosize=False)
    figure.show()
stream_plot()
