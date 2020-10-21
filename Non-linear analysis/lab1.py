import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np
np.random.seed(1)



def eigs():
    vals, vecs = np.linalg.eig([[4,1],[2,5]])
    print('Власні значення: ', vals)
    print('Власні вектори: ', vecs)

def to_plot():
    x_source, y_source = 0.0, 0.0
    x = np.linspace(-5, 5, 500)
    y = np.linspace(-5, 5, 500)
    X, Y = np.meshgrid(x, y)
    dx = 1*X - 1*Y - 5
    dy = 2*X + 1*Y
    figure = ff.create_streamline(x, y, dx, dy, arrow_scale = .2,
    density = 1.5, name = "Лінії потоку")

    figure.add_trace(go.Scatter(x=[0, -0.7], y=[0,-0.44],
                            mode='lines+markers',
                            name = "Перший власний вектор",
                            line=dict(color='green',width=3)))

    figure.add_trace(go.Scatter(x=[0, 0.7], y=[0, -0.9],
                            mode='lines+markers',
                            name = "Другий власний вектор",
                            line=dict(color='green',width=3)))

    figure.add_trace(go.Scatter(x=[-1, 2], y=[0, -0.5],
                            text=[r"$\overline{V} \text{ (λ=3) }$",
                                  r"$\overline{U} \text{ (λ=6) }$"],
                            mode="text", name="Власні вектори"))


    figure.update_layout(
        title=r"dsadasd",
        font=dict(
            family="Arial",
            size=18,
            color="black"
        )
    )

    figure.update_xaxes(range=[-5, 5])
    figure.update_yaxes(range=[-5, 5])
    figure['layout'].update(width=700, height=600, autosize=False)
    figure.show()

to_plot()
eigs()
