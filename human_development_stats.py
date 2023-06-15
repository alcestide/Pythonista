from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

#Change to default browser
pio.renderers.default = "chromium"

df = pd.read_csv(r"./Datasets/humandevelopment.csv")
fig = make_subplots(rows=2, cols=2)
values = list(range(200))

fig.add_trace(
    go.Histogram(histfunc='min',
                 y=df['HDI rank'],
                 x=df['Country'],
                 xbins={"end":200.5,"size":1,"start":-1.5},
                 marker={"cmax":200,
                         "cmin":0,
                         "color":list(range(400)),
                         'colorscale':'blues',
                         "reversescale":False},
                 name='Human Development Index'),
            row=1,col=1)

fig.add_trace(
    go.Scatter(x=df['Life expectancy at birth'],
                    y=df['Country'],
                    name='Life expectancy at birth',
                    marker=dict(size=20,
                                color=values,
                                colorscale='viridis',
                                reversescale=True),
                    mode='markers'),
              row=1, col=2)

fig.add_trace(
    go.Histogram(histfunc='min',
                 y=df['Mean years of schooling'],
                 x=df['Country'],
                 xbins={"end":200.5,"size":0.5,"start":-1.5},
                 marker={"cmax":200,
                         "cmin":0,
                         "color":list(range(400)),
                         'colorscale':'amp',
                         "reversescale":True},
                 name='Mean Years of Schooling'),
            row=2,col=1)

fig.add_trace(
    go.Scatter(x=df['Gross national income (GNI) per capita'],
                y=df['Country'],
                name='Gross national income (GNI) per capita',
                marker=dict(size=10,
                            color=values,
                            colorscale='algae',
                            reversescale=False),
                        mode='markers'),
              row=2, col=2)

fig.update_layout(yaxis=dict(autorange="reversed"),
                  xaxis=dict(autorange="reversed"),
                  title_text="<b>Human Development Around the Globe</b>")

fig.update_yaxes(autorange="reversed", row=1, col=2)
fig.update_yaxes(autorange="reversed", row=2, col=2)
fig.update_xaxes(autorange="reversed", row=2, col=2)

fig.show()
