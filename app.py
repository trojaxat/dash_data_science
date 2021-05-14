import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import os

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
avocadePath = os.path.join(CURR_DIR, "avocado.csv")

# Hosted
data = pd.read_csv(avocadePath, sep=',')

data = data.query("type == 'conventional' and region == 'Albany'")
data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
data.sort_values("Date", inplace=True)

app = dash.Dash(__name__)
app.layout = html.Div(
    children=[
        html.H1(children="Avocado Analytics",),
        html.P(
            children="Analyze the behavior of avocado prices"
            " and the number of avocados sold in the US"
            " between 2015 and 2018",
        ),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": data["Date"],
                        "y": data["AveragePrice"],
                        "type": "lines",
                    },
                ],
                "layout": {"title": "Average Price of Avocados"},
            },
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)

server = app.server
