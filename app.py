import dash
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
avocadePath = os.path.join(CURR_DIR, "avocado.csv")
muschelnPath = os.path.join(CURR_DIR, "muscheln.csv")

data = pd.read_csv(avocadePath, sep=',')
df = pd.read_csv(muschelnPath, sep=';')

data = data.query("type == 'conventional' and region == 'Albany'")
data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
data.sort_values("Date", inplace=True)

fig = px.scatter(df, x="volumen", y="parasiten_gesamt")

app = dash.Dash(__name__)
app.layout = html.Div(
    children=[
        html.H1(children="Mussels on Sylt",),
        html.P(
            children="Analysis of the amount of parasites in mussels dependant on their relative positional space "
            "and to see if one can predict where the mussel was picked from based on various features",
        ),
        dcc.Graph(
            figure=fig
        ),
    ]
)
# app.layout = html.Div(
#     children=[
#         html.H1(children="Avocado Analytics",),
#         html.P(
#             children="Analyze the behavior of avocado prices"
#             " and the number of avocados sold in the US"
#             " between 2015 and 2018",
#         ),
#         dcc.Graph(
#             figure={
#                 "data": [
#                     {
#                         "x": data["Date"],
#                         "y": data["AveragePrice"],
#                         "type": "lines",
#                     },
#                 ],
#                 "layout": {"title": "Average Price of Avocados"},
#             },
#         ),
#     ]
# )

if __name__ == "__main__":
    app.run_server(debug=True)

server = app.server
