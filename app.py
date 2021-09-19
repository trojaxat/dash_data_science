import dash
from dash import dcc
from dash import html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from plotly.subplots import make_subplots
import dash_html_components as html
import base64
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn import svm, datasets
# from sklearn import metrics
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import plot_precision_recall_curve
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import average_precision_score

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
muschelnPath = os.path.join(CURR_DIR, "muscheln.csv")

df = pd.read_csv(muschelnPath, sep=';')
df = df.drop(columns=["Unnamed: 0", "nr.", "Unnamed: 13"], axis=1)
df.isna().sum()
df["extinktion_kontrolle"] = [
    x.replace(',', '.') for x in df["extinktion_kontrolle"]]
df["extinktion_1"] = [x.replace(',', '.') for x in df["extinktion_1"]]
df["extinktion_2"] = [x.replace(',', '.') for x in df["extinktion_2"]]
df["extinktion_differenz"] = [
    x.replace(',', '.') for x in df["extinktion_differenz"]]
df["extinktion_korrigiert"] = [
    x.replace(',', '.') for x in df["extinktion_korrigiert"]]

# fig = make_subplots(rows=1, cols=2)
# fig.add_trace(
#     px.scatter(df, x="volumen", y="parasiten_gesamt", color="zone"),
#     row=1, col=1
# )

# fig.add_trace(
#     px.histogram(df, x="volumen", y="parasiten_gesamt",
#                        color="zone", marginal="box", hover_data=df.columns),
#     row=1, col=2
# )

# fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")

figMain = px.scatter(df, x="volumen", y="parasiten_gesamt", color="zone")
figDist = px.histogram(df, x="volumen", y="parasiten_gesamt",
                       color="zone", marginal="box", hover_data=df.columns)

image_filename = 'plots\scatter.png'  # replace with your own image
# encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app = dash.Dash(__name__)
app.layout = html.Div(html.Img(src=app.get_asset_url(image_filename)))
app.layout = html.Div(
    children=[
        html.H1(children="Mussels on Sylt",),
        html.P(
            children="Analysis of the amount of parasites in mussels dependant on their relative positional space "
            "and to see if one can predict where the mussel was picked from based on various features",
        ),
        # dcc.Graph(
        #     figure=fig
        # ),
        dcc.Graph(
            figure=figMain,
            style={
                'height': 'auto',
                # all three widths are needed
                # 'minWidth': '500px', 'width': '500px', 'maxWidth': '500px',
                'whiteSpace': 'normal'
            }
        ),
        dcc.Graph(
            figure=figDist
        ),
    ]
)
# layout = go.Layout(xaxis=dict(domain=[0.0, 0.45]),
#                    xaxis2=dict(domain=[0.55, 1.0]),
#                    yaxis2=dict(overlaying='y',
#                                anchor='free',
#                                position=0.55
#                                )

#                    )
# fig = go.Figure(data=df, layout=layout)
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
