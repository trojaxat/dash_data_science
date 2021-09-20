import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from plotly.subplots import make_subplots
import dash_html_components as html
import seaborn as sns
import numpy as np

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
muschelnPath = os.path.join(CURR_DIR, "muscheln.csv")

df = pd.read_csv(muschelnPath, sep=';')

# Clean data
df = df.drop(columns=["Unnamed: 0", "nr.", "Unnamed: 13"], axis=1)
df["extinktion_kontrolle"] = [
    x.replace(',', '.') for x in df["extinktion_kontrolle"]]
df["extinktion_1"] = [x.replace(',', '.') for x in df["extinktion_1"]]
df["extinktion_2"] = [x.replace(',', '.') for x in df["extinktion_2"]]
df["extinktion_differenz"] = [
    x.replace(',', '.') for x in df["extinktion_differenz"]]
df["extinktion_korrigiert"] = [
    x.replace(',', '.') for x in df["extinktion_korrigiert"]]

# Organise features
y = df['zone']
X = df[["extinktion_korrigiert", "volumen",
        "m.intestinalis", "m.orientalis", "parasiten_gesamt"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Logisitic regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
clf_report = metrics.classification_report(
    y_test,
    y_pred,
    output_dict=True
)
clf_df = pd.DataFrame(
    clf_report).iloc[:-1, :].T
figLogisticHeatmap = px.imshow(
    clf_df,
    labels=dict(x="Heatmap for Logistic Regression", y="", color="Scale"),
    x=clf_df.columns,
    y=clf_df.index
)
figLogisticHeatmap.update_xaxes(side="top")

# Random forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred = model.predict(X_test)
clf_report = metrics.classification_report(
    y_test,
    y_pred,
    output_dict=True
)
clf_df = pd.DataFrame(
    clf_report).iloc[:-1, :].T
figForestHeatmap = px.imshow(
    clf_df,
    labels=dict(x="Heatmap for Random Forest", y="", color="Scale"),
    x=clf_df.columns,
    y=clf_df.index
)
figForestHeatmap.update_xaxes(side="top")

# C-Support Vector Classification
reg_svc = SVC()
reg_svc.fit(X_train, y_train)
y_pred = reg_svc.predict(X_test)
clf_report = metrics.classification_report(
    y_test,
    y_pred,
    output_dict=True
)
clf_df = pd.DataFrame(
    clf_report).iloc[:-1, :].T
figSvcHeatmap = px.imshow(
    clf_df,
    labels=dict(x="Heatmap for Support Vector Classification",
                y="", color="Scale"),
    x=clf_df.columns,
    y=clf_df.index
)
figSvcHeatmap.update_xaxes(side="top")


figMain = px.scatter(df, x="volumen", y="parasiten_gesamt", color="zone")
figDist = px.histogram(df, x="volumen", y="parasiten_gesamt",
                       color="zone", marginal="box", hover_data=df.columns)

# image_filename = 'plots\scatter.png'  # replace with your own image
# app.layout = html.Div(html.Img(src=app.get_asset_url(image_filename)))

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1(children="Mussels on Sylt",),
    html.P(
        children="Analysis of the amount of parasites in mussels dependant on their relative positional space "
        "and to see if one can predict where the mussel was picked from based on various features",
    ),
    dcc.Graph(
        figure=figMain,
        style={
            'height': 'auto',
            'whiteSpace': 'normal'
        }
    ),
    dcc.Graph(
        figure=figDist
    ),
    html.Div(
        dcc.Graph(
            figure=figLogisticHeatmap
        ),
        style={'width': '33%', 'display': 'inline-block'}
    ),
    html.Div(
        dcc.Graph(
            figure=figForestHeatmap
        ),
        style={'width': '33%', 'display': 'inline-block'}
    ),
    html.Div(
        dcc.Graph(
            figure=figSvcHeatmap
        ),
        style={'width': '33%', 'display': 'inline-block'}
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)

server = app.server
