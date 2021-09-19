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


df = pd.read_csv("Muscheln.csv", sep=";")
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

sns.heatmap(df.corr(), cmap='coolwarm')
plt.show()
quit
sns.jointplot(x=df['volumen'], y=df['parasiten_gesamt'], data=df, kind='reg')
sns.pairplot(df)
sns.scatterplot(data=df, x="volumen", y="parasiten_gesamt", hue="zone")


sns.lmplot(x="volumen", y="parasiten_gesamt", hue="zone", data=df,
           palette="muted", height=4, scatter_kws={"s": 50, "alpha": 1})


sns.displot(
    df, x="volumen", col="parasiten_gesamt", row="zone",
    binwidth=3, height=3, facet_kws=dict(margin_titles=True),
)


sns.kdeplot(x=df["parasiten_gesamt"], y=df['volumen'])

sns.kdeplot(x=df["parasiten_gesamt"], y=df['volumen'],
            cmap="Reds", shade=True, bw_adjust=.5)

sns.kdeplot(x=df["parasiten_gesamt"], y=df['volumen'],
            cmap="Blues", shade=True, thresh=0)

df['parasiten_gesamt'].mean()
df.groupby(['zone']).mean()

# what we are predicting
y = df['zone']
# features
X = df[["extinktion_korrigiert", "volumen",
        "m.intestinalis", "m.orientalis", "parasiten_gesamt"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Logisitic regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
print(metrics.classification_report(y_test, y_pred))

# Random forest
# rf_model = RandomForestClassifier()
# rf_model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print(metrics.classification_report(y_test, y_pred))
# feature_df = pd.DataFrame(
#     {'Importance': rf_model.feature_importances_, 'Features': X})
# print(feature_df)

# C-Support Vector Classification
# reg_svc = SVC()
# reg_svc.fit(X_train, y_train)
# y_pred = reg_svc.predict(X_test)
# print(metrics.classification_report(y_test, y_pred))
