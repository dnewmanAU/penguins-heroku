import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

penguins = pd.read_csv("penguins_cleaned.csv")

df = penguins.copy()
target = "species"  # predict the species of penguin
encode = ["sex", "island"]  # input parameters

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

target_mapper = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}


def target_encode(val):
    return target_mapper[val]


df["species"] = df["species"].apply(target_encode)

x = df.drop("species", axis=1)
y = df["species"]

# Build the model
clf = RandomForestClassifier()
clf.fit(x, y)

# Save the model
pickle.dump(clf, open("penguins_clf.pkl", "wb"))
