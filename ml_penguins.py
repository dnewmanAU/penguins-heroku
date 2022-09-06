import streamlit as st
import pandas as pd
import numpy as np
import pickle


def user_input_params():
    """
    Collect user input parameters from the sidebar.

    Returns
    -------
    DataFrame
        A dataframe containing the selected user input parameters.

    """
    island = st.sidebar.selectbox("Island", ("Biscoe", "Dream", "Torgersen"))
    sex = st.sidebar.selectbox("Sex", ("male", "female"))
    bill_length_mm = st.sidebar.slider("Bill legnth (mm)", 32.1, 59.6, 43.9)
    bill_depth_mm = st.sidebar.slider("Bill depth (mm)", 13.1, 21.5, 17.2)
    flipper_length_mm = st.sidebar.slider("Flipper length (mm)", 172.0, 231.0, 201.0)
    body_mass_g = st.sidebar.slider("Body mass (g)", 2700.0, 6300.0, 4207.0)
    input_data = {
        "island": island,
        "bill_length_mm": bill_length_mm,
        "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g,
        "sex": sex,
    }
    penguin_params = pd.DataFrame(input_data, index=[0])
    return penguin_params


# App Title
st.write(
    """
    # Palmer Penguin Species Prediction

    Predicts the **Palmer Penguin** species using the random forest classifer.

    * **Python libraries:** streamlit, pandas, pickle
    Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
    """
)

# Sidebar title
st.sidebar.header("Penguin Input parameters")

# Let user download an example CSV to upload
st.sidebar.markdown(
    """
[Example CSV](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
"""
)

# Upload box for CSV input parameters
uploaded_csv_params = st.sidebar.file_uploader(
    "Upload your input parameters CSV", type=["csv"]
)
if uploaded_csv_params is not None:
    input_df = pd.read_csv(uploaded_csv_params)
else:
    input_df = user_input_params()

# Read in cleaned penguin dataset
penguins_raw_data = pd.read_csv("penguins_cleaned.csv")
penguins_df = penguins_raw_data.drop(columns=["species"])
# Combine user input parameters with the entire penguins dataset
df = pd.concat([input_df, penguins_df], axis=0)

# Encode ordinal parameters
encode = ["sex", "island"]
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]  # select only the first row containing the user input parameters

# Display the currently selected input parameters
st.subheader("User Input Parameters")
st.write(df)

# Read the saved classification model
penguin_randfor = pickle.load(open("penguins_clf.pkl", "rb"))

# Apply the classification model to make prediction and the probabilities
prediction = penguin_randfor.predict(df)
probability = penguin_randfor.predict_proba(df)

# Assign the species to the prediction index
penguin_species = np.array(["Adelie", "Chinstrap", "Gentoo"])

# Make the prediction
st.subheader("Palmer Penguin Species Prediction")
st.write(penguin_species[prediction])

# Give the probability
st.subheader("Prediction probability")
st.write(probability)
