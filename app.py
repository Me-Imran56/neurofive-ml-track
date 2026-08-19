import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

@st.cache_resource
def load_pipeline():
    return joblib.load("titanic_pipeline.joblib")

pipeline = load_pipeline()

st.title("🚢 Titanic Survival Predictor")
st.write(
    "This app uses a Logistic Regression model (80.45% test accuracy) trained on the "
    "Titanic dataset as part of the Neurofive ML Track. Enter a passenger's details "
    "below and click **Predict** to see the model's prediction."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", options=[1, 2, 3], index=2,
                           help="1 = First class, 2 = Second class, 3 = Third class")
    sex = st.selectbox("Sex", options=["male", "female"])
    age = st.slider("Age", min_value=0, max_value=80, value=30)
    fare = st.number_input("Fare paid ($)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

with col2:
    sibsp = st.number_input("Siblings / Spouses aboard", min_value=0, max_value=8, value=0)
    parch = st.number_input("Parents / Children aboard", min_value=0, max_value=6, value=0)
    embarked = st.selectbox("Port of Embarkation", options=["S", "C", "Q"],
                             help="S = Southampton, C = Cherbourg, Q = Queenstown")
    has_cabin = st.selectbox("Had a recorded cabin?", options=["No", "Yes"])

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    has_cabin_flag = 1 if has_cabin == "Yes" else 0

    input_df = pd.DataFrame([{
        "Age": age,
        "Fare": fare,
        "SibSp": sibsp,
        "Parch": parch,
        "FamilySize": family_size,
        "IsAlone": is_alone,
        "Has_Cabin": has_cabin_flag,
        "Sex": sex,
        "Embarked": embarked,
        "Pclass": pclass,
    }])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0]

    if prediction == 1:
        st.success(f"### ✅ Survived — {probability[1]*100:.1f}% confidence")
    else:
        st.error(f"### ❌ Did not survive — {probability[0]*100:.1f}% confidence")

    st.caption(
        f"Model output — P(did not survive): {probability[0]*100:.1f}% | "
        f"P(survived): {probability[1]*100:.1f}%"
    )

st.divider()
st.caption(
    "Model: Logistic Regression trained inside a scikit-learn Pipeline "
    "(StandardScaler + OneHotEncoder) — built for the Neurofive ML Track."
)
