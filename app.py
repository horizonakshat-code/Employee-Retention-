import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------- PAGE ---------------- #

st.set_page_config(
    page_title="Employee Retention Predictor",
    page_icon="💼",
    layout="wide"
)

# ---------------- CSS ---------------- #

try:
    with open("style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
except:
    pass

# ---------------- DATA ---------------- #

@st.cache_data
def load_data():
    df = pd.read_csv("HR_comma_sep.csv")
    df = df.drop_duplicates()
    return df

df = load_data()
# ---------------- MODEL ---------------- #

X = df[[
    "satisfaction_level",
    "average_montly_hours",
    "promotion_last_5years"
]]

y = df["left"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)
st.markdown("""
<div class="hero">
<h1>💼 Employee Retention Predictor</h1>
<p>Predict whether an employee is likely to stay or leave.</p>
</div>
""", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Employees",
        len(df)
    )

with c2:
    st.metric(
        "Model Accuracy",
        f"{accuracy*100:.1f}%"
    )

with c3:
    st.metric(
        "Algorithm",
        "Logistic"
    )
