import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Insurance Predictor",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# Load CSS
# -----------------------------
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass

# -----------------------------
# Load Model
# -----------------------------
try:
    model = pickle.load(open("model.pkl", "rb"))
except:
    st.error("model.pkl not found")
    st.stop()

# -----------------------------
# Load Dataset
# -----------------------------
try:
    df = pd.read_csv("insurance_data.csv")
except:
    df = None

# -----------------------------
# Hero Section
# -----------------------------
st.markdown("""
<div class='hero'>
<h1>🛡️ Life Insurance Predictor</h1>
<p>Predict whether a customer is likely to purchase life insurance using Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Metrics
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class='card'>
    <div class='card-title'>Dataset Size</div>
    <div class='card-value'>{len(df) if df is not None else 0}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='card'>
    <div class='card-title'>Algorithm</div>
    <div class='card-value'>Logistic Regression</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='card'>
    <div class='card-title'>Prediction</div>
    <div class='card-value'>Binary</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -----------------------------
# Prediction
# -----------------------------
st.markdown("<div class='predict-box'>", unsafe_allow_html=True)

st.subheader("Prediction")

age = st.slider(
    "Select Age",
    18,
    80,
    35
)

if st.button("Predict"):

    probability = model.predict_proba([[age]])[0][1]
    prediction = model.predict([[age]])[0]

    st.progress(float(probability))

    st.write(f"### Probability : {probability*100:.2f}%")

    if prediction == 1:

        st.markdown(f"""
        <div class='success-box'>
        <h3>✅ Customer is likely to Buy Insurance</h3>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class='warning-box'>
        <h3>❌ Customer is unlikely to Buy Insurance</h3>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Dataset
# -----------------------------
if df is not None:

    st.write("")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class='footer'>
Made with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)