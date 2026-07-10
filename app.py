import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Employee Retention Predictor",
    page_icon="💼",
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
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Model Error: {e}")
    st.stop()

# -----------------------------
# Load Dataset
# -----------------------------
try:
    df = pd.read_csv("HR_comma_sep.csv")
except:
    df = None

# -----------------------------
# Hero
# -----------------------------
st.markdown("""
<div class="hero">
<h1>💼 Employee Retention Predictor</h1>
<p>Predict whether an employee is likely to stay or leave the company.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Input Form
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.50)
    evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.70)
    projects = st.number_input("Number of Projects", 2, 10, 4)
    monthly_hours = st.number_input("Average Monthly Hours", 90, 320, 200)
    years = st.number_input("Years at Company", 1, 10, 3)

with col2:
    accident = st.selectbox("Work Accident", ["No", "Yes"])
    promotion = st.selectbox("Promotion in Last 5 Years", ["No", "Yes"])

    department = st.selectbox(
        "Department",
        [
            "sales",
            "accounting",
            "hr",
            "technical",
            "support",
            "management",
            "IT",
            "product_mng",
            "marketing",
            "RandD"
        ]
    )

    salary = st.selectbox(
        "Salary",
        ["low", "medium", "high"]
    )

# Convert values
accident = 1 if accident == "Yes" else 0
promotion = 1 if promotion == "Yes" else 0

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict"):

    input_df = pd.DataFrame({
        "satisfaction_level":[satisfaction],
        "last_evaluation":[evaluation],
        "number_project":[projects],
        "average_montly_hours":[monthly_hours],
        "time_spend_company":[years],
        "Work_accident":[accident],
        "promotion_last_5years":[promotion],
        "Department":[department],
        "salary":[salary]
    })

    try:

        probability = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.progress(float(probability))

        st.metric(
            "Probability of Leaving",
            f"{probability*100:.2f}%"
        )

        if prediction == 1:

            st.markdown("""
            <div class="warning-box">
            <h3>⚠ Employee is likely to Leave</h3>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="success-box">
            <h3>✅ Employee is likely to Stay</h3>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(e)

# -----------------------------
# Dataset
# -----------------------------
if df is not None:

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
Employee Retention Prediction using Logistic Regression
</div>
""", unsafe_allow_html=True)
