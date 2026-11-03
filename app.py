import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

@st.cache_resource
def load_all():
    models = {
        "Logistic Regression": joblib.load("lr_model.joblib"),
        "Decision Tree": joblib.load("dt_model.joblib"),
        "Random Forest": joblib.load("rf_model.joblib"),
    }
    return models, joblib.load("scaler.joblib"), joblib.load("columns.joblib")

models, scaler, columns = load_all()

# ---------- Header ----------
st.markdown(
    """
    <div style="background:linear-gradient(90deg,#e74c3c,#c0392b);padding:20px;border-radius:12px;text-align:center;">
        <h1 style="color:white;margin:0;">❤️ Heart Disease Prediction</h1>
        <p style="color:white;margin:4px 0 0 0;">AI-based early risk screening | SDG 3: Good Health and Well-being</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# ---------- Sidebar inputs ----------
st.sidebar.header("🧑‍⚕️ Patient Details")
choice = st.sidebar.selectbox("Model chuno", list(models.keys()))
age = st.sidebar.slider("Age", 1, 100, 50)
sex = st.sidebar.radio("Sex", ["M", "F"], horizontal=True)
cp = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"],
                          help="ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic, TA: Typical Angina")
bp = st.sidebar.slider("Resting BP (mm Hg)", 80, 200, 120)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.sidebar.radio("Fasting BS > 120 mg/dl?", [0, 1], horizontal=True)
ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
ang = st.sidebar.radio("Exercise Angina", ["N", "Y"], horizontal=True)
old = st.sidebar.slider("Oldpeak", 0.0, 7.0, 1.0, 0.1)
slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ---------- Prepare input ----------
row = pd.DataFrame([{
    "Age": age, "Sex": sex, "ChestPainType": cp, "RestingBP": bp,
    "Cholesterol": chol, "FastingBS": fbs, "RestingECG": ecg,
    "MaxHR": hr, "ExerciseAngina": ang, "Oldpeak": old, "ST_Slope": slope
}])
row = pd.get_dummies(row).reindex(columns=columns, fill_value=0)
num = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
row[num] = scaler.transform(row[num])

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Model Comparison", "🌳 Feature Importance"])

with tab1:
    prob = models[choice].predict_proba(row)[0][1]
    pred = models[choice].predict(row)[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", choice)
    c2.metric("Risk Probability", f"{prob*100:.1f}%")
    c3.metric("Result", "High Risk" if pred == 1 else "Low Risk")

    st.progress(float(prob))
    if pred == 1:
        st.error("⚠️ Heart disease ka risk zyada hai. Doctor se zaroor milein.")
    else:
        st.success("✅ Risk kam hai. Healthy lifestyle banaye rakhein.")

    st.subheader("Teeno models ka result")
    cols = st.columns(3)
    for col, (name, m) in zip(cols, models.items()):
        p = m.predict_proba(row)[0][1]
        col.metric(name, f"{p*100:.1f}%", "High Risk" if p >= 0.5 else "Low Risk",
                   delta_color="inverse" if p >= 0.5 else "normal")

with tab2:
    st.subheader("Model Performance (Test set)")
    comp = pd.DataFrame({
        "Accuracy": [89.13, 75.00, 84.78],
        "F1-Score": [90.29, 77.23, 86.14],
        "Recall": [91.18, 76.47, 85.29],
        "Precision": [89.42, 78.00, 87.00],
    }, index=["Logistic Regression", "Decision Tree", "Random Forest"])
    st.dataframe(comp, use_container_width=True)
    metric = st.selectbox("Metric chuno", comp.columns)
    st.bar_chart(comp[metric])
    st.info("Best model: Logistic Regression (89.13% accuracy, 91.18% recall)")

with tab3:
    st.subheader("Random Forest - Top Features")
    fi = pd.Series({
        "ST_Slope_Up": 0.1703, "MaxHR": 0.1224, "ST_Slope_Flat": 0.1108,
        "Oldpeak": 0.1031, "Age": 0.0872, "Cholesterol": 0.0857,
        "ExerciseAngina_Y": 0.0828, "RestingBP": 0.0747,
        "ChestPainType_ATA": 0.0369, "Sex_M": 0.0349,
    })
    st.bar_chart(fi)

# ---------- Footer ----------
st.markdown(
    """
    <hr>
    <div style="text-align:center;color:gray;font-size:14px;">
        Made by Nancy | Built with Streamlit & Scikit-learn<br>
        This is a screening tool, not a replacement for a doctor.
    </div>
    """,
    unsafe_allow_html=True,
)