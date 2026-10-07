
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="TriageAI",
    page_icon="🚑",
    layout="wide"
)

DATA_FILE = "triage_healthcare_data.csv"

@st.cache_resource
def train_model():
    data = pd.read_csv(DATA_FILE)

    X = data.drop(columns=["triage_priority"])
    y = data["triage_priority"]

    numeric_features = [
        "age", "heart_rate_bpm", "systolic_bp_mmhg",
        "respiratory_rate_bpm", "spo2_percent",
        "temperature_c", "pain_score_0_10"
    ]
    categorical_features = [
        "consciousness", "chest_pain",
        "breathing_difficulty", "trauma"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    return data, pipeline, X_test, y_test, pred

data, pipeline, X_test, y_test, pred = train_model()
accuracy = accuracy_score(y_test, pred)

st.title("🚑 TriageAI")
st.subheader("AI-Based Emergency Patient Priority Prediction System")
st.caption("Experiment No. 9 — Web-based AI application using healthcare data")

st.error(
    "ACADEMIC DEMONSTRATION ONLY — This prototype is not a medical device, "
    "does not replace a qualified clinician, and must not be used for real emergency triage."
)

# Sidebar
st.sidebar.header("Experiment 9")
st.sidebar.write("Problem: Emergency patient prioritization")
st.sidebar.write("AI model: Random Forest")
st.sidebar.write("Application: Streamlit web app")
st.sidebar.write(f"Test accuracy: {accuracy*100:.2f}%")
st.sidebar.divider()
st.sidebar.info(
    "Dataset used in this academic prototype is simulated healthcare data. "
    "Labels were generated from a transparent demonstration scoring rule."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🚑 Patient Assessment",
    "📊 Explainable AI",
    "📈 Model Evaluation",
    "📚 Project Methodology"
])

with tab1:
    st.header("Emergency Patient Assessment")
    st.write("Enter the patient's observed values for an academic risk-priority demonstration.")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age (years)", 18, 100, 45)
        heart_rate = st.number_input("Heart Rate (bpm)", 35, 200, 85)
        systolic_bp = st.number_input("Systolic BP (mmHg)", 60, 230, 120)

    with c2:
        respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", 6, 60, 18)
        spo2 = st.number_input("SpO₂ (%)", 70.0, 100.0, 97.0, step=0.5)
        temperature = st.number_input("Temperature (°C)", 34.0, 42.0, 37.0, step=0.1)

    with c3:
        pain = st.slider("Pain Score (0–10)", 0, 10, 2)
        consciousness = st.selectbox(
            "Consciousness",
            ["Alert", "Confused", "Unresponsive"]
        )
        chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])
        breathing_difficulty = st.selectbox("Breathing Difficulty", ["No", "Yes"])
        trauma = st.selectbox("Trauma", ["No", "Yes"])

    patient = pd.DataFrame([{
        "age": age,
        "heart_rate_bpm": heart_rate,
        "systolic_bp_mmhg": systolic_bp,
        "respiratory_rate_bpm": respiratory_rate,
        "spo2_percent": spo2,
        "temperature_c": temperature,
        "pain_score_0_10": pain,
        "consciousness": consciousness,
        "chest_pain": chest_pain,
        "breathing_difficulty": breathing_difficulty,
        "trauma": trauma
    }])

    if st.button("🔍 Assess Priority", type="primary"):
        prediction = pipeline.predict(patient)[0]
        probabilities = pipeline.predict_proba(patient)[0]
        classes = pipeline.classes_
        confidence = probabilities[list(classes).index(prediction)]

        st.divider()
        st.header("Assessment Output")

        if prediction == "High Priority":
            st.error("🔴 HIGH PRIORITY")
            message = "The model assigns this simulated case to the high-priority category."
        elif prediction == "Moderate Priority":
            st.warning("🟠 MODERATE PRIORITY")
            message = "The model assigns this simulated case to the moderate-priority category."
        else:
            st.success("🟢 LOW PRIORITY")
            message = "The model assigns this simulated case to the low-priority category."

        st.write(message)
        st.metric("Model confidence", f"{confidence*100:.1f}%")

        prob_df = pd.DataFrame({
            "Priority": classes,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        st.bar_chart(prob_df.set_index("Priority"))

        st.info(
            "For this academic prototype, priority means the model's predicted class. "
            "It is not a clinical triage recommendation."
        )

with tab2:
    st.header("Explainable AI")
    st.write(
        "Feature importance shows which input variables contributed most to the "
        "Random Forest's overall decisions on the training data. It does not prove causation."
    )

    preprocessor = pipeline.named_steps["preprocessor"]
    rf = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = rf.feature_importances_

    imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(12)

    st.bar_chart(imp.set_index("Feature"))

    st.subheader("Interpretation")
    st.write(
        "The chart is intended to make the model easier to inspect rather than "
        "presenting the prediction as an unexplained black box."
    )

with tab3:
    st.header("Model Evaluation")
    st.metric("Test Accuracy", f"{accuracy*100:.2f}%")
    st.metric("Test Samples", len(y_test))

    cm = confusion_matrix(
        y_test, pred,
        labels=["High Priority", "Moderate Priority", "Low Priority"]
    )

    st.subheader("Confusion Matrix")
    st.dataframe(pd.DataFrame(
        cm,
        index=["Actual High", "Actual Moderate", "Actual Low"],
        columns=["Predicted High", "Predicted Moderate", "Predicted Low"]
    ))

    st.subheader("Classification Report")
    report = classification_report(
        y_test, pred,
        labels=["High Priority", "Moderate Priority", "Low Priority"],
        output_dict=True,
        zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Dataset Preview")
    st.dataframe(data.head(15), use_container_width=True)

with tab4:
    st.header("Experiment 9 Methodology")

    st.markdown("""
### 1. Problem Identification
Develop a web-based AI application that demonstrates emergency patient
priority classification using healthcare-related inputs.

### 2. Data Collection
A simulated healthcare dataset is included with the project for academic use.
It contains demographic information, vital signs, symptoms and a priority label.

### 3. Data Preprocessing
- Separate input features and target.
- Identify numerical and categorical features.
- Encode categorical variables using One-Hot Encoding.
- Split the data into training and testing sets.

### 4. Model Development
A Random Forest classifier is trained on the processed healthcare data.

### 5. Application Development
Streamlit is used to create an interactive web interface where the user
enters patient observations and receives a model prediction.

### 6. Deployment
The application is run locally through the Streamlit web server.

### 7. Innovation Focus
Explainable AI is added through feature-importance visualization and
class-probability display.
""")

    st.subheader("System Flow")
    st.code("""
Healthcare Data
      ↓
Data Preprocessing
      ↓
Train / Test Split
      ↓
Random Forest Model
      ↓
Streamlit Web Application
      ↓
Patient Inputs
      ↓
Priority Prediction
      ↓
Confidence + Explainability
""")

st.divider()
st.caption("Academic mini-project. Simulated data. Not for clinical use.")
