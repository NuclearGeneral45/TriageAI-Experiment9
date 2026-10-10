import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="TriageAI", page_icon="🚑", layout="wide")

DATA_FILE = "triage_healthcare_data.csv"
PRIORITY_ORDER = ["High Priority", "Moderate Priority", "Low Priority"]


@st.cache_resource
def train_model():
    data = pd.read_csv(DATA_FILE)
    feature_columns = [
        "age", "heart_rate_bpm", "systolic_bp_mmhg",
        "respiratory_rate_bpm", "spo2_percent", "temperature_c",
        "pain_score_0_10", "consciousness", "chest_pain",
        "breathing_difficulty", "trauma"
    ]
    X = data[feature_columns]
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
    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    return data, pipeline, X_test, y_test, pred


data, pipeline, X_test, y_test, pred = train_model()
accuracy = accuracy_score(y_test, pred)

if "patient_records" not in st.session_state:
    st.session_state.patient_records = []

st.title("🚑 TriageAI")
st.subheader("AI-Based Emergency Patient Priority Prediction System")
st.caption("Experiment No. 9 — Web-based AI application using simulated healthcare data")
st.error(
    "ACADEMIC DEMONSTRATION ONLY — This prototype is not a medical device, "
    "does not replace a qualified clinician, and must not be used for real emergency triage."
)

st.sidebar.header("Experiment 9")
st.sidebar.write("AI model: Random Forest")
st.sidebar.write("Application: Streamlit web app")
st.sidebar.write(f"Model test accuracy: {accuracy * 100:.2f}%")
st.sidebar.write(f"Assessments in this session: {len(st.session_state.patient_records)}")
st.sidebar.info(
    "The training dataset is simulated. Patient assessment records entered here are kept "
    "in the current Streamlit session and may be cleared if the app restarts or the session ends. "
    "Do not enter real or identifiable medical information."
)

tab_dashboard, tab_assessment, tab_xai, tab_eval, tab_method = st.tabs([
    "📊 Dashboard",
    "🚑 Patient Assessment",
    "🔎 Explainable AI",
    "📈 Model Evaluation",
    "📚 Project Methodology"
])


def priority_counts(records_df):
    if records_df.empty:
        return pd.DataFrame({
            "Priority": PRIORITY_ORDER,
            "Patients": [0, 0, 0],
            "Percentage": [0.0, 0.0, 0.0]
        })

    counts = records_df["Predicted Priority"].value_counts()
    total = len(records_df)
    return pd.DataFrame({
        "Priority": PRIORITY_ORDER,
        "Patients": [int(counts.get(priority, 0)) for priority in PRIORITY_ORDER],
        "Percentage": [
            round(counts.get(priority, 0) / total * 100, 1)
            for priority in PRIORITY_ORDER
        ]
    })


with tab_dashboard:
    st.header("Patient Priority Dashboard")
    st.write("This dashboard summarizes assessments made during the current app session.")
    records = pd.DataFrame(st.session_state.patient_records)
    summary = priority_counts(records)
    total = len(records)

    high_n = int(summary.loc[summary["Priority"] == "High Priority", "Patients"].iloc[0])
    moderate_n = int(summary.loc[summary["Priority"] == "Moderate Priority", "Patients"].iloc[0])
    low_n = int(summary.loc[summary["Priority"] == "Low Priority", "Patients"].iloc[0])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Patients Assessed", total)
    m2.metric("🔴 High Priority", high_n, f"{high_n / total * 100:.1f}%" if total else "0.0%")
    m3.metric("🟠 Moderate Priority", moderate_n, f"{moderate_n / total * 100:.1f}%" if total else "0.0%")
    m4.metric("🟢 Low Priority", low_n, f"{low_n / total * 100:.1f}%" if total else "0.0%")

    st.subheader("Category-wise patient count")
    st.bar_chart(summary.set_index("Priority")[["Patients"]], use_container_width=True)

    st.subheader("Category-wise percentage")
    st.bar_chart(summary.set_index("Priority")[["Percentage"]], use_container_width=True)

    st.subheader("Individual priority percentages")
    graph_col1, graph_col2, graph_col3 = st.columns(3)

    with graph_col1:
        st.markdown("### 🔴 High Priority")
        high_pct = float(summary.loc[summary["Priority"] == "High Priority", "Percentage"].iloc[0])
        st.metric("Share of assessed patients", f"{high_pct:.1f}%")
        st.progress(int(round(high_pct)))

    with graph_col2:
        st.markdown("### 🟠 Moderate Priority")
        moderate_pct = float(summary.loc[summary["Priority"] == "Moderate Priority", "Percentage"].iloc[0])
        st.metric("Share of assessed patients", f"{moderate_pct:.1f}%")
        st.progress(int(round(moderate_pct)))

    with graph_col3:
        st.markdown("### 🟢 Low Priority")
        low_pct = float(summary.loc[summary["Priority"] == "Low Priority", "Percentage"].iloc[0])
        st.metric("Share of assessed patients", f"{low_pct:.1f}%")
        st.progress(int(round(low_pct)))

    st.dataframe(summary, hide_index=True, use_container_width=True)

    st.subheader("Patient lists and confidence graphs by category")
    cat_high, cat_mod, cat_low = st.tabs([
        "🔴 High Priority", "🟠 Moderate Priority", "🟢 Low Priority"
    ])

    for category_tab, category in [
        (cat_high, "High Priority"),
        (cat_mod, "Moderate Priority"),
        (cat_low, "Low Priority")
    ]:
        with category_tab:
            if records.empty:
                st.info("No patient assessments yet. Add a demo patient in the Patient Assessment tab.")
            else:
                category_records = records[
                    records["Predicted Priority"] == category
                ].copy()

                if category_records.empty:
                    st.info("No patients in this category yet.")
                else:
                    st.subheader(f"{category} — Model Confidence by Patient")
                    chart_data = category_records[
                        ["Patient Name", "Model Confidence (%)"]
                    ].copy()
                    chart_data = chart_data.sort_values(
                        "Model Confidence (%)", ascending=False
                    )
                    st.line_chart(
                        chart_data.set_index("Patient Name"),
                        y="Model Confidence (%)",
                        y_label="Model confidence (%)",
                        x_label="Patient name",
                        use_container_width=True
                    )
                    st.caption(
                        "This shows the model's confidence in its predicted category, "
                        "not a medically validated risk percentage."
                    )
                    st.dataframe(
                        category_records,
                        hide_index=True,
                        use_container_width=True
                    )

    if not records.empty:
        csv_bytes = records.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download current-session patient records (CSV)",
            data=csv_bytes,
            file_name="triageai_patient_assessments.csv",
            mime="text/csv"
        )
        if st.button("Clear current-session records", type="secondary"):
            st.session_state.patient_records = []
            if "last_assessment" in st.session_state:
                del st.session_state["last_assessment"]
            st.rerun()


with tab_assessment:
    st.header("Patient Assessment")
    st.write("Enter a demo identity label and observed values, then run the academic model.")
    st.info("For privacy, use fictional/demo names only. Do not enter real patient identifiers or clinical data.")

    with st.form("patient_assessment_form"):
        st.subheader("Patient identification")
        p1, p2 = st.columns(2)
        with p1:
            patient_name = st.text_input(
                "Patient name / demo name",
                placeholder="e.g., Demo Patient 001"
            )
        with p2:
            patient_id = st.text_input(
                "Patient ID / record label",
                placeholder="e.g., DEMO-001"
            )

        st.subheader("Patient observations")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=45)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=35, max_value=200, value=85)
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=230, value=120)
            respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=6, max_value=60, value=18)
        with c2:
            spo2 = st.number_input("SpO₂ (%)", min_value=70.0, max_value=100.0, value=97.0, step=0.5)
            temperature = st.number_input("Temperature (°C)", min_value=34.0, max_value=42.0, value=37.0, step=0.1)
            pain = st.slider("Pain Score (0–10)", min_value=0, max_value=10, value=2)
            consciousness = st.selectbox("Consciousness", ["Alert", "Confused", "Unresponsive"])
        with c3:
            chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])
            breathing_difficulty = st.selectbox("Breathing Difficulty", ["No", "Yes"])
            trauma = st.selectbox("Trauma", ["No", "Yes"])

        submitted = st.form_submit_button("🔍 Assess and Add to Dashboard", type="primary")

    if submitted:
        if not patient_name.strip():
            st.error("Please enter a demo name so the assessment can be identified.")
        elif not patient_id.strip():
            st.error("Please enter a patient ID / record label.")
        else:
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

            prediction = pipeline.predict(patient)[0]
            probabilities = pipeline.predict_proba(patient)[0]
            classes = list(pipeline.classes_)
            confidence = float(probabilities[classes.index(prediction)])
            probability_map = {
                str(cls): float(prob)
                for cls, prob in zip(classes, probabilities)
            }

            new_record = {
                "Assessment Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Patient Name": patient_name.strip(),
                "Patient ID": patient_id.strip(),
                "Age": int(age),
                "Heart Rate": int(heart_rate),
                "Systolic BP": int(systolic_bp),
                "Respiratory Rate": int(respiratory_rate),
                "SpO2 (%)": float(spo2),
                "Temperature (C)": float(temperature),
                "Pain Score": int(pain),
                "Consciousness": consciousness,
                "Chest Pain": chest_pain,
                "Breathing Difficulty": breathing_difficulty,
                "Trauma": trauma,
                "Predicted Priority": str(prediction),
                "Model Confidence (%)": round(confidence * 100, 1),
                "High Priority (%)": round(probability_map.get("High Priority", 0) * 100, 1),
                "Moderate Priority (%)": round(probability_map.get("Moderate Priority", 0) * 100, 1),
                "Low Priority (%)": round(probability_map.get("Low Priority", 0) * 100, 1)
            }
               existing_ids = {
        str(record["Patient ID"]).strip()
        for record in st.session_state.patient_records
    }

    if str(new_record["Patient ID"]).strip() in existing_ids:
        st.error("Patient ID already exists. Please enter a unique ID.")
    else:
        st.session_state.patient_records.append(new_record)
        st.session_state.last_assessment = new_record
        st.rerun()

    if "last_assessment" in st.session_state:
        last = st.session_state.last_assessment
        st.divider()
        st.subheader("Latest assessment result")
        st.write(f"**Patient:** {last['Patient Name']} | **ID:** {last['Patient ID']}")

        if last["Predicted Priority"] == "High Priority":
            st.error(f"🔴 HIGH PRIORITY — model confidence {last['Model Confidence (%)']:.1f}%")
        elif last["Predicted Priority"] == "Moderate Priority":
            st.warning(f"🟠 MODERATE PRIORITY — model confidence {last['Model Confidence (%)']:.1f}%")
        else:
            st.success(f"🟢 LOW PRIORITY — model confidence {last['Model Confidence (%)']:.1f}%")

        probability_df = pd.DataFrame({
            "Priority": PRIORITY_ORDER,
            "Probability (%)": [
                last["High Priority (%)"],
                last["Moderate Priority (%)"],
                last["Low Priority (%)"]
            ]
        })
        st.bar_chart(probability_df.set_index("Priority"))
        st.caption("This is an academic model output, not a clinical recommendation.")


with tab_xai:
    st.header("Explainable AI")
    st.write(
        "Feature importance indicates which inputs were influential across the trained Random Forest model. "
        "It does not prove causation and should not be interpreted as medical advice."
    )
    preprocessor = pipeline.named_steps["preprocessor"]
    rf = pipeline.named_steps["model"]
    raw_feature_names = preprocessor.get_feature_names_out()
    importances = rf.feature_importances_

    feature_labels = {
        "age": "Age",
        "heart_rate_bpm": "Heart Rate (bpm)",
        "systolic_bp_mmhg": "Systolic Blood Pressure (mmHg)",
        "respiratory_rate_bpm": "Respiratory Rate (breaths/min)",
        "spo2_percent": "Oxygen Saturation (SpO2)",
        "temperature_c": "Body Temperature (°C)",
        "pain_score_0_10": "Pain Score (0–10)",
        "consciousness_alert": "Consciousness Alert",
        "consciousness_confused": "Consciousness Confused",
        "consciousness_unresponsive": "Consciousness Unresponsive",
        "chest_pain_no": "Chest Pain — No",
        "chest_pain_yes": "Chest Pain — Yes",
        "breathing_difficulty_no": "Breathing Difficulty — No",
        "breathing_difficulty_yes": "Breathing Difficulty — Yes",
        "trauma_no": "Trauma — No",
        "trauma_yes": "Trauma — Yes"
    }

    feature_names = [
        feature_labels.get(
            name.replace("num__", "").replace("cat__", ""),
            name.replace("num__", "").replace("cat__", "").replace("_", " ").title()
        )
        for name in raw_feature_names
    ]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })
    importance_df = importance_df.sort_values(
        "Importance", ascending=False
    ).head(12)
    st.bar_chart(importance_df.set_index("Feature"))


with tab_eval:
    st.header("Model Evaluation")
    m1, m2 = st.columns(2)
    m1.metric("Test Accuracy", f"{accuracy * 100:.2f}%")
    m2.metric("Test Samples", len(y_test))

    cm = confusion_matrix(y_test, pred, labels=PRIORITY_ORDER)
    st.subheader("Confusion Matrix")
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual High", "Actual Moderate", "Actual Low"],
            columns=["Predicted High", "Predicted Moderate", "Predicted Low"]
        ),
        use_container_width=True
    )

    st.subheader("Classification Report")
    report = classification_report(
        y_test, pred, labels=PRIORITY_ORDER,
        output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    st.subheader("Training Dataset Preview")
    st.dataframe(data.head(15), use_container_width=True)


with tab_method:
    st.header("Experiment 9 Methodology")
    st.markdown("""
### 1. Problem Identification
Develop a web-based AI application that demonstrates emergency patient-priority classification.

### 2. Data Collection
A simulated dataset is provided for academic demonstration.

### 3. Data Preprocessing
Numerical and categorical features are separated, categorical values are one-hot encoded, and the dataset is split into training and test sets.

### 4. Model Development
A Random Forest classifier is trained and evaluated on held-out data.

### 5. Application Development
Streamlit provides patient assessment, patient identification labels, a three-category dashboard, percentage charts, and exportable session records.

### 6. Deployment
The app is deployed through Streamlit Community Cloud and the source code is hosted in GitHub.

### 7. Innovation Focus
The dashboard groups assessed demo records into High, Moderate, and Low priority categories and shows each category's percentage of the session's assessed records.
""")
    st.subheader("System Flow")
    st.code("""Simulated Healthcare Data
        ↓
Data Preprocessing
        ↓
Random Forest Model
        ↓
Patient Name / Demo ID + Observed Values
        ↓
Priority Prediction
        ↓
Session Dashboard: High / Moderate / Low
        ↓
Counts + Percentage Charts + CSV Export""")

st.divider()
st.caption(
    "Academic mini-project using simulated data. Not for clinical use. "
    "Dashboard records are session-only and are not a permanent medical record."
)
