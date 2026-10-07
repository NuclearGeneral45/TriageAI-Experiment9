
# Experiment 9 — TriageAI

## Project title
AI-Based Emergency Patient Priority Prediction System

## Important
This is an academic prototype only. It is NOT a medical device and must not be used for real emergency triage.

The included dataset is simulated healthcare data created for demonstrating the complete Experiment 9 workflow. The target labels are generated from a transparent demonstration scoring rule and are not clinical labels.

## Experiment 9 mapping

1. Problem Identification
2. Data Collection
3. Data Preprocessing
4. Model Development
5. Application Development
6. Deployment
7. Innovation Focus

## Requirements
- Python 3.10+
- VS Code recommended

## Windows setup

Open a terminal inside this folder:

```bat
python -m venv venv
```

Activate:

```bat
venv\Scripts\activate
```

Install packages:

```bat
pip install -r requirements.txt
```

Run:

```bat
streamlit run app.py
```

Open the local URL shown by Streamlit, normally:

http://localhost:8501

## Files

- app.py — complete Streamlit web application
- triage_healthcare_data.csv — simulated healthcare dataset
- requirements.txt — Python packages
- Experiment9_Report.docx — practical/journal-ready report
- README.txt — setup and viva guidance

## 30-second viva answer

"I developed a web-based AI application called TriageAI for emergency patient priority prediction. The project follows the Experiment 9 workflow: problem identification, healthcare data collection, preprocessing, model development, application development and deployment. I used a Random Forest classifier and integrated it into a Streamlit web interface. The user enters patient observations and the model predicts low, moderate or high priority. I also added explainable AI using feature importance and class probabilities. The dataset is simulated for academic demonstration, so the application is not intended for real clinical use."
