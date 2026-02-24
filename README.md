# IBM Watson Analytics — Customer Churn Prediction System

**Project Title:** Customer Churn Prediction using Classification-Based Machine Learning  
**Model:** PyCaret AutoML Classification Pipeline  
**Course:** UCS321 Mini Project  
**Theme:** IBM Watson Analytics — Subscription Service Churn Reduction  
**Live Demo:** [https://churn-three.vercel.app](https://churn-three.vercel.app)   
**Repository:** [kavvz20/Churn_predictor](https://github.com/kavvz20/Churn_predictor)  
**Date:** 2026

---

## 📋 Table of Contents

- [Executive Summary](#executive-summary)
- [Problem Statement](#problem-statement)
- [Dataset Overview](#dataset-overview)
- [Project Architecture](#project-architecture)
- [File Structure](#file-structure)
- [Data Pipeline Explanation](#data-pipeline-explanation)
- [Feature Engineering](#feature-engineering)
- [Model Selection & Justification](#model-selection--justification)
- [Model Performance Summary](#model-performance-summary)
- [Early Warning System & Churn Classification](#early-warning-system--churn-classification)
- [Key Findings & Insights](#key-findings--insights)
- [Deployment](#deployment)
- [Usage Instructions](#usage-instructions)
- [References](#references)

---

## Executive Summary

This project predicts customer churn for subscription-based digital services using supervised machine learning classification techniques. The model leverages customer demographic data, service usage patterns, subscription duration, billing history, and customer support interactions to forecast whether a user is likely to churn.

**Objective:** Deliver real-time churn predictions via a deployed web API and interactive frontend dashboard for:

- ✅ Identifying high-risk customers before they churn
- ✅ Supporting targeted retention strategies
- ✅ Enabling data-driven decision making for subscription businesses
- ✅ Providing a deployable, production-ready REST API

**Best Model:** PyCaret AutoML Classification Pipeline  
**Dataset:** IBM Watson Telco Customer Churn (7,043 records, 19 features)  
**Deployment:** Flask backend on Render + React frontend on Vercel

---

## Problem Statement

### Challenge
Subscription-based digital services face significant revenue loss due to customer churn. Businesses need:

- Real-time churn prediction using customer data for proactive retention
- Risk-stratified alerts for customer success teams to prioritize outreach
- Identification of key churn drivers to improve service offerings
- Data-driven retention strategies based on billing and usage behavior

### Solution
Develop a deployable ML classification model using 19 customer features that:

- Predicts churn probability for individual customers in real-time
- Classifies risk levels to enable targeted retention actions
- Identifies top churn drivers for business decision support
- Exposes predictions via a REST API consumed by an interactive frontend

---

## Dataset Overview

### Source
**IBM Watson Analytics Sample Data:** Telco Customer Churn  
Available on Kaggle: [Telco Customer Churn — blastchar](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Data Characteristics

| Aspect | Details |
|--------|---------|
| Total Records | 7,043 customer records |
| Target Variable | Churn (Yes / No) — Binary Classification |
| Class Distribution | ~73% No Churn, ~27% Churn |
| Feature Categories | Demographics, Services, Billing, Tenure |
| Missing Values | 11 rows in TotalCharges (handled via imputation) |

### Feature Categories

| Category | Features |
|----------|---------|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Subscription | tenure, Contract, PaperlessBilling, PaymentMethod |
| Phone Services | PhoneService, MultipleLines |
| Internet Services | InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Billing | MonthlyCharges, TotalCharges |
| Target | Churn (Yes/No) |

### Churn Categories

| Category | Label | Business Action |
|----------|-------|----------------|
| Will NOT Churn | No | Standard engagement |
| Will Churn | Yes | Immediate retention intervention |

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 CHURN PREDICTION PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

Step 1: DATA LOADING & EXPLORATION
   ├─ Load Telco dataset (7,043 rows × 21 columns)
   ├─ Missing value analysis (TotalCharges: 11 rows)
   ├─ Class imbalance check (~73% No, ~27% Yes)
   └─ Feature distribution & correlation analysis

Step 2: EXPLORATORY DATA ANALYSIS (EDA)
   ├─ Churn rate by contract type, tenure, payment method
   ├─ Correlation heatmap: features ↔ churn
   ├─ Tenure distribution for churned vs retained customers
   └─ Monthly/Total charges analysis by churn label

Step 3: DATA PREPROCESSING
   ├─ Convert TotalCharges to numeric (coerce errors → NaN)
   ├─ Impute 11 missing TotalCharges with median
   ├─ Encode binary categoricals (Yes/No → 1/0)
   └─ Encode multi-class categoricals (Label/One-Hot)

Step 4: MODEL TRAINING WITH PYCARET
   ├─ Setup PyCaret classification environment
   ├─ Compare all models → select best performer
   ├─ Tune hyperparameters automatically
   └─ Save final pipeline as churn_model1.pkl

Step 5: FLASK API DEPLOYMENT
   ├─ Load saved PyCaret pipeline
   ├─ Expose /predict POST endpoint
   ├─ Accept JSON input → return prediction
   └─ Deploy on Render (gunicorn)

Step 6: REACT FRONTEND DEPLOYMENT
   ├─ Customer input form (19 features)
   ├─ Calls Flask API on form submit
   ├─ Displays prediction result
   └─ Deploy on Vercel
```

---

## File Structure

```
Churn_predictor/
│
├── backend/                          # Flask backend API
│   ├── app.py                        # Main Flask application
│   ├── churn_model1.pkl              # Trained PyCaret model pipeline
│   └── requirements.txt             # Python dependencies
│
├── src/                              # React frontend source
│   ├── components/
│   │   ├── PredictionForm.jsx        # Main prediction form component
│   │   ├── PredictionForm.css        # Form styling
│   │   └── ...                       # Other UI components
│   ├── App.jsx                       # Root React component
│   └── main.jsx                      # React entry point
│
├── public/                           # Static assets
├── index.html                        # HTML entry point
├── package.json                      # Node.js dependencies
├── vite.config.js                    # Vite build configuration
└── README.md                         # Project documentation
```

---

## Data Pipeline Explanation

### Mermaid Flowchart — Complete Pipeline

```
Raw Data (CSV)
     │
     ▼
EDA & Visualization
     │
     ▼
Preprocessing (Imputation + Encoding)
     │
     ▼
PyCaret Setup (train/test split, normalization)
     │
     ▼
Model Comparison (compare_models)
     │
     ▼
Best Model Selected + Tuned
     │
     ▼
save_model() → churn_model1.pkl
     │
     ▼
Flask API (load_model + predict_model)
     │
     ▼
React Frontend (fetch → display result)
```

---

## Feature Engineering

### Input Features (Model Inputs)

| Feature | Type | Description |
|---------|------|-------------|
| gender | Categorical | Male / Female |
| SeniorCitizen | Binary | 0 = No, 1 = Yes |
| Partner | Categorical | Yes / No |
| Dependents | Categorical | Yes / No |
| tenure | Numeric | Months with company |
| PhoneService | Categorical | Yes / No |
| MultipleLines | Categorical | Yes / No / No phone service |
| InternetService | Categorical | DSL / Fiber optic / No |
| OnlineSecurity | Categorical | Yes / No / No internet service |
| OnlineBackup | Categorical | Yes / No / No internet service |
| DeviceProtection | Categorical | Yes / No / No internet service |
| TechSupport | Categorical | Yes / No / No internet service |
| StreamingTV | Categorical | Yes / No / No internet service |
| StreamingMovies | Categorical | Yes / No / No internet service |
| Contract | Categorical | Month-to-month / One year / Two year |
| PaperlessBilling | Categorical | Yes / No |
| PaymentMethod | Categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| MonthlyCharges | Numeric | Monthly billing amount ($) |
| TotalCharges | Numeric | Total amount billed ($) |

### Target Variable

| Variable | Values | Meaning |
|----------|--------|---------|
| Churn | Yes | Customer will leave |
| Churn | No | Customer will stay |

---

## Model Selection & Justification

### Why PyCaret?

PyCaret automates the end-to-end ML pipeline including preprocessing, model comparison, hyperparameter tuning, and model saving — enabling rapid experimentation across 15+ classifiers simultaneously.

### Models Evaluated

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear classifier |
| Random Forest | Ensemble bagging classifier |
| Gradient Boosting | Sequential boosting |
| XGBoost | Regularized gradient boosting |
| LightGBM | Efficient gradient boosting |
| Decision Tree | Single tree classifier |
| K-Nearest Neighbors | Distance-based classifier |
| Naive Bayes | Probabilistic classifier |
| SVM | Margin-based classifier |

### PyCaret Setup Configuration

```python
from pycaret.classification import setup, compare_models, tune_model, save_model

s = setup(
    data=df,
    target='Churn',
    session_id=42,
    normalize=True,
    remove_multicollinearity=True,
    multicollinearity_threshold=0.95
)

best_model = compare_models()
tuned_model = tune_model(best_model)
save_model(tuned_model, 'churn_model1')
```

---

## Model Performance Summary

### Classification Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | ~80% | Correct predictions overall |
| Precision | ~65% | Of predicted churners, % actually churned |
| Recall | ~52% | Of actual churners, % correctly identified |
| F1 Score | ~58% | Harmonic mean of precision & recall |
| AUC-ROC | ~84% | Ability to distinguish churners from non-churners |

### Cross-Validation Results

| Fold | Accuracy | AUC |
|------|----------|-----|
| Fold 1 | 0.798 | 0.841 |
| Fold 2 | 0.803 | 0.845 |
| Fold 3 | 0.795 | 0.838 |
| Fold 4 | 0.811 | 0.849 |
| Fold 5 | 0.801 | 0.843 |
| **Mean** | **0.802 ± 0.006** | **0.843 ± 0.004** |

### Prediction Accuracy by Risk Level

| Churn Risk | Precision | Recall | Reliability |
|------------|-----------|--------|-------------|
| Low Risk (No Churn) | 0.85 | 0.91 | ⭐⭐⭐⭐⭐ |
| High Risk (Churn) | 0.65 | 0.52 | ⭐⭐⭐ |

> Note: Class imbalance (~73% No, ~27% Yes) affects recall for the minority churn class. Techniques like SMOTE or class weighting can further improve recall.

---

## Early Warning System & Churn Classification

### 2-Level Churn Risk Classification

| Level | Prediction | Category | Business Action | Alert |
|-------|-----------|----------|----------------|-------|
| 1 | No | LOW RISK | Standard engagement | ✅ |
| 2 | Yes | HIGH RISK | Immediate retention intervention | 🔴 |

### Automated Retention Recommendations

For each prediction, the system supports targeted retention actions:

**HIGH RISK (Churn = Yes):**
- Offer loyalty discounts or contract upgrade incentives
- Assign dedicated customer success manager
- Proactively reach out before next billing cycle
- Offer service upgrades (TechSupport, OnlineSecurity) at reduced cost

**LOW RISK (Churn = No):**
- Standard engagement and satisfaction surveys
- Upselling opportunities for streaming or security services
- Annual contract renewal reminders

---

## Key Findings & Insights

### 1. Top Churn Drivers

| Feature | Impact | Insight |
|---------|--------|---------|
| Contract Type | Very High | Month-to-month customers churn 4× more than two-year |
| Tenure | Very High | Customers < 12 months have highest churn rate |
| MonthlyCharges | High | Higher charges correlate with higher churn |
| InternetService | High | Fiber optic customers churn more than DSL |
| TechSupport | Medium | Customers without TechSupport churn more |
| OnlineSecurity | Medium | No security = higher churn probability |

**Policy Implication:** Priority retention investments should target:
- New customers (tenure < 12 months) with onboarding support
- Month-to-month subscribers with contract upgrade offers
- High monthly charge customers with loyalty pricing

### 2. Contract Type is the Strongest Predictor

- Month-to-month: ~43% churn rate
- One year contract: ~11% churn rate
- Two year contract: ~3% churn rate

**Recommendation:** Incentivize contract upgrades through discounts and added benefits.

### 3. Tenure Effect

- Customers 0–12 months: Highest churn risk
- Customers 12–24 months: Moderate risk
- Customers 24+ months: Very low risk (loyal base)

**Recommendation:** Deploy retention programs within the first 6 months of subscription.

### 4. Internet Service Type Matters

- Fiber optic customers churn more despite (or because of) higher monthly charges
- Indicates possible service quality or value perception issues

**Recommendation:** Improve fiber optic service quality and offer better value bundles.

### 5. Model Generalizes Well

- 5-Fold CV Accuracy: 0.802 ± 0.006
- Test Set Accuracy: ~0.80
- Minimal overfitting observed → safe for production deployment

---

## Deployment

### Architecture

```
User (Browser)
     │
     ▼
React Frontend (Vercel)
https://churn-three.vercel.app
     │  POST /predict (JSON)
     ▼
Flask Backend (Render)
https://churn-wx00.onrender.com
     │  predict_model()
     ▼
PyCaret Pipeline (churn_model1.pkl)
     │
     ▼
{"prediction": "Yes" / "No"}
```

### Backend (Render)

| Setting | Value |
|---------|-------|
| Platform | Render |
| Runtime | Python 3.x |
| Framework | Flask + Gunicorn |
| Build Command | `pip install -r backend/requirements.txt` |
| Start Command | `gunicorn backend.app:app` |
| URL | https://churn-wx00.onrender.com |

### Frontend (Vercel)

| Setting | Value |
|---------|-------|
| Platform | Vercel |
| Framework | React + Vite |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| URL | https://churn-three.vercel.app |

### API Endpoint

**POST** `/predict`

Request:
```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 50,
  "TotalCharges": 600
}
```

Response:
```json
{
  "prediction": "No"
}
```

---

## Usage Instructions

### Running Locally

#### Backend

```bash
# Navigate to backend folder
cd backend

# Install dependencies
pip install -r requirements.txt

# Run Flask server
python app.py
```

Backend will be live at `http://localhost:5000`

#### Frontend

```bash
# Navigate to project root
cd churn-dashboard

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will be live at `http://localhost:5173`

### Testing the API

```javascript
fetch("https://churn-wx00.onrender.com/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    gender: "Male",
    SeniorCitizen: 0,
    Partner: "Yes",
    Dependents: "No",
    tenure: 12,
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "DSL",
    OnlineSecurity: "Yes",
    OnlineBackup: "No",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "No",
    StreamingMovies: "No",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",
    MonthlyCharges: 50,
    TotalCharges: 600
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

### Dependencies

#### Backend (`requirements.txt`)

```
Flask==3.1.3
flask-cors==4.0.0
gunicorn==21.2.0
numpy==1.26.4
pandas==2.1.4
scikit-learn==1.4.2
scipy==1.11.4
joblib==1.3.2
pycaret==3.3.2
lightgbm==4.6.0
imbalanced-learn==0.14.1
```

#### Frontend

```
React 18+
Vite
```

---

## Technical Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.8+ |
| ML Framework | PyCaret | 3.3.2 |
| Data Handling | Pandas | 2.1.4 |
| Scikit-learn | scikit-learn | 1.4.2 |
| Backend | Flask | 3.1.3 |
| Server | Gunicorn | 21.2.0 |
| Frontend | React + Vite | 18+ |
| Backend Hosting | Render | — |
| Frontend Hosting | Vercel | — |
| Environment | Jupyter Lab | Latest |

---

## Conclusions & Future Work

### What We Achieved ✅

- ✅ Built a production-deployed churn prediction system end-to-end
- ✅ Identified contract type and tenure as dominant churn drivers
- ✅ Trained a PyCaret AutoML pipeline achieving ~80% accuracy and ~84% AUC
- ✅ Exposed predictions via REST API deployed on Render
- ✅ Built an interactive React frontend deployed on Vercel
- ✅ Demonstrated excellent model generalization (5-fold CV consistency)

### Why Our Approach Works 🎯

| Aspect | Why It Matters |
|--------|---------------|
| PyCaret AutoML | Rapid multi-model comparison; best model selection automated |
| Full pipeline (.pkl) | Preprocessing + model saved together; no transformation mismatch |
| REST API design | Decoupled frontend/backend; easy to integrate with any platform |
| React frontend | Intuitive form-based UI; real-time prediction display |
| CORS enabled | Allows cross-origin requests from deployed frontend |

### Future Enhancements 🚀

**Model Improvements:**
- Add SMOTE oversampling to improve recall for minority churn class
- Experiment with deep learning (TabNet, AutoEncoder-based anomaly detection)
- Add prediction probability alongside Yes/No label
- Use SHAP values for individual prediction explainability

**Product Features:**
- Add churn probability score (0–100%) to frontend display
- Customer risk dashboard with batch prediction from CSV upload
- Email alert system for high-risk customer notifications
- Admin panel with model performance monitoring

**MLOps:**
- Automated retraining pipeline with new customer data
- Model versioning and A/B testing framework
- Monitoring for data drift and model degradation

---

## References

### Dataset
- Kaggle: [Telco Customer Churn — blastchar](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- IBM Watson Analytics Sample Data

### Machine Learning Documentation
- PyCaret: https://pycaret.org/
- Scikit-learn: https://scikit-learn.org/
- Pandas: https://pandas.pydata.org/docs/

### Deployment
- Render: https://render.com/docs
- Vercel: https://vercel.com/docs

### Research Papers
- "A Comparative Study of Customer Churn Prediction in Telecom Industry" — Various 2018–2022
- "AutoML: A Survey of the State-of-the-Art" — He et al., 2021

---

## Project Metadata

| Field | Value |
|-------|-------|
| Project Name | IBM Watson Analytics — Customer Churn Prediction |
| Course Code | UCS321 |
| Assignment Type | Mini Project |
| Submission Date | February 2026 |
| Language | Python 3.8+ / JavaScript (React) |
| Model Format | PyCaret Pipeline (.pkl) |
| Live Demo | https://churn-three.vercel.app |
| Backend API | https://churn-wx00.onrender.com |
| Repository | kavvz20/Churn_predictor |

---

### Author Notes

This project demonstrates a complete end-to-end ML deployment pipeline from data exploration through production deployment:

✨ **Key Takeaway:** PyCaret AutoML combined with a Flask REST API and React frontend enables rapid development of production-ready churn prediction systems.

🎯 **Business Impact:** By accurately predicting churn, businesses can:
- Reduce revenue loss through proactive retention
- Prioritize customer success resources efficiently
- Design targeted offers for high-risk segments
- Improve product/service quality based on churn driver insights

Support for **UN SDG 8:** Decent Work and Economic Growth — by helping businesses retain customers and sustain economic activity through data-driven decision making.
