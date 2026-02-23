from flask import Flask, request, jsonify
from flask_cors import CORS
from pycaret.classification import load_model, predict_model
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model once
model = load_model("churn_model")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    input_df = pd.DataFrame([data])
    
    prediction = predict_model(model, data=input_df)
    
    result = prediction["prediction_label"][0]
    probability = prediction["prediction_score"][0]
    
    risk = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
    
    return jsonify({
        "prediction": result,
        "probability": float(probability),
        "riskLevel": risk
    })


import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))