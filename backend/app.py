from flask import Flask, request, jsonify
from flask_cors import CORS
from pycaret.classification import load_model, predict_model
import pandas as pd
import os
import logging
import traceback

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load model once
try:
    model_path = os.path.join(os.path.dirname(__file__), "churn_model1")
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Model loading failed:", str(e))
    print(traceback.format_exc())
    model = None


@app.route("/")
def home():
    return "Backend is working"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        print("Received JSON:", data)

        df = pd.DataFrame([data])
        print("Constructed DataFrame:")
        print(df)

        prediction = predict_model(model, data=df)
        print("Raw Prediction Output:")
        print(prediction)

        result = prediction.iloc[0]["prediction_label"]

        return jsonify({"prediction": result})

    except Exception as e:
        print("❌ ERROR:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))