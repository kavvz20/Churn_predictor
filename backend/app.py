from flask import Flask, request, jsonify
from flask_cors import CORS
from pycaret.classification import load_model, predict_model
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load model once
import os
model_path = os.path.join("backend", "churn_model")
model = load_model(model_path)

@app.route("/")
def home():
    return "Backend is working"


from flask import request, jsonify
import pandas as pd

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Received JSON:", data)

        df = pd.DataFrame([data])
        print("Constructed DataFrame:")
        print(df)

        prediction = predict_model(model, data=df)
        print("Raw Prediction Output:")
        print(prediction)

        result = prediction.iloc[0]["prediction_label"]

        return jsonify({
            "prediction": result
        })

    except Exception as e:
        print("ERROR OCCURRED:", str(e))
        return jsonify({"error": str(e)}), 500


import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))