import { useState } from "react";
import "./PredictionForm.css";

function PredictionForm() {
  const [formData, setFormData] = useState({
    gender: "",
    SeniorCitizen: "",
    Partner: "",
    Dependents: "",
    PhoneService: "",
    MultipleLines: "",
    InternetService: "",
    OnlineSecurity: "",
    OnlineBackup: "",
    DeviceProtection: "",
    TechSupport: "",
    StreamingTV: "",
    StreamingMovies: "",
    Contract: "",
    PaperlessBilling: "",
    PaymentMethod: "",
    tenure: "",
    MonthlyCharges: "",
    TotalCharges: ""
  });

  const [error, setError] = useState("");

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();

    for (let key in formData) {
      if (formData[key] === "") {
        setError("Please fill all fields before predicting.");
        return;
      }
    }

    setError("");
    setLoading(true);
    setResult(null);
    try {
      const response = await fetch("https://churn-wx00.onrender.com/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(formData)
      });

      const data = await response.json();
      setResult(data);
      setLoading(false);

    } catch (error) {
      console.error("Error connecting to backend:", error);
      setLoading(false);
    }

    // API call will go here later
  };

  return (
    <section className="prediction-section" id="predict">
      <div className="prediction-wrapper">
        <form className="prediction-card" onSubmit={handleSubmit}>
          <h2 className="prediction-title">Churn Prediction</h2>

          {/* Section 1 */}
          <h3 className="section-title">Basic Customer Information</h3>
          <div className="form-grid">
            <select name="gender" value={formData.gender} onChange={handleChange}>
              <option value="">Select Gender</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>

            <select name="SeniorCitizen" value={formData.SeniorCitizen} onChange={handleChange}>
              <option value="">Senior Citizen?</option>
              <option value="0">No</option>
              <option value="1">Yes</option>
            </select>

            <select name="Partner" value={formData.Partner} onChange={handleChange}>
              <option value="">Has Partner?</option>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
            </select>

            <select name="Dependents" value={formData.Dependents} onChange={handleChange}>
              <option value="">Has Dependents?</option>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
            </select>
          </div>

          {/* Section 2 */}
          <h3 className="section-title">Service Details</h3>
          <div className="form-grid">
            <select name="PhoneService" value={formData.PhoneService} onChange={handleChange}>
              <option value="">Phone Service?</option>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
            </select>

            <select name="MultipleLines" value={formData.MultipleLines} onChange={handleChange}>
              <option value="">Multiple Lines?</option>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
              <option value="No phone service">No phone service</option>
            </select>

            <select name="InternetService" value={formData.InternetService} onChange={handleChange}>
              <option value="">Internet Service</option>
              <option value="DSL">DSL</option>
              <option value="Fiber optic">Fiber optic</option>
              <option value="No">No</option>
            </select>

            <select name="OnlineSecurity" value={formData.OnlineSecurity} onChange={handleChange}>
              <option value="">Online Security?</option>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
              <option value="No internet service">No internet service</option>
            </select>

            <select name="OnlineBackup" value={formData.OnlineBackup} onChange={handleChange}>
              <option value="">Online Backup?</option>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
              <option value="No internet service">No internet service</option>
            </select>

            <select name="DeviceProtection" value={formData.DeviceProtection} onChange={handleChange}>
              <option value="">Device Protection?</option>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
              <option value="No internet service">No internet service</option>
            </select>

            <select name="TechSupport" value={formData.TechSupport} onChange={handleChange}>
              <option value="">Tech Support?</option>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
              <option value="No internet service">No internet service</option>
            </select>

            <select name="StreamingTV" value={formData.StreamingTV} onChange={handleChange}>
              <option value="">Streaming TV?</option>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
              <option value="No internet service">No internet service</option>
            </select>

            <select name="StreamingMovies" value={formData.StreamingMovies} onChange={handleChange}>
              <option value="">Streaming Movies?</option>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>

          {/* Section 3 */}
          <h3 className="section-title">Subscription Details</h3>
          <div className="form-grid">
            <select name="Contract" value={formData.Contract} onChange={handleChange}>
              <option value="">Contract Type</option>
              <option value="Month-to-month">Month-to-month</option>
              <option value="One year">One year</option>
              <option value="Two year">Two year</option>
            </select>

            <select name="PaperlessBilling" value={formData.PaperlessBilling} onChange={handleChange}>
              <option value="">Paperless Billing?</option>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
            </select>

            <select name="PaymentMethod" value={formData.PaymentMethod} onChange={handleChange}>
              <option value="">Payment Method</option>
              <option value="Electronic check">Electronic check</option>
              <option value="Mailed check">Mailed check</option>
              <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
              <option value="Credit card (automatic)">Credit card (automatic)</option>
            </select>

            <input
              type="number"
              name="tenure"
              placeholder="Tenure (months)"
              value={formData.tenure}
              onChange={handleChange}
            />

            <input
              type="number"
              name="MonthlyCharges"
              placeholder="Monthly Charges"
              value={formData.MonthlyCharges}
              onChange={handleChange}
            />

            <input
              type="number"
              name="TotalCharges"
              placeholder="Total Charges"
              value={formData.TotalCharges}
              onChange={handleChange}
            />
          </div>

          {error && <p className="error-text">{error}</p>}

          <button type="submit" className="predict-button" disabled={loading}>
            {loading ? "Predicting..." : "Predict Churn"}
          </button>
        </form>
        {result && (
  <div className="result-card">
    <h3>Prediction Result</h3>
    <p><strong>Status:</strong> {result.prediction}</p>
  </div>
)}
      </div>
    </section>
  );
}

export default PredictionForm;