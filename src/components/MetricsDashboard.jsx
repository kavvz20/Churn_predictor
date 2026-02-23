import "./MetricsDashboard.css";

function MetricsDashboard() {
  return (
    <section className="metrics-section" id="dashboard">
      <h2 className="metrics-title">Model Performance</h2>

      <div className="metrics-grid">
        <div className="metric-card">
          <h3>Accuracy</h3>
          <p>89%</p>
        </div>

        <div className="metric-card">
          <h3>Precision</h3>
          <p>85%</p>
        </div>

        <div className="metric-card">
          <h3>Recall</h3>
          <p>81%</p>
        </div>

        <div className="metric-card">
          <h3>F1 Score</h3>
          <p>83%</p>
        </div>
      </div>
    </section>
  );
}

export default MetricsDashboard;