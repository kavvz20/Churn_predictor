import Navbar from "./components/navbar";
import Hero from "./components/Hero";
import PredictionForm from "./components/PredictionForm";
import MetricsDashboard from "./components/MetricsDashboard";
import Footer from "./components/Footer";
import "./App.css";

function App() {
  return (
    <div className="app">
      <Navbar />
      <Hero />
      <PredictionForm />
      <MetricsDashboard />
      <Footer />
    </div>
  );
}

export default App;