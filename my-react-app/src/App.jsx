import { useState } from "react";
import "./App.css";

function App() {
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const evaluatePrompt = async () => {
    if (!prompt.trim()) {
      setError("Silakan masukkan prompt terlebih dahulu.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            prompt: prompt,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Gagal mengevaluasi prompt.");
      }

      setResult(data);

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getQualityLabel = (score) => {
    if (score >= 0.7) return "Kualitas Tinggi";
    if (score >= 0.4) return "Kualitas Sedang";
    return "Kualitas Rendah";
  };

  return (
    <main className="container">
      <h1 className="main-title">Prompt Quality Evaluator</h1>
      <div className="input-section">
        <label>Masukkan Prompt</label>

        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Tuliskan prompt yang ingin dievaluasi..."
          rows={8}
        />

        <button
          onClick={evaluatePrompt}
          disabled={loading || !prompt.trim()}
        >
          {loading ? "Mengevaluasi..." : "Evaluate Prompt"}
        </button>
      </div>

      {error && (
        <p className="error">{error}</p>
      )}

      {result && (
        <div className="result-section">
          <h2>Hasil Evaluasi</h2>
          <h1 className="score">
            {result.score_percentage.toFixed(0)} %
          </h1>
          <p className="quality-label">
            {getQualityLabel(result.score)}
          </p>
        </div>
      )}
    </main>
  );
}

export default App;
