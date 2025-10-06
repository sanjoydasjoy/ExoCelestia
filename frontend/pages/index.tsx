import { useState } from 'react';
import Head from 'next/head';
import UploadForm from '../components/UploadForm';
import styles from '../styles/Home.module.css';

interface PredictionResult {
  prediction: number;
  confidence: number;
  probabilities: {
    no_exoplanet: number;
    exoplanet: number;
  };
}

export default function Home() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePrediction = async (file: File) => {
    setLoading(true);
    setError(null);
    
    try {
      // TODO: Update with your actual backend URL
      const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${backendUrl}/api/predict/batch`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      
      // TODO: Handle batch predictions appropriately
      // For now, showing the first prediction
      if (data.predictions && data.predictions.length > 0) {
        setResult(data.predictions[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <Head>
        <title>Exoplanet Detection</title>
        <meta name="description" content="AI-powered exoplanet detection from light curve data" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={styles.main}>
        <h1 className={styles.title}>
          Exoplanet Detection
        </h1>

        <p className={styles.description}>
          Upload Kepler/K2/TESS light curve data to detect potential exoplanets
        </p>

        <div className={styles.uploadSection}>
          <UploadForm 
            onFileUpload={handlePrediction}
            isLoading={loading}
          />
        </div>

        {error && (
          <div className={styles.error}>
            <p>Error: {error}</p>
          </div>
        )}

        {result && (
          <div className={styles.results}>
            <h2>Prediction Results</h2>
            <div className={styles.resultCard}>
              <div className={styles.predictionLabel}>
                <span className={result.prediction === 1 ? styles.exoplanet : styles.noExoplanet}>
                  {result.prediction === 1 ? '🪐 Exoplanet Detected!' : '⭐ No Exoplanet'}
                </span>
              </div>
              
              <div className={styles.confidence}>
                <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
              </div>

              <div className={styles.probabilities}>
                <h3>Probabilities</h3>
                <div className={styles.probBar}>
                  <div className={styles.probLabel}>
                    <span>No Exoplanet</span>
                    <span>{(result.probabilities.no_exoplanet * 100).toFixed(1)}%</span>
                  </div>
                  <div className={styles.barContainer}>
                    <div 
                      className={styles.bar}
                      style={{ width: `${result.probabilities.no_exoplanet * 100}%` }}
                    />
                  </div>
                </div>
                
                <div className={styles.probBar}>
                  <div className={styles.probLabel}>
                    <span>Exoplanet</span>
                    <span>{(result.probabilities.exoplanet * 100).toFixed(1)}%</span>
                  </div>
                  <div className={styles.barContainer}>
                    <div 
                      className={styles.bar}
                      style={{ 
                        width: `${result.probabilities.exoplanet * 100}%`,
                        backgroundColor: '#4CAF50'
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TODO: Add visualization section for light curve plotting */}
        <div className={styles.visualizationPlaceholder}>
          <h3>Light Curve Visualization</h3>
          <p>TODO: Implement light curve plotting using Chart.js or D3.js</p>
        </div>
      </main>

      <footer className={styles.footer}>
        <p>Powered by NASA Kepler/K2/TESS data</p>
      </footer>
    </div>
  );
}

