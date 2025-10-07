import { useState } from 'react';
import Head from 'next/head';
import UploadForm from '../components/UploadForm';
import styles from '../styles/Home.module.css';

interface PredictionResponse {
  prediction: string;
  confidence: number;
  flux?: number[]; // optional for light curve plotting
}

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const handleFileUpload = async (file: File) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data: PredictionResponse = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Unexpected error');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <Head>
        <title>Exoplanet Detector</title>
      </Head>

      <main className={styles.main}>
        <h1 className={styles.title}>Exoplanet Detection</h1>
        <p className={styles.description}>
          Upload a Kepler/K2/TESS light curve CSV or pre-computed feature JSON to
          predict whether an exoplanet is present.
        </p>

        <UploadForm onFileUpload={handleFileUpload} isLoading={isLoading} />

        {error && <p className={styles.error}>Error: {error}</p>}

        {result && (
          <div className={styles.resultCard}>
            <h2>Prediction Result</h2>
            <p>
              <strong>Prediction:</strong> {result.prediction}
            </p>
            <p>
              <strong>Confidence:</strong>{' '}
              {(result.confidence * 100).toFixed(2)}%
            </p>

            {/* Chart placeholder. Replace with Plotly or Chart.js */}
            {result.flux && result.flux.length > 0 && (
              <div className={styles.chartPlaceholder}>
                {/* TODO: Insert Plotly or Chart.js light-curve plot here */}
                <p>Light curve plot will appear here.</p>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

