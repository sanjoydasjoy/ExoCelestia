import { motion } from 'framer-motion';
import { useState } from 'react';
import { useInView } from 'react-intersection-observer';

interface PredictionResponse {
  prediction: string;
  confidence: number;
  flux?: number[];
}

interface DetectionLabProps {
  onFileUpload: (file: File) => void;
  isLoading: boolean;
  result: PredictionResponse | null;
  error: string | null;
}

export default function DetectionLab({ onFileUpload, isLoading, result, error }: DetectionLabProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const { ref, inView } = useInView({ threshold: 0.2, triggerOnce: true });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      
      if (!file.name.endsWith('.csv') && !file.name.endsWith('.json')) {
        alert('Please upload a CSV or JSON file');
        return;
      }
      
      setSelectedFile(file);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      
      if (!file.name.endsWith('.csv') && !file.name.endsWith('.json')) {
        alert('Please upload a CSV or JSON file');
        return;
      }
      
      setSelectedFile(file);
    }
  };

  const handleAnalyze = () => {
    if (selectedFile) {
      onFileUpload(selectedFile);
    }
  };

  const getPredictionIcon = (prediction: string) => {
    return prediction.toLowerCase().includes('exoplanet') ? '🪐' : '⭐';
  };

  const getPredictionColor = (prediction: string) => {
    return prediction.toLowerCase().includes('exoplanet') 
      ? 'from-green-500 to-emerald-500' 
      : 'from-blue-500 to-indigo-500';
  };

  return (
    <section id="detection-lab" className="relative py-16 lg:py-20" ref={ref}>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        {/* Section Header */}
        <motion.div 
          className="text-center mb-12 lg:mb-16"
          initial={{ opacity: 0, y: 50 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold mb-6 text-center" style={{ fontFamily: 'Orbitron, monospace' }}>
            <span className="gradient-text">DETECTION LAB</span>
          </h2>
          <p className="text-lg sm:text-xl text-gray-400 max-w-4xl mx-auto text-center px-4">
            Upload Kepler/K2/TESS light curve data or pre-computed features to analyze for exoplanet signatures
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8 lg:gap-12 items-start max-w-6xl mx-auto">
          {/* Upload Section */}
          <motion.div 
            className="glass-strong p-6 lg:p-8 glow-blue w-full"
            initial={{ opacity: 0, x: -50 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <h3 className="text-xl lg:text-2xl font-semibold mb-6 flex items-center justify-center lg:justify-start text-center lg:text-left">
              📡 Data Upload Terminal
            </h3>
            
            <motion.div
              className={`
                relative border-2 border-dashed rounded-xl p-6 lg:p-8 text-center transition-all duration-300 w-full
                ${dragActive 
                  ? 'border-cyan-400 bg-cyan-400/10 glow-blue' 
                  : 'border-gray-600 hover:border-gray-500'
                }
              `}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              whileHover={{ scale: 1.02 }}
              transition={{ duration: 0.2 }}
            >
              <input
                type="file"
                accept=".csv,.json"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              
              <motion.div 
                className="mb-4"
                animate={{ y: [0, -5, 0] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              >
                {selectedFile ? (
                  <div className="text-6xl">📄</div>
                ) : (
                  <div className="text-6xl">🛸</div>
                )}
              </motion.div>
              
              {selectedFile ? (
                <div>
                  <p className="text-lg font-semibold text-green-400 mb-2">
                    {selectedFile.name}
                  </p>
                  <p className="text-gray-400">
                    {(selectedFile.size / 1024).toFixed(2)} KB
                  </p>
                </div>
              ) : (
                <div>
                  <p className="text-lg mb-2">
                    Drag & drop your space data here
                  </p>
                  <p className="text-gray-400 text-sm">
                    or click to browse files
                  </p>
                </div>
              )}
            </motion.div>

            {/* Data Format Info */}
            <motion.div 
              className="mt-6 p-4 bg-gray-800/50 rounded-lg"
              initial={{ opacity: 0 }}
              animate={inView ? { opacity: 1 } : {}}
              transition={{ duration: 0.8, delay: 0.4 }}
            >
              <h4 className="font-semibold mb-3 text-cyan-400 text-center lg:text-left">📋 Supported Formats:</h4>
              <ul className="text-sm text-gray-300 space-y-2 text-center lg:text-left">
                <li>• CSV: Light curve flux data (FLUX.1, FLUX.2, ...)</li>
                <li>• JSON: Pre-computed features from space telescopes</li>
                <li>• Data from Kepler, K2, TESS missions</li>
              </ul>
            </motion.div>

            {/* Analyze Button */}
            {selectedFile && (
              <motion.button
                className="w-full mt-6 py-4 bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl font-semibold text-lg glass glow-purple btn-space"
                onClick={handleAnalyze}
                disabled={isLoading}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                {isLoading ? (
                  <div className="flex items-center justify-center">
                    <motion.div 
                      className="w-5 h-5 border-2 border-white border-t-transparent rounded-full mr-3"
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    />
                    Analyzing Cosmic Data...
                  </div>
                ) : (
                  '🔬 Analyze for Exoplanets'
                )}
              </motion.button>
            )}
          </motion.div>

          {/* Results Section */}
          <motion.div 
            className="space-y-6 w-full"
            initial={{ opacity: 0, x: 50 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            {/* Error Display */}
            {error && (
              <motion.div 
                className="p-6 bg-red-500/10 border border-red-500/50 rounded-xl glass"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
              >
                <div className="flex items-center">
                  <span className="text-2xl mr-3">❌</span>
                  <div>
                    <h3 className="font-semibold text-red-400">Analysis Error</h3>
                    <p className="text-red-300">{error}</p>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Results Display */}
            {result && (
              <motion.div 
                className="glass-strong p-6 lg:p-8 glow-green w-full"
                initial={{ opacity: 0, scale: 0.9, y: 20 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                transition={{ duration: 0.5, type: "spring" }}
              >
                <h3 className="text-xl lg:text-2xl font-semibold mb-6 flex items-center justify-center lg:justify-start text-center lg:text-left">
                  🔬 Analysis Results
                </h3>

                <div className="space-y-6">
                  {/* Prediction Result */}
                  <div className={`p-6 rounded-xl bg-gradient-to-r ${getPredictionColor(result.prediction)} bg-opacity-20 border border-current border-opacity-30`}>
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-4xl">{getPredictionIcon(result.prediction)}</span>
                      <motion.div 
                        className="text-right"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: 0.3, type: "spring" }}
                      >
                        <div className="text-2xl font-bold">{(result.confidence * 100).toFixed(1)}%</div>
                        <div className="text-sm opacity-75">Confidence</div>
                      </motion.div>
                    </div>
                    
                    <h4 className="text-xl font-semibold mb-2">
                      {result.prediction}
                    </h4>
                    
                    {/* Confidence Bar */}
                    <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                      <motion.div
                        className="h-full bg-gradient-to-r from-cyan-400 to-blue-500"
                        initial={{ width: 0 }}
                        animate={{ width: `${result.confidence * 100}%` }}
                        transition={{ duration: 1, delay: 0.5 }}
                      />
                    </div>
                  </div>

                  {/* Light Curve Visualization */}
                  {result.flux && result.flux.length > 0 && (
                    <div className="p-6 bg-gray-800/50 rounded-xl">
                      <h4 className="font-semibold mb-4 flex items-center">
                        📈 Light Curve Analysis
                      </h4>
                      <div className="h-40 bg-gray-900/50 rounded-lg flex items-center justify-center">
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 0.7 }}
                          className="text-gray-400"
                        >
                          Interactive light curve visualization coming soon...
                        </motion.div>
                      </div>
                    </div>
                  )}
                </div>
              </motion.div>
            )}

            {/* Status Display */}
            {!result && !error && !isLoading && (
              <motion.div 
                className="glass p-6 lg:p-8 text-center w-full"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5 }}
              >
                <motion.div 
                  className="text-5xl lg:text-6xl mb-4"
                  animate={{ 
                    scale: [1, 1.1, 1],
                    rotate: [0, 5, -5, 0]
                  }}
                  transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                >
                  🌌
                </motion.div>
                <h3 className="text-lg lg:text-xl font-semibold text-gray-400 mb-2">
                  Ready for Analysis
                </h3>
                <p className="text-gray-500 text-center px-4">
                  Upload your space telescope data to begin exoplanet detection
                </p>
              </motion.div>
            )}
          </motion.div>
        </div>
      </div>
    </section>
  );
}
