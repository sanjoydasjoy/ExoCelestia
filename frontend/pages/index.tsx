import { useState, useEffect } from 'react';
import Head from 'next/head';
import { motion, AnimatePresence } from 'framer-motion';
import SpaceBackground from '../components/SpaceBackground';
import HeroSection from '../components/HeroSection';
import DetectionLab from '../components/DetectionLab';
import AboutSection from '../components/AboutSection';

interface PredictionResponse {
  prediction: string;
  confidence: number;
  flux?: number[];
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
    <>
      <Head>
        <title>Exoplanet Detection System | NASA Space Apps</title>
        <meta name="description" content="Advanced machine learning system for detecting exoplanets using NASA space telescope data" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="relative">
        {/* Animated Space Background */}
        <SpaceBackground />
        
        {/* Navigation */}
        <motion.nav 
          className="fixed top-0 left-0 right-0 z-50 glass-strong"
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
            <div className="flex items-center justify-between py-4">
              <motion.div 
                className="flex items-center space-x-3"
                whileHover={{ scale: 1.05 }}
              >
                <div className="text-xl lg:text-2xl">🌍</div>
                <div className="text-lg lg:text-xl font-bold" style={{ fontFamily: 'Orbitron, monospace' }}>
                  <span className="gradient-text">EXOPLANET HUNTER</span>
                </div>
              </motion.div>
              
              <div className="hidden sm:flex space-x-4 lg:space-x-6">
                <motion.a 
                  href="#detection-lab"
                  className="text-gray-300 hover:text-cyan-400 transition-colors font-medium text-sm lg:text-base"
                  whileHover={{ scale: 1.05 }}
                  onClick={(e) => {
                    e.preventDefault();
                    document.getElementById('detection-lab')?.scrollIntoView({ behavior: 'smooth' });
                  }}
                >
                  Detection Lab
                </motion.a>
                <motion.a 
                  href="#about"
                  className="text-gray-300 hover:text-cyan-400 transition-colors font-medium text-sm lg:text-base"
                  whileHover={{ scale: 1.05 }}
                  onClick={(e) => {
                    e.preventDefault();
                    document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' });
                  }}
                >
                  About
                </motion.a>
              </div>
              
              {/* Mobile Menu Button */}
              <div className="sm:hidden">
                <motion.button
                  className="text-gray-300 hover:text-cyan-400 transition-colors"
                  whileTap={{ scale: 0.95 }}
                >
                  <div className="w-6 h-6 flex flex-col justify-center items-center">
                    <span className="w-4 h-0.5 bg-current mb-1"></span>
                    <span className="w-4 h-0.5 bg-current mb-1"></span>
                    <span className="w-4 h-0.5 bg-current"></span>
                  </div>
                </motion.button>
              </div>
            </div>
          </div>
        </motion.nav>

        {/* Main Content */}
        <main className="relative z-10">
          {/* Hero Section */}
          <HeroSection />

          {/* Detection Lab Section */}
          <DetectionLab 
            onFileUpload={handleFileUpload}
            isLoading={isLoading}
            result={result}
            error={error}
          />

          {/* About Section */}
          <AboutSection />
        </main>

        {/* Footer */}
        <motion.footer 
          className="relative z-10 py-8 lg:py-12 border-t border-gray-800/50 glass"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 1 }}
        >
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl text-center">
            <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6 mb-6">
              <motion.div 
                className="text-2xl lg:text-3xl"
                animate={{ rotate: [0, 360] }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              >
                🛰️
              </motion.div>
              <div className="text-center sm:text-left">
                <h3 className="text-lg lg:text-xl font-semibold gradient-text">NASA Space Apps Challenge 2024</h3>
                <p className="text-gray-400 text-sm lg:text-base">Exoplanet Detection System</p>
              </div>
            </div>
            
            <p className="text-gray-500 mb-4 text-sm lg:text-base px-4">
              Built with ❤️ for space exploration and the search for life beyond Earth
            </p>
            
            <div className="flex flex-col sm:flex-row justify-center items-center space-y-2 sm:space-y-0 sm:space-x-4 lg:space-x-8 text-xs lg:text-sm text-gray-600">
              <span>NASA Exoplanet Archive</span>
              <span className="hidden sm:inline">•</span>
              <span>Kepler Mission Data</span>
              <span className="hidden sm:inline">•</span>
              <span>TESS Observatory</span>
            </div>
          </div>
        </motion.footer>
      </div>
    </>
  );
}

