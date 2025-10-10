import { useState, useEffect } from 'react';
import Head from 'next/head';
import { motion, AnimatePresence } from 'framer-motion';
import SpaceBackground from '../components/SpaceBackground';
import HeroSection from '../components/HeroSection';
import DetectionLab from '../components/DetectionLab';
import AboutSection from '../components/AboutSection';
import AuthModal from '../components/AuthModal';
import { useAuth } from '../contexts/AuthContext';

interface PredictionResponse {
  prediction: string;
  confidence: number;
  flux?: number[];
}

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authModalMode, setAuthModalMode] = useState<'login' | 'signup'>('login');
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  
  const { user, logout } = useAuth();

  const handleFileUpload = async (file: File) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/predict/batch', {
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
              
              <div className="hidden sm:flex items-center space-x-4 lg:space-x-6">
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
                
                {/* Auth Section */}
                <div className="flex items-center space-x-3 ml-4 lg:ml-6 border-l border-gray-700/50 pl-4 lg:pl-6">
                  {user ? (
                    // User Profile Menu
                    <div className="relative">
                      <motion.button
                        className="flex items-center space-x-2 px-3 py-2 rounded-full glass hover:bg-white/5 transition-colors"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => setUserMenuOpen(!userMenuOpen)}
                      >
                        <div className="w-8 h-8 rounded-full bg-gradient-to-r from-blue-600 to-purple-600 flex items-center justify-center text-white font-bold">
                          {user.full_name ? user.full_name[0].toUpperCase() : user.email[0].toUpperCase()}
                        </div>
                        <span className="text-gray-300 font-medium text-sm hidden lg:block">
                          {user.full_name || user.email.split('@')[0]}
                        </span>
                        <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                      </motion.button>
                      
                      {/* Dropdown Menu */}
                      <AnimatePresence>
                        {userMenuOpen && (
                          <motion.div
                            className="absolute right-0 mt-2 w-48 glass-strong rounded-lg border border-gray-700/50 overflow-hidden shadow-xl"
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                          >
                            <div className="p-3 border-b border-gray-700/50">
                              <p className="text-sm text-gray-400">Signed in as</p>
                              <p className="text-sm font-medium text-white truncate">{user.email}</p>
                            </div>
                            <button
                              onClick={() => {
                                logout();
                                setUserMenuOpen(false);
                              }}
                              className="w-full text-left px-4 py-3 text-sm text-red-400 hover:bg-red-500/10 transition-colors"
                            >
                              Sign Out
                            </button>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  ) : (
                    // Login/Signup Buttons
                    <>
                      <motion.button
                        className="px-4 py-2 text-gray-300 hover:text-white transition-colors font-medium text-sm lg:text-base"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => {
                          setAuthModalMode('login');
                          setAuthModalOpen(true);
                        }}
                      >
                        Login
                      </motion.button>
                      <motion.button
                        className="px-4 lg:px-5 py-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full text-white font-semibold text-sm lg:text-base glass glow-blue"
                        whileHover={{ 
                          scale: 1.05,
                          boxShadow: '0 0 20px rgba(0, 212, 255, 0.4)'
                        }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => {
                          setAuthModalMode('signup');
                          setAuthModalOpen(true);
                        }}
                      >
                        Sign Up
                      </motion.button>
                    </>
                  )}
                </div>
              </div>
              
              {/* Mobile Menu Button */}
              <div className="sm:hidden">
                <motion.button
                  className="text-gray-300 hover:text-cyan-400 transition-colors"
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                >
                  <div className="w-6 h-6 flex flex-col justify-center items-center">
                    <motion.span 
                      className="w-4 h-0.5 bg-current mb-1"
                      animate={mobileMenuOpen ? { rotate: 45, y: 4 } : { rotate: 0, y: 0 }}
                    ></motion.span>
                    <motion.span 
                      className="w-4 h-0.5 bg-current mb-1"
                      animate={mobileMenuOpen ? { opacity: 0 } : { opacity: 1 }}
                    ></motion.span>
                    <motion.span 
                      className="w-4 h-0.5 bg-current"
                      animate={mobileMenuOpen ? { rotate: -45, y: -4 } : { rotate: 0, y: 0 }}
                    ></motion.span>
                  </div>
                </motion.button>
              </div>
            </div>
          </div>
          
          {/* Mobile Menu Dropdown */}
          <AnimatePresence>
            {mobileMenuOpen && (
              <motion.div
                className="sm:hidden glass-strong border-t border-gray-700/50"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div className="container mx-auto px-4 py-4 space-y-3">
                  <motion.a
                    href="#detection-lab"
                    className="block py-2 text-gray-300 hover:text-cyan-400 transition-colors font-medium"
                    whileTap={{ scale: 0.95 }}
                    onClick={(e) => {
                      e.preventDefault();
                      setMobileMenuOpen(false);
                      document.getElementById('detection-lab')?.scrollIntoView({ behavior: 'smooth' });
                    }}
                  >
                    🔬 Detection Lab
                  </motion.a>
                  <motion.a
                    href="#about"
                    className="block py-2 text-gray-300 hover:text-cyan-400 transition-colors font-medium"
                    whileTap={{ scale: 0.95 }}
                    onClick={(e) => {
                      e.preventDefault();
                      setMobileMenuOpen(false);
                      document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' });
                    }}
                  >
                    📖 About
                  </motion.a>
                  
                  <div className="pt-3 border-t border-gray-700/50 space-y-3">
                    {user ? (
                      <>
                        <div className="py-2 px-4 glass rounded-lg">
                          <p className="text-xs text-gray-400">Signed in as</p>
                          <p className="text-sm font-medium text-white truncate">{user.email}</p>
                        </div>
                        <motion.button
                          className="w-full py-2.5 px-4 border border-red-500/50 rounded-full text-red-400 hover:bg-red-500/10 transition-colors font-medium"
                          whileTap={{ scale: 0.95 }}
                          onClick={() => {
                            logout();
                            setMobileMenuOpen(false);
                          }}
                        >
                          Sign Out
                        </motion.button>
                      </>
                    ) : (
                      <>
                        <motion.button
                          className="w-full py-2.5 px-4 border border-gray-600 rounded-full text-gray-300 hover:text-white hover:border-cyan-400 transition-colors font-medium"
                          whileTap={{ scale: 0.95 }}
                          onClick={() => {
                            setMobileMenuOpen(false);
                            setAuthModalMode('login');
                            setAuthModalOpen(true);
                          }}
                        >
                          Login
                        </motion.button>
                        <motion.button
                          className="w-full py-2.5 px-4 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full text-white font-semibold glass glow-blue"
                          whileTap={{ scale: 0.95 }}
                          onClick={() => {
                            setMobileMenuOpen(false);
                            setAuthModalMode('signup');
                            setAuthModalOpen(true);
                          }}
                        >
                          Sign Up
                        </motion.button>
                      </>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
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
        
        {/* Auth Modal */}
        <AuthModal 
          isOpen={authModalOpen}
          onClose={() => setAuthModalOpen(false)}
          initialMode={authModalMode}
        />
      </div>
    </>
  );
}

