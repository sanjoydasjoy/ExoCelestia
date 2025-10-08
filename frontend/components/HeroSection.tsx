import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';

export default function HeroSection() {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth) * 100,
        y: (e.clientY / window.innerHeight) * 100,
      });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Animated Gradient Orbs */}
      <motion.div
        className="absolute w-96 h-96 rounded-full bg-gradient-to-r from-blue-500/20 to-purple-500/20 blur-3xl"
        animate={{
          x: mousePosition.x * 0.1,
          y: mousePosition.y * 0.1,
          scale: [1, 1.2, 1],
        }}
        transition={{
          x: { duration: 0.8, ease: "easeOut" },
          y: { duration: 0.8, ease: "easeOut" },
          scale: { duration: 4, repeat: Infinity, ease: "easeInOut" }
        }}
        style={{
          top: '20%',
          left: '20%',
        }}
      />
      
      <motion.div
        className="absolute w-80 h-80 rounded-full bg-gradient-to-r from-green-500/15 to-cyan-500/15 blur-3xl"
        animate={{
          x: mousePosition.x * -0.05,
          y: mousePosition.y * -0.05,
          scale: [1, 0.8, 1],
        }}
        transition={{
          x: { duration: 1.2, ease: "easeOut" },
          y: { duration: 1.2, ease: "easeOut" },
          scale: { duration: 6, repeat: Infinity, ease: "easeInOut" }
        }}
        style={{
          bottom: '20%',
          right: '20%',
        }}
      />

      {/* Main Content */}
      <div className="text-center z-10 max-w-6xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: "easeOut" }}
          className="mb-8"
        >
          <motion.h1 
            className="text-7xl md:text-9xl font-black mb-6 leading-tight"
            style={{ fontFamily: 'Orbitron, monospace' }}
            animate={{ 
              textShadow: [
                '0 0 20px #00D4FF',
                '0 0 40px #8B5CF6',
                '0 0 20px #10B981',
                '0 0 20px #00D4FF'
              ]
            }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
          >
            <span className="gradient-text">EXOPLANET</span>
          </motion.h1>
          
          <motion.h2 
            className="text-3xl md:text-5xl font-light mb-8 text-gray-300"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.3 }}
          >
            DETECTION SYSTEM
          </motion.h2>
        </motion.div>

        <motion.p 
          className="text-xl md:text-2xl text-gray-400 max-w-3xl mx-auto mb-12 leading-relaxed"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
        >
          Discover distant worlds beyond our solar system using advanced machine learning 
          and NASA's space telescope data. Join the search for potentially habitable exoplanets.
        </motion.p>

        <motion.div 
          className="flex flex-col sm:flex-row gap-6 justify-center"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.9 }}
        >
          <motion.button
            className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full text-lg font-semibold glass glow-blue btn-space"
            whileHover={{ 
              scale: 1.05,
              boxShadow: '0 0 30px rgba(0, 212, 255, 0.5)'
            }}
            whileTap={{ scale: 0.95 }}
            onClick={() => document.getElementById('detection-lab')?.scrollIntoView({ behavior: 'smooth' })}
          >
            🚀 Start Detection
          </motion.button>
          
          <motion.button
            className="px-8 py-4 border border-cyan-500/50 rounded-full text-lg font-semibold glass glow-blue btn-space hover:bg-cyan-500/10"
            whileHover={{ 
              scale: 1.05,
              borderColor: '#00D4FF'
            }}
            whileTap={{ scale: 0.95 }}
            onClick={() => document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' })}
          >
            📊 Learn More
          </motion.button>
        </motion.div>
      </div>

      {/* Floating Elements */}
      <motion.div 
        className="absolute top-1/4 left-10 w-4 h-4 bg-cyan-400 rounded-full opacity-60"
        animate={{
          y: [0, -20, 0],
          scale: [1, 1.2, 1],
          opacity: [0.6, 1, 0.6],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
      
      <motion.div 
        className="absolute bottom-1/3 right-16 w-6 h-6 bg-purple-400 rounded-full opacity-40"
        animate={{
          y: [0, 15, 0],
          x: [0, 10, 0],
          scale: [1, 0.8, 1],
        }}
        transition={{
          duration: 4,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 1
        }}
      />

      {/* Scroll Indicator */}
      <motion.div 
        className="absolute bottom-10 left-1/2 transform -translate-x-1/2"
        animate={{ y: [0, 10, 0] }}
        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
      >
        <div className="w-6 h-10 border-2 border-cyan-400 rounded-full flex justify-center">
          <motion.div 
            className="w-1 h-3 bg-cyan-400 rounded-full mt-2"
            animate={{ y: [0, 16, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          />
        </div>
        <p className="text-cyan-400 text-sm mt-2 font-medium">Scroll to explore</p>
      </motion.div>
    </section>
  );
}
