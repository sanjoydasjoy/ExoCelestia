import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';

export default function AboutSection() {
  const { ref, inView } = useInView({ threshold: 0.2, triggerOnce: true });

  const features = [
    {
      icon: '🔬',
      title: 'Machine Learning Analysis',
      description: 'Advanced neural networks trained on NASA\'s Kepler, K2, and TESS mission data to identify exoplanet transit signatures.',
      gradient: 'from-blue-500 to-cyan-500'
    },
    {
      icon: '📡',
      title: 'Real-time Processing',
      description: 'Upload light curve data and receive instant analysis with confidence metrics and detailed explanations.',
      gradient: 'from-purple-500 to-pink-500'
    },
    {
      icon: '🌌',
      title: 'NASA Data Integration',
      description: 'Built on thousands of confirmed exoplanets and false positives from space telescope observations.',
      gradient: 'from-green-500 to-emerald-500'
    },
    {
      icon: '📊',
      title: 'Interactive Visualizations',
      description: 'Explore light curves, transit patterns, and statistical analysis with dynamic charts and graphs.',
      gradient: 'from-orange-500 to-red-500'
    }
  ];

  return (
    <section id="about" className="relative py-20 px-6" ref={ref}>
      {/* Grid Background */}
      <div className="absolute inset-0 grid-bg opacity-30" />
      
      <div className="max-w-7xl mx-auto relative z-10">
        {/* Section Header */}
        <motion.div 
          className="text-center mb-16"
          initial={{ opacity: 0, y: 50 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          <h2 className="text-5xl md:text-6xl font-bold mb-6" style={{ fontFamily: 'Orbitron, monospace' }}>
            <span className="gradient-text">MISSION OVERVIEW</span>
          </h2>
          <p className="text-xl text-gray-400 max-w-4xl mx-auto leading-relaxed">
            Our exoplanet detection system combines cutting-edge machine learning with NASA's space telescope data 
            to identify distant worlds that could potentially harbor life.
          </p>
        </motion.div>

        {/* Statistics */}
        <motion.div 
          className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-16"
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          {[
            { label: 'Exoplanets Discovered', value: '5,000+', icon: '🪐' },
            { label: 'Light Curves Analyzed', value: '200K+', icon: '📈' },
            { label: 'Accuracy Rate', value: '94.2%', icon: '🎯' },
            { label: 'Active Telescopes', value: '3', icon: '🔭' }
          ].map((stat, index) => (
            <motion.div
              key={stat.label}
              className="text-center glass p-6 glow-blue"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={inView ? { opacity: 1, scale: 1 } : {}}
              transition={{ duration: 0.5, delay: 0.3 + index * 0.1 }}
              whileHover={{ scale: 1.05 }}
            >
              <div className="text-3xl mb-2">{stat.icon}</div>
              <div className="text-2xl font-bold gradient-text mb-1">{stat.value}</div>
              <div className="text-sm text-gray-400">{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 gap-8 mb-16">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              className="glass-strong p-8 glow-blue"
              initial={{ opacity: 0, y: 50 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.8, delay: 0.4 + index * 0.1 }}
              whileHover={{ 
                scale: 1.02,
                boxShadow: '0 0 30px rgba(0, 212, 255, 0.3)'
              }}
            >
              <div className="flex items-start space-x-4">
                <motion.div 
                  className={`p-4 rounded-xl bg-gradient-to-r ${feature.gradient} bg-opacity-20`}
                  whileHover={{ rotate: 5 }}
                  transition={{ duration: 0.2 }}
                >
                  <span className="text-3xl">{feature.icon}</span>
                </motion.div>
                <div className="flex-1">
                  <h3 className="text-xl font-semibold mb-3 text-white">
                    {feature.title}
                  </h3>
                  <p className="text-gray-400 leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Technology Stack */}
        <motion.div 
          className="text-center"
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.8 }}
        >
          <h3 className="text-2xl font-semibold mb-8">Powered by Advanced Technology</h3>
          <div className="flex flex-wrap justify-center items-center gap-6">
            {[
              { name: 'TensorFlow', icon: '🧠' },
              { name: 'NASA Exoplanet Archive', icon: '🏛️' },
              { name: 'Kepler Mission', icon: '🛰️' },
              { name: 'Python & FastAPI', icon: '🐍' },
              { name: 'React & Next.js', icon: '⚛️' },
              { name: 'Docker', icon: '🐳' }
            ].map((tech, index) => (
              <motion.div
                key={tech.name}
                className="flex items-center space-x-2 glass p-3 rounded-full"
                whileHover={{ scale: 1.1 }}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={inView ? { opacity: 1, scale: 1 } : {}}
                transition={{ duration: 0.5, delay: 0.9 + index * 0.05 }}
              >
                <span className="text-xl">{tech.icon}</span>
                <span className="text-sm font-medium">{tech.name}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Floating Elements */}
      <motion.div 
        className="absolute top-1/4 left-10 w-6 h-6 bg-blue-400 rounded-full opacity-40"
        animate={{
          y: [0, -30, 0],
          x: [0, 15, 0],
          scale: [1, 1.3, 1],
        }}
        transition={{
          duration: 6,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
      
      <motion.div 
        className="absolute bottom-1/3 right-16 w-4 h-4 bg-purple-400 rounded-full opacity-60"
        animate={{
          y: [0, 25, 0],
          rotate: [0, 180, 360],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 2
        }}
      />
    </section>
  );
}
