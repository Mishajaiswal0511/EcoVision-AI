import React, { useState, useEffect, useCallback } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Detector from './components/Detector';
import DetectionResults from './components/DetectionResults';
import HowItWorks from './components/HowItWorks';
import CategoriesSection from './components/CategoriesSection';
import Footer from './components/Footer';
import EcoQuestPage from './components/ecoquest/EcoQuestPage';
import AnimatedOrganicBackground from './components/AnimatedOrganicBackground';
import { checkBackendHealth } from './services/api';
import { ArrowUp } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('detector'); // 'detector' | 'ecoquest'
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [health, setHealth] = useState(null);
  const [isRefreshingHealth, setIsRefreshingHealth] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);
  const [originalPreviewUrl, setOriginalPreviewUrl] = useState(null);

  // Health fetcher
  const fetchHealth = useCallback(async () => {
    setIsRefreshingHealth(true);
    const data = await checkBackendHealth();
    setHealth(data);
    setIsRefreshingHealth(false);
  }, []);

  useEffect(() => {
    fetchHealth();
    // Refresh health status every 3 minutes (prevents Render rate-limiting)
    const interval = setInterval(fetchHealth, 180000);
    return () => clearInterval(interval);
  }, [fetchHealth]);

  // Handle scroll position for Quick Floating "Back to Top" Action
  useEffect(() => {
    const handleScroll = () => {
      setShowScrollTop(window.scrollY > 350);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Support direct hash navigation (#ecoquest)
  useEffect(() => {
    const handleHash = () => {
      if (window.location.hash === '#ecoquest') {
        setActiveTab('ecoquest');
      } else if (window.location.hash === '#detector') {
        setActiveTab('detector');
      }
    };
    handleHash();
    window.addEventListener('hashchange', handleHash);
    return () => window.removeEventListener('hashchange', handleHash);
  }, []);

  // Handle navigation change
  const handleNavigate = (tab) => {
    setActiveTab(tab);
    window.location.hash = tab === 'ecoquest' ? '#ecoquest' : '#detector';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Handle completed detection
  const handleDetectionComplete = (result, previewUrl) => {
    setDetectionResult(result);
    setOriginalPreviewUrl(previewUrl);

    // Smooth scroll down to results section
    setTimeout(() => {
      const resultsEl = document.getElementById('results');
      if (resultsEl) {
        resultsEl.scrollIntoView({ behavior: 'smooth' });
      }
    }, 150);
  };

  return (
    <div className="flex flex-col min-h-screen relative overflow-x-hidden">
      {/* Living Animated Organic Mesh Background */}
      <AnimatedOrganicBackground />

      {/* Real-time Navbar */}
      <Navbar
        health={health}
        onRefreshHealth={fetchHealth}
        isRefreshingHealth={isRefreshingHealth}
        activeTab={activeTab}
        onNavigate={handleNavigate}
      />

      {/* Main Content */}
      <main className="flex-1">
        {activeTab === 'ecoquest' ? (
          <EcoQuestPage onBackToDetector={() => handleNavigate('detector')} />
        ) : (
          <>
            {/* Hero Section */}
            <Hero health={health} />

            {/* Waste Detector Section */}
            <Detector
              onDetectionComplete={handleDetectionComplete}
              isBackendReady={health?.status === 'healthy'}
            />

            {/* Results Section */}
            {detectionResult && (
              <DetectionResults
                result={detectionResult}
                originalPreviewUrl={originalPreviewUrl}
                onScanAgain={() => {
                  const detectorEl = document.getElementById('detector');
                  if (detectorEl) {
                    detectorEl.scrollIntoView({ behavior: 'smooth' });
                  }
                }}
              />
            )}

            {/* How It Works Section */}
            <HowItWorks />

            {/* Supported Classes */}
            <CategoriesSection />
          </>
        )}
      </main>

      {/* Footer */}
      <Footer onNavigateEcoQuest={() => handleNavigate('ecoquest')} />

      {/* Floating Quick "Back to Top" Action Button */}
      {showScrollTop && (
        <button
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          title="Scroll back to top"
          className="fixed bottom-6 right-6 z-40 p-3 rounded-2xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold shadow-xl shadow-emerald-500/30 border border-emerald-300 transition-all hover:scale-110 active:scale-95 cursor-pointer flex items-center justify-center animate-fadeIn"
        >
          <ArrowUp className="w-5 h-5 stroke-[2.5]" />
        </button>
      )}
    </div>
  );
}
