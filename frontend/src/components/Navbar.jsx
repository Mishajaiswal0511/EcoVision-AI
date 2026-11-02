import React from 'react';
import { Leaf, Activity, CheckCircle2, AlertTriangle, RefreshCw, Trophy, Sparkles } from 'lucide-react';

export default function Navbar({ health, onRefreshHealth, isRefreshingHealth, activeTab = 'detector', onNavigate }) {
  const isHealthy = health?.status === 'healthy' && health?.model_loaded;
  const isConnecting = !health;

  return (
    <header className="sticky top-0 z-50 backdrop-blur-xl bg-slate-950/75 border-b border-emerald-500/20 transition-all shadow-lg shadow-black/20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div
          onClick={() => onNavigate && onNavigate('detector')}
          className="flex items-center gap-3 cursor-pointer"
        >
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center text-slate-950 shadow-md shadow-emerald-500/30">
            <Leaf className="w-5 h-5 font-black" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-lg text-white tracking-tight">EcoVision</span>
              <span className="text-xs px-2 py-0.5 rounded-full font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-400/30 font-mono">
                AI
              </span>
            </div>
            <p className="text-[11px] text-emerald-100/60 font-medium hidden sm:block">
              YOLO11s Waste Segregation Engine
            </p>
          </div>
        </div>

        {/* Center Nav Links */}
        <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-300">
          <button
            onClick={() => onNavigate && onNavigate('detector')}
            className={`transition-colors cursor-pointer ${
              activeTab === 'detector'
                ? 'text-emerald-400 font-bold border-b-2 border-emerald-400 pb-0.5'
                : 'hover:text-emerald-400'
            }`}
          >
            Detector
          </button>

          {/* EcoQuest Nav Pill */}
          <button
            onClick={() => onNavigate && onNavigate('ecoquest')}
            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold transition-all cursor-pointer ${
              activeTab === 'ecoquest'
                ? 'bg-emerald-500 text-slate-950 shadow-md shadow-emerald-500/40 font-extrabold'
                : 'bg-emerald-950/60 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-900/50'
            }`}
          >
            <Trophy className="w-3.5 h-3.5 text-amber-300" />
            <span>EcoQuest Challenge</span>
            <span className="px-1.5 py-0.2 rounded-full text-[9px] bg-amber-400 text-slate-950 font-black">
              NEW
            </span>
          </button>

          <a href="#how-it-works" onClick={() => onNavigate && onNavigate('detector')} className="hover:text-emerald-400 transition-colors">
            How It Works
          </a>
          <a href="#categories" onClick={() => onNavigate && onNavigate('detector')} className="hover:text-emerald-400 transition-colors">
            Supported Classes
          </a>
          <a href="#about" onClick={() => onNavigate && onNavigate('detector')} className="hover:text-emerald-400 transition-colors">
            About
          </a>
        </nav>

        {/* Real Backend Status Indicator */}
        <div className="flex items-center gap-3">
          <div
            className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
              isConnecting
                ? 'bg-slate-900 text-slate-400 border-slate-700'
                : isHealthy
                ? 'bg-emerald-950/60 text-emerald-300 border-emerald-500/30'
                : 'bg-amber-950/60 text-amber-300 border-amber-500/30'
            }`}
            title={
              isHealthy
                ? `Backend Online — Model: ${health.model_architecture || 'YOLO11s'} (${health.device || 'CPU'})`
                : health?.error || 'Cannot connect to backend'
            }
          >
            <span className="relative flex h-2 w-2">
              <span
                className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                  isHealthy ? 'bg-emerald-400' : 'bg-amber-400'
                }`}
              />
              <span
                className={`relative inline-flex rounded-full h-2 w-2 ${
                  isHealthy ? 'bg-emerald-400' : 'bg-amber-400'
                }`}
              />
            </span>
            <span className="hidden sm:inline font-semibold">
              {isConnecting
                ? 'Checking backend...'
                : isHealthy
                ? 'Model Ready (CPU)'
                : 'Backend Degraded'}
            </span>
          </div>

          <button
            onClick={onRefreshHealth}
            disabled={isRefreshingHealth}
            title="Refresh backend status"
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${isRefreshingHealth ? 'animate-spin' : ''}`} />
          </button>

          {activeTab === 'detector' ? (
            <button
              onClick={() => onNavigate && onNavigate('ecoquest')}
              className="inline-flex items-center justify-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold text-emerald-300 bg-emerald-950/70 hover:bg-emerald-900 border border-emerald-500/30 transition-all cursor-pointer"
            >
              <Trophy className="w-3.5 h-3.5 text-amber-400" />
              <span>EcoQuest</span>
            </button>
          ) : (
            <button
              onClick={() => onNavigate && onNavigate('detector')}
              className="inline-flex items-center justify-center px-4 py-2 rounded-xl text-xs font-semibold text-slate-950 bg-emerald-400 hover:bg-emerald-300 shadow-sm shadow-emerald-400/30 font-bold transition-all cursor-pointer"
            >
              Start Detecting
            </button>
          )}
        </div>
      </div>
    </header>
  );
}

