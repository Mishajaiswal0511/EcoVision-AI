import React from 'react';
import { ArrowDown, Sparkles, ShieldCheck, Cpu, Recycle, CheckCircle2 } from 'lucide-react';

export default function Hero({ health }) {
  return (
    <section className="relative overflow-hidden pt-12 pb-16 lg:pt-16 lg:pb-24">
      {/* Background soft ambient glows */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-emerald-500/15 rounded-full blur-3xl pointer-events-none -z-10" />
      <div className="absolute top-1/3 right-10 w-80 h-80 bg-teal-500/10 rounded-full blur-3xl pointer-events-none -z-10" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        {/* Top Tag */}
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-950/60 border border-emerald-500/30 text-emerald-300 text-xs font-semibold mb-6 shadow-sm backdrop-blur-md">
          <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
          <span>Real-World Waste Detection Engine &bull; YOLO11s</span>
        </div>

        {/* Main Heading */}
        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black text-white tracking-tight max-w-4xl mx-auto leading-[1.15]">
          Give Your Waste a{' '}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-teal-300 to-emerald-500 drop-shadow-[0_0_25px_rgba(16,185,129,0.3)]">
            Smarter Destination.
          </span>
        </h1>

        {/* Subtitle */}
        <p className="mt-5 text-base sm:text-lg text-slate-300 max-w-2xl mx-auto leading-relaxed">
          Powered by a trained YOLO11s computer vision model, EcoVision AI instantly classifies
          waste items, localizes them with precision bounding boxes, and delivers compliant segregation guidance.
        </p>

        {/* Action CTAs */}
        <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
          <a
            href="#detector"
            className="inline-flex items-center gap-2 px-6 py-3.5 rounded-xl text-sm font-bold text-slate-950 bg-emerald-400 hover:bg-emerald-300 shadow-lg shadow-emerald-500/30 transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            Start Detecting
            <ArrowDown className="w-4 h-4 font-bold" />
          </a>
          <a
            href="#categories"
            className="inline-flex items-center gap-2 px-6 py-3.5 rounded-xl text-sm font-semibold text-slate-200 bg-slate-900/80 backdrop-blur-md hover:bg-slate-800 border border-slate-700/80 shadow-sm transition-all hover:border-slate-500"
          >
            Explore Taxonomy (7 Classes)
          </a>
        </div>

        {/* Key Real Architecture Highlights */}
        <div className="mt-12 pt-8 border-t border-slate-800/80 max-w-4xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-4 text-left">
          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 shadow-xs backdrop-blur-md">
            <div className="flex items-center gap-2 text-emerald-400 mb-1">
              <Recycle className="w-4 h-4" />
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Taxonomy</span>
            </div>
            <div className="text-base font-bold text-white">7 Standard Classes</div>
            <div className="text-xs text-slate-400">Plastic, Metal, Organic, etc.</div>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 shadow-xs backdrop-blur-md">
            <div className="flex items-center gap-2 text-teal-400 mb-1">
              <Cpu className="w-4 h-4" />
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Model</span>
            </div>
            <div className="text-base font-bold text-white">YOLO11s (CPU)</div>
            <div className="text-xs text-slate-400">Non-blocking threadpool</div>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 shadow-xs backdrop-blur-md">
            <div className="flex items-center gap-2 text-cyan-400 mb-1">
              <CheckCircle2 className="w-4 h-4" />
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Segregation</span>
            </div>
            <div className="text-base font-bold text-white">Disposal Rules</div>
            <div className="text-xs text-slate-400">Preparation & Bin advice</div>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 shadow-xs backdrop-blur-md">
            <div className="flex items-center gap-2 text-amber-400 mb-1">
              <ShieldCheck className="w-4 h-4" />
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Privacy</span>
            </div>
            <div className="text-base font-bold text-white">In-Memory Only</div>
            <div className="text-xs text-slate-400">Zero permanent disk saving</div>
          </div>
        </div>
      </div>
    </section>
  );
}

