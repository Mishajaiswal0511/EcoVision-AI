import React from 'react';
import { Leaf, Code2, Server, Eye, ExternalLink } from 'lucide-react';

export default function Footer({ onNavigateEcoQuest }) {
  return (
    <footer id="about" className="bg-slate-900 text-slate-400 py-12 border-t border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 pb-10 border-b border-slate-800">
          {/* Brand Col */}
          <div className="md:col-span-2 space-y-3">
            <div className="flex items-center gap-2.5 text-white">
              <div className="w-8 h-8 rounded-xl bg-emerald-500 flex items-center justify-center text-white">
                <Leaf className="w-4 h-4" />
              </div>
              <span className="font-extrabold text-lg tracking-tight">EcoVision AI</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed max-w-sm">
              An intelligent computer vision system for real-time waste localization, classification,
              and compliant municipal segregation guidance.
            </p>
            <div className="flex items-center gap-2 pt-2">
              <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 text-[11px] font-mono border border-slate-700">
                YOLO11s (CPU)
              </span>
              <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 text-[11px] font-mono border border-slate-700">
                FastAPI 1.0.0
              </span>
              <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 text-[11px] font-mono border border-slate-700">
                React + Vite
              </span>
            </div>
          </div>

          {/* Quick Links & Classes */}
          <div>
            <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider mb-3">
              Quick Navigation
            </h4>
            <ul className="text-xs space-y-2 mb-4">
              <li>
                <button
                  onClick={onNavigateEcoQuest}
                  className="text-emerald-400 hover:text-emerald-300 font-semibold transition-colors cursor-pointer text-left flex items-center gap-1.5"
                >
                  <span>🎮 EcoQuest Challenge</span>
                  <span className="text-[10px] bg-emerald-950 px-1.5 py-0.5 rounded border border-emerald-800">Daily</span>
                </button>
              </li>
              <li><a href="#detector" className="text-slate-300 hover:text-emerald-400 font-medium transition-colors">↑ Waste Detector</a></li>
              <li><a href="#how-it-works" className="text-slate-300 hover:text-emerald-400 font-medium transition-colors">↑ How It Works</a></li>
              <li><a href="#categories" className="text-slate-300 hover:text-emerald-400 font-medium transition-colors">↑ Supported Classes</a></li>
              <li>
                <button
                  onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                  className="text-emerald-400 hover:underline flex items-center gap-1 font-semibold pt-1 cursor-pointer"
                >
                  ↑ Back to Top
                </button>
              </li>
            </ul>

            <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider mb-2">
              Taxonomy Classes (7)
            </h4>
            <p className="text-[11px] text-slate-400">
              Plastic &bull; Metal &bull; Cardboard &bull; Glass &bull; Paper &bull; General Trash &bull; Organic
            </p>
          </div>

          {/* Academic Info */}
          <div>
            <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider mb-3">
              Project Context
            </h4>
            <p className="text-xs text-slate-400 leading-relaxed mb-3">
              College Capstone &amp; Portfolio Project. Built to demonstrate automated waste detection and responsible environmental AI.
            </p>
            <div className="text-[11px] text-emerald-400">
              Workspace: Localhost / Windows
            </div>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="pt-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-[11px] text-slate-500">
          <div>
            &copy; {new Date().getFullYear()} EcoVision AI. Built for educational demonstration and portfolio presentation.
          </div>
          <div className="max-w-md text-right text-slate-500">
            Segregation rules are advisory based on benchmark guidelines. Always verify local municipal solid waste policies.
          </div>
        </div>
      </div>
    </footer>
  );
}

