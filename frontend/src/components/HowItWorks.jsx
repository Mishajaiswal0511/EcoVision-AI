import React from 'react';
import { UploadCloud, Cpu, Recycle, ArrowRight } from 'lucide-react';

export default function HowItWorks() {
  const steps = [
    {
      step: '01',
      title: 'Upload or Capture',
      description: 'Upload an image from your computer or snap a photo directly using your webcam or phone camera.',
      icon: UploadCloud,
      color: 'emerald',
    },
    {
      step: '02',
      title: 'AI Category Detection',
      description: 'Our YOLO11s model evaluates pixel features to locate items and categorize them across six target classes.',
      icon: Cpu,
      color: 'teal',
    },
    {
      step: '03',
      title: 'Segregation & Disposal',
      description: 'Receive recommended disposal bins, recyclability verification, and step-by-step cleaning and prep guidance.',
      icon: Recycle,
      color: 'blue',
    },
  ];

  return (
    <section id="how-it-works" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 scroll-mt-20">
      <div className="text-center max-w-2xl mx-auto mb-12">
        <div className="text-xs font-bold text-emerald-400 uppercase tracking-wider mb-2">
          Intelligent Workflow
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          How EcoVision AI Works
        </h2>
        <p className="text-sm sm:text-base text-slate-300 mt-2">
          Three simple steps to transform raw waste images into actionable recycling insights.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {steps.map((s, idx) => {
          const Icon = s.icon;
          return (
            <div
              key={idx}
              className="relative p-6 sm:p-8 rounded-3xl bg-slate-900/85 border border-slate-700/80 shadow-xl shadow-black/40 hover:border-emerald-500/50 backdrop-blur-md transition-all group"
            >
              <div className="flex items-center justify-between mb-6">
                <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center justify-center font-bold shadow-xs group-hover:scale-105 transition-transform">
                  <Icon className="w-6 h-6" />
                </div>
                <span className="text-3xl font-black text-slate-600 group-hover:text-emerald-400 transition-colors font-mono">
                  {s.step}
                </span>
              </div>

              <h3 className="text-lg font-bold text-white mb-2">
                {s.title}
              </h3>
              <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
                {s.description}
              </p>
            </div>
          );
        })}
      </div>
    </section>
  );
}

