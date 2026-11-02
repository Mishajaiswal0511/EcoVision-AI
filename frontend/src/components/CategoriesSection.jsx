import React from 'react';
import {
  Package,
  ShieldAlert,
  Box,
  Wine,
  FileText,
  Trash2,
  CheckCircle,
  XCircle,
  Sparkles,
} from 'lucide-react';
import { WASTE_CLASSES } from '../constants/classes';

const ICON_MAP = {
  Package,
  ShieldAlert,
  Box,
  Wine,
  FileText,
  Trash2,
};

export default function CategoriesSection() {
  return (
    <section id="categories" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 scroll-mt-20">
      <div className="text-center max-w-2xl mx-auto mb-12">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-950/60 text-emerald-300 border border-emerald-500/30 text-xs font-semibold mb-3 backdrop-blur-md">
          <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
          <span>Taxonomy Overview</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          Supported Waste Categories
        </h2>
        <p className="text-sm sm:text-base text-slate-300 mt-2">
          EcoVision AI is trained specifically to distinguish these 7 standard categories.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {WASTE_CLASSES.map((item) => {
          const Icon = ICON_MAP[item.iconName] || Package;

          return (
            <div
              key={item.id}
              className="p-6 rounded-3xl bg-slate-900/85 border border-slate-700/80 shadow-xl shadow-black/40 hover:border-emerald-500/50 backdrop-blur-md transition-all flex flex-col justify-between"
            >
              <div>
                {/* Header */}
                <div className="flex items-center justify-between mb-4">
                  <div
                    className="w-12 h-12 rounded-2xl flex items-center justify-center text-white shadow-md"
                    style={{ backgroundColor: item.color }}
                  >
                    <Icon className="w-6 h-6" />
                  </div>

                  <span
                    className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold border ${
                      item.recyclable
                        ? 'bg-emerald-950/80 text-emerald-300 border-emerald-500/40'
                        : 'bg-rose-950/80 text-rose-300 border-rose-500/40'
                    }`}
                  >
                    {item.recyclable ? (
                      <>
                        <CheckCircle className="w-3 h-3 text-emerald-400" />
                        Recyclable
                      </>
                    ) : (
                      <>
                        <XCircle className="w-3 h-3 text-rose-400" />
                        Non-Recyclable
                      </>
                    )}
                  </span>
                </div>

                {/* Name */}
                <h3 className="text-lg font-bold text-white capitalize mb-1">
                  {item.label}
                </h3>
                <div className="text-xs text-emerald-400/80 font-mono mb-3">
                  Class ID: {item.id}
                </div>

                {/* Examples */}
                <p className="text-xs text-slate-300 leading-relaxed mb-4">
                  <span className="font-semibold text-slate-100">Common items: </span>
                  {item.examples}
                </p>
              </div>

              {/* Bottom Bin Badge */}
              <div className="pt-4 border-t border-slate-800 flex items-center justify-between text-xs">
                <span className="text-slate-400 font-medium">Recommended Bin:</span>
                <span
                  className="px-2.5 py-1 rounded-lg text-white font-bold text-[11px] shadow-xs"
                  style={{ backgroundColor: item.binColorHex }}
                >
                  {item.binName.split(' ')[0]} {item.binName.split(' ')[1]}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

