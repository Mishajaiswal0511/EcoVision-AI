import React, { useState } from 'react';
import {
  Sparkles,
  HelpCircle,
  Sliders,
  Sun,
  Camera,
  Layers,
  CheckCircle2,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Info,
} from 'lucide-react';

export default function SmartDetectionAssistant({
  diagnostic,
  imageQuality,
  numberOfDetections,
  onScanAgain,
  onApplySuggestedConf,
}) {
  const [showEvidence, setShowEvidence] = useState(false);

  if (!diagnostic) return null;

  const isSuccess = diagnostic.status === 'success' || numberOfDetections > 0;
  const reason = diagnostic.primary_reason || 'UNKNOWN';

  // Dynamic status badges and theme colors
  let theme = {
    bg: 'bg-slate-50',
    border: 'border-slate-200',
    badgeBg: 'bg-slate-100 text-slate-800 border-slate-200',
    iconBg: 'bg-slate-100 text-slate-600',
    Icon: HelpCircle,
    statusLabel: 'No Waste Detected',
  };

  if (isSuccess) {
    theme = {
      bg: 'bg-emerald-50/60',
      border: 'border-emerald-200',
      badgeBg: 'bg-emerald-100 text-emerald-800 border-emerald-200',
      iconBg: 'bg-emerald-500 text-white',
      Icon: CheckCircle2,
      statusLabel: `${numberOfDetections} Item${numberOfDetections > 1 ? 's' : ''} Localized`,
    };
  } else if (reason === 'BELOW_CONFIDENCE_THRESHOLD') {
    theme = {
      bg: 'bg-amber-50/60',
      border: 'border-amber-200',
      badgeBg: 'bg-amber-100 text-amber-800 border-amber-200',
      iconBg: 'bg-amber-500 text-white',
      Icon: Sliders,
      statusLabel: 'Low Confidence Rejection',
    };
  } else if (reason === 'POOR_LIGHTING_DARK' || reason === 'POOR_LIGHTING_OVEREXPOSED') {
    theme = {
      bg: 'bg-indigo-50/60',
      border: 'border-indigo-200',
      badgeBg: 'bg-indigo-100 text-indigo-800 border-indigo-200',
      iconBg: 'bg-indigo-500 text-white',
      Icon: Sun,
      statusLabel: 'Lighting Issue Detected',
    };
  } else if (reason === 'MOTION_BLUR') {
    theme = {
      bg: 'bg-rose-50/60',
      border: 'border-rose-200',
      badgeBg: 'bg-rose-100 text-rose-800 border-rose-200',
      iconBg: 'bg-rose-500 text-white',
      Icon: Camera,
      statusLabel: 'Motion Blur Detected',
    };
  } else if (reason === 'CONTAINMENT_SUPPRESSION') {
    theme = {
      bg: 'bg-teal-50/60',
      border: 'border-teal-200',
      badgeBg: 'bg-teal-100 text-teal-800 border-teal-200',
      iconBg: 'bg-teal-500 text-white',
      Icon: Layers,
      statusLabel: 'Redundant Overlap Filtered',
    };
  }

  const { Icon } = theme;
  const suggestedConf = diagnostic.evidence?.top_candidate_confidence
    ? Math.max(0.05, Math.round(diagnostic.evidence.top_candidate_confidence * 100) / 100)
    : null;

  return (
    <div className={`rounded-3xl border ${theme.border} ${theme.bg} p-6 sm:p-8 shadow-sm transition-all animate-fadeIn mb-8`}>
      {/* Top Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-slate-200/60">
        <div className="flex items-center gap-3.5">
          <div className={`w-11 h-11 rounded-2xl flex items-center justify-center shadow-sm ${theme.iconBg}`}>
            <Icon className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-extrabold text-base sm:text-lg text-slate-900 tracking-tight">
                Smart Detection Assistant
              </h3>
              <span className={`text-[11px] font-bold px-2.5 py-0.5 rounded-full border ${theme.badgeBg}`}>
                {theme.statusLabel}
              </span>
            </div>
            <p className="text-xs text-slate-500 font-medium mt-0.5">
              Inference Diagnostics &amp; Physical Quality Insights
            </p>
          </div>
        </div>

        {/* Action Button: Scan Again */}
        {onScanAgain && (
          <button
            onClick={onScanAgain}
            className="inline-flex items-center justify-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 shadow-sm transition-all cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5 text-slate-500" />
            <span>Scan Again</span>
          </button>
        )}
      </div>

      {/* Main Diagnostic Body */}
      <div className="pt-5 space-y-6">
        {/* Section 1: Why did this happen? */}
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs font-bold text-slate-700 uppercase tracking-wider">
            <Info className="w-4 h-4 text-emerald-600" />
            <span>Why did this happen?</span>
          </div>
          <div className="p-4 rounded-2xl bg-white/90 border border-slate-200/80 shadow-xs">
            <h4 className="text-sm font-bold text-slate-900 mb-1">
              {diagnostic.title || "Scan Diagnostic Summary"}
            </h4>
            <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
              {diagnostic.explanation}
            </p>

            {/* Quick action button if confidence was the culprit */}
            {suggestedConf && onApplySuggestedConf && (
              <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between gap-3">
                <span className="text-xs text-amber-800 font-medium">
                  Candidate found with {Math.round(suggestedConf * 100)}% score.
                </span>
                <button
                  onClick={() => onApplySuggestedConf(suggestedConf)}
                  className="px-3 py-1.5 rounded-lg text-xs font-bold bg-amber-500 hover:bg-amber-600 text-white shadow-xs transition-all cursor-pointer"
                >
                  Apply {Math.round(suggestedConf * 100)}% Threshold
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Section 2: What can you try? */}
        {diagnostic.suggestions && diagnostic.suggestions.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs font-bold text-slate-700 uppercase tracking-wider">
              <Sparkles className="w-4 h-4 text-amber-500" />
              <span>What can you try?</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {diagnostic.suggestions.map((suggestion, idx) => (
                <div
                  key={idx}
                  className="p-3.5 rounded-xl bg-white/80 border border-slate-200/70 flex items-start gap-3 shadow-2xs"
                >
                  <span className="w-5 h-5 rounded-full bg-emerald-100 text-emerald-800 text-[11px] font-bold flex items-center justify-center shrink-0 mt-0.5">
                    {idx + 1}
                  </span>
                  <p className="text-xs text-slate-700 leading-snug">
                    {suggestion}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Section 3: Measurable Optical Evidence Drawer */}
        {(diagnostic.evidence || imageQuality) && (
          <div className="pt-2">
            <button
              onClick={() => setShowEvidence(!showEvidence)}
              className="text-xs font-semibold text-slate-500 hover:text-slate-800 flex items-center gap-1.5 transition-colors cursor-pointer"
            >
              <span>{showEvidence ? 'Hide Optical Evidence' : 'Inspect Measured Optical Evidence'}</span>
              {showEvidence ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>

            {showEvidence && (
              <div className="mt-3 p-4 rounded-2xl bg-white border border-slate-200 text-xs space-y-2 animate-fadeIn font-mono">
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-slate-600">
                  {imageQuality?.brightness !== undefined && (
                    <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-100">
                      <span className="block text-[10px] text-slate-400 uppercase">Scene Brightness</span>
                      <strong className="text-slate-900">{imageQuality.brightness} / 255</strong>
                    </div>
                  )}
                  {imageQuality?.sharpness_score !== undefined && (
                    <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-100">
                      <span className="block text-[10px] text-slate-400 uppercase">Sharpness Score</span>
                      <strong className="text-slate-900">{imageQuality.sharpness_score}</strong>
                    </div>
                  )}
                  {diagnostic.evidence?.current_threshold !== undefined && (
                    <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-100">
                      <span className="block text-[10px] text-slate-400 uppercase">Active Cutoff</span>
                      <strong className="text-slate-900">{diagnostic.evidence.current_threshold * 100}%</strong>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
