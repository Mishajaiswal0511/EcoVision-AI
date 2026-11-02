import React, { useState } from 'react';
import {
  Clock,
  Layers,
  Sparkles,
  AlertTriangle,
  CheckCircle,
  HelpCircle,
  Trash2,
  Box,
  Eye,
  FileText,
  ShieldAlert,
  Maximize2,
  Info,
  Check,
  ChevronRight,
} from 'lucide-react';
import { CLASS_BY_NAME } from '../constants/classes';

export default function DetectionResults({ result, originalPreviewUrl }) {
  const [activeTab, setActiveTab] = useState('annotated'); // 'annotated' | 'original'

  if (!result) return null;

  const {
    image_width,
    image_height,
    inference_time_ms,
    number_of_detections,
    detections = [],
    annotated_image,
  } = result;

  const hasDetections = number_of_detections > 0;

  // Group detected classes count
  const categoryCounts = detections.reduce((acc, det) => {
    const name = det.class_name.toLowerCase();
    acc[name] = (acc[name] || 0) + 1;
    return acc;
  }, {});

  return (
    <section id="results" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 animate-fadeIn">
      {/* Metrics Banner */}
      <div className="bg-white rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-200/80 p-6 sm:p-8 mb-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-100">
          <div>
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
              <h2 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
                Detection Analysis Results
              </h2>
            </div>
            <p className="text-xs sm:text-sm text-slate-500 mt-1">
              YOLO11s localized {number_of_detections} waste {number_of_detections === 1 ? 'item' : 'items'} in {inference_time_ms} ms.
            </p>
          </div>

          {/* Quick Metrics Bar */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="px-3.5 py-1.5 rounded-xl bg-slate-100 text-slate-700 text-xs font-semibold flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-slate-500" />
              <span>{inference_time_ms} ms</span>
            </div>
            <div className="px-3.5 py-1.5 rounded-xl bg-slate-100 text-slate-700 text-xs font-semibold flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-slate-500" />
              <span>{image_width} &times; {image_height} px</span>
            </div>
            <div className="px-3.5 py-1.5 rounded-xl bg-emerald-100 text-emerald-800 text-xs font-bold flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
              <span>{number_of_detections} Detected</span>
            </div>
          </div>
        </div>

        {/* Empty State */}
        {!hasDetections ? (
          <div className="py-12 text-center max-w-md mx-auto">
            <div className="w-16 h-16 rounded-2xl bg-amber-100 text-amber-600 flex items-center justify-center mx-auto mb-4">
              <HelpCircle className="w-8 h-8" />
            </div>
            <h3 className="text-lg font-bold text-slate-800">
              No waste items detected
            </h3>
            <p className="text-sm text-slate-500 mt-2 leading-relaxed">
              No waste items detected. Try another image with clearer visibility, closer framing, or lower the confidence threshold.
            </p>
            {annotated_image && (
              <div className="mt-6 rounded-2xl overflow-hidden border border-slate-200 aspect-video max-h-72 mx-auto">
                <img
                  src={annotated_image}
                  alt="Processed preview"
                  className="w-full h-full object-contain bg-slate-50"
                />
              </div>
            )}
          </div>
        ) : (
          /* Results Grid: Visual Comparison (Left) & Breakdowns (Right) */
          <div className="mt-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Visual Prominence Preview (7 cols) */}
            <div className="lg:col-span-7 flex flex-col">
              {/* Tab Selector */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2 p-1 bg-slate-100 rounded-xl">
                  <button
                    onClick={() => setActiveTab('annotated')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      activeTab === 'annotated'
                        ? 'bg-white text-emerald-700 shadow-xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Annotated Prediction
                  </button>
                  <button
                    onClick={() => setActiveTab('original')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      activeTab === 'original'
                        ? 'bg-white text-emerald-700 shadow-xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Original Upload
                  </button>
                </div>

                <div className="text-xs text-slate-400 font-medium">
                  {activeTab === 'annotated' ? 'Color-coded bounding boxes' : 'Unprocessed input'}
                </div>
              </div>

              {/* Main Image Container */}
              <div className="relative rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 aspect-4/3 flex items-center justify-center shadow-inner">
                <img
                  src={activeTab === 'annotated' ? annotated_image : originalPreviewUrl}
                  alt="Detection output"
                  className="w-full h-full object-contain"
                />
              </div>

              {/* Category Count Chips */}
              <div className="mt-4 flex flex-wrap items-center gap-2">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider mr-1">
                  Detected Summary:
                </span>
                {Object.entries(categoryCounts).map(([catName, count]) => {
                  const meta = CLASS_BY_NAME[catName] || {};
                  return (
                    <span
                      key={catName}
                      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${
                        meta.badgeClass || 'bg-slate-100 text-slate-700 border-slate-200'
                      }`}
                    >
                      <span
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: meta.color || '#64748B' }}
                      />
                      <span className="capitalize">{catName}</span>
                      <span className="opacity-75 font-mono">(&times;{count})</span>
                    </span>
                  );
                })}
              </div>
            </div>

            {/* Individual Detections Breakdown (5 cols) */}
            <div className="lg:col-span-5 flex flex-col space-y-4 max-h-[600px] overflow-y-auto pr-1">
              <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
                <FileText className="w-4 h-4 text-emerald-600" />
                Detected Items &amp; Segregation ({detections.length})
              </h3>

              {detections.map((det, index) => {
                const confPercent = Math.round(det.confidence * 100);
                const className = det.class_name.toLowerCase();
                const meta = CLASS_BY_NAME[className] || {};
                const disposal = det.disposal || {};
                const isModerate = confPercent < 50;

                return (
                  <div
                    key={index}
                    className="p-4 sm:p-5 rounded-2xl bg-slate-50/80 border border-slate-200/90 shadow-xs hover:border-slate-300 transition-all"
                  >
                    {/* Header: Class & Confidence Pill */}
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2.5">
                        <span
                          className="w-3.5 h-3.5 rounded-lg shrink-0 shadow-xs"
                          style={{ backgroundColor: meta.color || '#64748B' }}
                        />
                        <span className="text-base font-extrabold text-slate-900 capitalize">
                          {det.class_name}
                        </span>
                        <span className="text-xs text-slate-400 font-mono">
                          #{index + 1}
                        </span>
                      </div>

                      <div className="flex items-center gap-2">
                        {isModerate && (
                          <span
                            title="Moderate confidence level"
                            className="p-1 rounded-md bg-amber-100 text-amber-700 text-xs"
                          >
                            <AlertTriangle className="w-3.5 h-3.5" />
                          </span>
                        )}
                        <span
                          className="px-2.5 py-1 rounded-lg text-xs font-bold text-white shadow-xs"
                          style={{ backgroundColor: meta.color || '#64748B' }}
                        >
                          {confPercent}%
                        </span>
                      </div>
                    </div>

                    {/* Bin Recommendation */}
                    {disposal.bin_label && (
                      <div className="mb-3 p-3 rounded-xl bg-white border border-slate-200 flex items-center justify-between">
                        <div>
                          <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                            Recommended Bin
                          </div>
                          <div className="text-xs sm:text-sm font-bold text-slate-800">
                            {disposal.bin_label}
                          </div>
                          {disposal.bin_color_india && (
                            <div className="text-[11px] text-slate-500 mt-0.5">
                              Bin Color: <span className="font-semibold text-slate-700">{disposal.bin_color_india}</span>
                            </div>
                          )}
                        </div>
                        <div
                          className="px-3 py-1.5 rounded-xl text-xs font-bold text-white shadow-xs"
                          style={{ backgroundColor: meta.binColorHex || '#4B5563' }}
                        >
                          {disposal.recyclable ? 'Recyclable' : 'General'}
                        </div>
                      </div>
                    )}

                    {/* Preparation Steps */}
                    {disposal.preparation_steps && disposal.preparation_steps.length > 0 && (
                      <div className="space-y-1.5 mb-3">
                        <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                          Preparation Steps:
                        </div>
                        <ul className="text-xs text-slate-600 space-y-1 pl-1">
                          {disposal.preparation_steps.map((step, sIdx) => (
                            <li key={sIdx} className="flex items-start gap-1.5">
                              <Check className="w-3.5 h-3.5 text-emerald-600 shrink-0 mt-0.5" />
                              <span>{step}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Safety Warning */}
                    {disposal.safety_warning && (
                      <div className="p-2.5 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-start gap-2">
                        <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                        <span className="leading-snug">{disposal.safety_warning}</span>
                      </div>
                    )}

                    {/* Coordinates chip */}
                    <div className="mt-3 pt-2 border-t border-slate-200/60 flex items-center justify-between text-[11px] text-slate-400">
                      <span>BBox (px): [{det.bounding_box.x1}, {det.bounding_box.y1}, {det.bounding_box.x2}, {det.bounding_box.y2}]</span>
                      <span>{meta.material || 'Material specified'}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

