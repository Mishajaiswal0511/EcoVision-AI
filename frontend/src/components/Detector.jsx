import React, { useState, useRef, useEffect } from 'react';
import {
  UploadCloud,
  Camera,
  Image as ImageIcon,
  Trash2,
  Sliders,
  AlertCircle,
  Loader2,
  X,
  ScanEye,
  Info,
} from 'lucide-react';
import { detectWasteImage } from '../services/api';

const ACCEPTED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
const MAX_FILE_SIZE_MB = 15;

// Compress image in browser canvas to max 1024px before uploading (saves 98% bandwidth & accelerates inference)
async function compressImageForInference(file) {
  return new Promise((resolve) => {
    if (!file || file.size < 350 * 1024) {
      return resolve(file);
    }
    const img = new window.Image();
    const reader = new FileReader();
    reader.onload = (e) => {
      img.onload = () => {
        const MAX_DIM = 1024;
        let { width, height } = img;
        if (width > MAX_DIM || height > MAX_DIM) {
          if (width > height) {
            height = Math.round((height * MAX_DIM) / width);
            width = MAX_DIM;
          } else {
            width = Math.round((width * MAX_DIM) / height);
            height = MAX_DIM;
          }
        }
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, width, height);
        canvas.toBlob(
          (blob) => {
            if (blob && blob.size < file.size) {
              const compressedFile = new File([blob], file.name.replace(/\.[^/.]+$/, '') + '.jpg', {
                type: 'image/jpeg',
                lastModified: Date.now(),
              });
              resolve(compressedFile);
            } else {
              resolve(file);
            }
          },
          'image/jpeg',
          0.85
        );
      };
      img.onerror = () => resolve(file);
      img.src = e.target.result;
    };
    reader.onerror = () => resolve(file);
    reader.readAsDataURL(file);
  });
}

export default function Detector({ onDetectionComplete, isBackendReady }) {
  // State
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [conf, setConf] = useState(0.15);
  const [agnosticNms, setAgnosticNms] = useState(true);
  const [imgsz, setImgsz] = useState(640);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // Camera state
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const fileInputRef = useRef(null);

  // Stop camera tracks helper
  const stopCameraStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsCameraActive(false);
    setCameraError(null);
  };

  // Clean up preview object URL & camera stream on unmount
  useEffect(() => {
    return () => {
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl);
      }
      stopCameraStream();
    };
  }, [previewUrl]);

  // Start Camera
  const startCamera = async () => {
    setCameraError(null);
    setIsCameraActive(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    } catch (err) {
      setIsCameraActive(false);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setCameraError('Camera access permission was denied. Please allow camera permissions in your browser.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setCameraError('No camera device was detected on your system.');
      } else {
        setCameraError(`Camera error: ${err.message || 'Could not access camera'}`);
      }
    }
  };

  // Capture Photo from Camera
  const capturePhoto = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const capturedFile = new File([blob], `ecovision_capture_${Date.now()}.jpg`, {
          type: 'image/jpeg',
        });
        handleFileSelection(capturedFile);
        stopCameraStream();
      },
      'image/jpeg',
      0.92
    );
  };

  // Validate and set file
  const handleFileSelection = (file) => {
    setErrorMessage(null);

    if (!file) return;

    if (!ACCEPTED_TYPES.includes(file.type)) {
      setErrorMessage(`Unsupported file format (${file.type || 'unknown'}). Please upload a JPEG, PNG, or WEBP image.`);
      return;
    }

    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setErrorMessage(`File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum allowed size is ${MAX_FILE_SIZE_MB} MB.`);
      return;
    }

    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  // Clear current selection
  const clearSelection = () => {
    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setErrorMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Handle Drag & Drop
  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  // Run Waste Detection
  const handleDetect = async () => {
    if (!selectedFile || isLoading) return;

    setIsLoading(true);
    setErrorMessage(null);

    try {
      const fileToSend = await compressImageForInference(selectedFile);
      const result = await detectWasteImage(fileToSend, {
        conf,
        iou: 0.45,
        imgsz,
        agnostic_nms: agnosticNms,
      });

      onDetectionComplete(result, previewUrl);
    } catch (err) {
      setErrorMessage(err.message || 'An unexpected error occurred during waste detection.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <section id="detector" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <div className="bg-white rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-200/80 overflow-hidden">
        {/* Section Header */}
        <div className="px-6 py-6 border-b border-slate-100 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <ScanEye className="w-5 h-5 text-emerald-600" />
              <h2 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
                AI Waste Detector
              </h2>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700 border border-emerald-200">
                Model: V5 (7-Class Fast @ 640px)
              </span>
            </div>
            <p className="text-xs sm:text-sm text-slate-500 mt-1">
              Select or capture a photo to localize waste items and receive compliant disposal instructions.
            </p>
          </div>

          {/* Quick Info Pill */}
          <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-100 text-slate-600 text-xs font-medium self-start md:self-auto">
            <Info className="w-3.5 h-3.5 text-slate-400" />
            <span>Images processed in memory &bull; Never stored</span>
          </div>
        </div>

        <div className="p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Upload / Camera Interface (7 cols) */}
          <div className="lg:col-span-7 flex flex-col">
            {/* Camera View Mode */}
            {isCameraActive ? (
              <div className="relative rounded-2xl overflow-hidden bg-slate-950 aspect-4/3 flex flex-col items-center justify-center border border-slate-800">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                />
                <div className="absolute top-3 right-3">
                  <button
                    onClick={stopCameraStream}
                    className="p-2 rounded-full bg-slate-900/80 text-white hover:bg-slate-800 transition-colors"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
                <div className="absolute bottom-4 flex items-center gap-4">
                  <button
                    onClick={capturePhoto}
                    className="px-5 py-2.5 rounded-full bg-emerald-500 hover:bg-emerald-600 text-white font-semibold text-sm shadow-lg flex items-center gap-2 transition-all"
                  >
                    <Camera className="w-4 h-4" />
                    Snap Photo
                  </button>
                  <button
                    onClick={stopCameraStream}
                    className="px-4 py-2.5 rounded-full bg-white/20 text-white hover:bg-white/30 text-xs font-medium backdrop-blur-xs transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : previewUrl ? (
              /* Selected Image Preview Mode */
              <div className="relative rounded-2xl overflow-hidden bg-slate-100 border border-slate-200 aspect-4/3 flex items-center justify-center group">
                <img
                  src={previewUrl}
                  alt="Selected waste preview"
                  className="w-full h-full object-contain"
                />
                <div className="absolute inset-0 bg-slate-900/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-3">
                  <button
                    onClick={clearSelection}
                    className="px-4 py-2 rounded-xl bg-white text-rose-600 font-semibold text-xs shadow-md hover:bg-rose-50 flex items-center gap-1.5 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                    Remove Image
                  </button>
                </div>
                <button
                  onClick={clearSelection}
                  className="absolute top-3 right-3 p-1.5 rounded-lg bg-white/90 text-slate-600 hover:text-rose-600 shadow-sm transition-colors md:hidden"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ) : (
              /* Drag-and-Drop Zone */
              <div
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
                className={`rounded-2xl border-2 border-dashed p-8 sm:p-12 text-center flex flex-col items-center justify-center aspect-4/3 transition-all ${
                  isDragging
                    ? 'border-emerald-500 bg-emerald-50/60 scale-[0.99]'
                    : 'border-slate-300 hover:border-emerald-400 bg-slate-50/50 hover:bg-emerald-50/20'
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      handleFileSelection(e.target.files[0]);
                    }
                  }}
                />

                <div className="w-14 h-14 rounded-2xl bg-emerald-100 text-emerald-600 flex items-center justify-center mb-4 shadow-sm">
                  <UploadCloud className="w-7 h-7" />
                </div>

                <h3 className="text-base sm:text-lg font-bold text-slate-800">
                  Drag and drop your waste photo
                </h3>
                <p className="text-xs sm:text-sm text-slate-500 mt-1 max-w-sm">
                  JPEG, PNG, or WEBP up to 15 MB. High-resolution desktop or mobile photos supported.
                </p>

                <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="px-4 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-semibold text-xs shadow-sm transition-all hover:shadow"
                  >
                    Browse Files
                  </button>

                  <button
                    type="button"
                    onClick={startCamera}
                    className="px-4 py-2.5 rounded-xl bg-white hover:bg-slate-100 text-slate-700 font-semibold text-xs border border-slate-200 shadow-xs flex items-center gap-1.5 transition-all"
                  >
                    <Camera className="w-3.5 h-3.5 text-slate-500" />
                    Use Camera
                  </button>
                </div>
              </div>
            )}

            {/* Error notifications */}
            {(errorMessage || cameraError) && (
              <div className="mt-4 p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start gap-2.5 animate-fadeIn">
                <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                <div className="flex-1 font-medium leading-relaxed">
                  {errorMessage || cameraError}
                </div>
                <button
                  onClick={() => {
                    setErrorMessage(null);
                    setCameraError(null);
                  }}
                  className="text-rose-400 hover:text-rose-600"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </div>

          {/* Right Column: Controls & Run Action (5 cols) */}
          <div className="lg:col-span-5 flex flex-col justify-between space-y-6">
            <div className="space-y-6">
              {/* Confidence Threshold Slider */}
              <div className="p-4 sm:p-5 rounded-2xl bg-slate-50/80 border border-slate-200/80">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Sliders className="w-4 h-4 text-emerald-600" />
                    <label htmlFor="conf-slider" className="text-xs sm:text-sm font-bold text-slate-800">
                      Detection Confidence Threshold
                    </label>
                  </div>
                  <span className="px-2.5 py-1 rounded-lg bg-emerald-600 text-white font-mono font-bold text-xs shadow-xs">
                    {Math.round(conf * 100)}%
                  </span>
                </div>

                <input
                  id="conf-slider"
                  type="range"
                  min="0.05"
                  max="0.50"
                  step="0.01"
                  value={conf}
                  onChange={(e) => setConf(parseFloat(e.target.value))}
                  className="w-full accent-emerald-600 cursor-pointer h-2 bg-slate-200 rounded-lg"
                />

                <div className="flex justify-between text-[11px] font-medium text-slate-400 mt-1">
                  <span>5% (More detections in piles)</span>
                  <span>50% (High confidence)</span>
                </div>

                <div className="mt-3 p-2.5 rounded-xl bg-white border border-slate-200 text-[11px] text-slate-500 leading-normal">
                  <span className="font-semibold text-slate-700">Trade-off note:</span> Lower thresholds (5%–15%) detect smaller or partial items in cluttered piles. Higher thresholds (30%+) demand high certainty.
                </div>
              </div>

              {/* Class-Agnostic NMS toggle */}
              <div className="p-4 rounded-2xl bg-slate-50/80 border border-slate-200/80 flex items-center justify-between gap-4">
                <div>
                  <div className="text-xs sm:text-sm font-bold text-slate-800">
                    Class-Agnostic NMS &amp; CAS
                  </div>
                  <div className="text-[11px] text-slate-500 mt-0.5">
                    Suppress duplicate overlapping boxes &amp; cross-class ghost proposals.
                  </div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer shrink-0">
                  <input
                    type="checkbox"
                    checked={agnosticNms}
                    onChange={(e) => setAgnosticNms(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-slate-200 peer-focus:outline-hidden rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-600"></div>
                </label>
              </div>

              {/* Inference Resolution Toggle */}
              <div className="p-4 rounded-2xl bg-slate-50/80 border border-slate-200/80 flex items-center justify-between gap-4">
                <div>
                  <div className="text-xs sm:text-sm font-bold text-slate-800">
                    Input Resolution
                  </div>
                  <div className="text-[11px] text-slate-500 mt-0.5">
                    640px runs 10x faster; 800px resolves distant details.
                  </div>
                </div>
                <div className="inline-flex p-1 bg-slate-200 rounded-xl">
                  <button
                    type="button"
                    onClick={() => setImgsz(640)}
                    className={`px-3 py-1 text-xs font-bold rounded-lg transition-all ${
                      imgsz === 640 ? 'bg-white text-emerald-700 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    640px (Fast)
                  </button>
                  <button
                    type="button"
                    onClick={() => setImgsz(800)}
                    className={`px-3 py-1 text-xs font-bold rounded-lg transition-all ${
                      imgsz === 800 ? 'bg-white text-emerald-700 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    800px
                  </button>
                </div>
              </div>

              {/* Supported Target Classes Summary Badge */}
              <div className="text-xs text-slate-500">
                <span className="font-semibold text-slate-700">Trained Categories:</span> Plastic, Metal, Cardboard, Glass, Paper, General Trash, Organic.
              </div>
            </div>

            {/* Run Button */}
            <div>
              <button
                type="button"
                onClick={handleDetect}
                disabled={!selectedFile || isLoading}
                className={`w-full py-4 rounded-2xl font-bold text-sm flex items-center justify-center gap-2 shadow-md transition-all ${
                  !selectedFile || isLoading
                    ? 'bg-slate-200 text-slate-400 cursor-not-allowed shadow-none'
                    : 'bg-emerald-600 hover:bg-emerald-700 active:scale-[0.99] text-white shadow-emerald-600/30 hover:shadow-lg'
                }`}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Running YOLO11s Inference (CPU)...</span>
                  </>
                ) : (
                  <>
                    <ScanEye className="w-5 h-5" />
                    <span>Detect Waste & Segregate</span>
                  </>
                )}
              </button>

              {!selectedFile && (
                <p className="text-center text-[11px] text-slate-400 mt-2">
                  Please upload or capture an image first.
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

