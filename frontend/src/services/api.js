/**
 * EcoVision AI — Backend API Client
 * Interfaces with FastAPI endpoints at /api/health, /api/detect, and /api/disposal/{class_name}
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

/**
 * Checks the operational health and model readiness of the FastAPI backend.
 * @returns {Promise<{ status: string, model_loaded: boolean, version: string, [key: string]: any }>}
 */
export async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    return {
      status: 'offline',
      api_status: 'offline',
      model_loaded: false,
      error: error.message || 'Cannot reach EcoVision backend server',
    };
  }
}

/**
 * Sends an image to the backend for YOLO11s waste detection.
 * @param {File|Blob} file - The image file to detect.
 * @param {Object} options - Inference parameters.
 * @param {number} [options.conf=0.15] - Confidence threshold (0.0 - 1.0).
 * @param {number} [options.iou=0.45] - IoU threshold for NMS.
 * @param {number} [options.imgsz=800] - Input image size.
 * @param {boolean} [options.agnostic_nms=true] - Class-agnostic NMS.
 * @returns {Promise<Object>} Detection response matching backend schema.
 */
export async function detectWasteImage(file, options = {}) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('conf', String(options.conf ?? 0.15));
  formData.append('iou', String(options.iou ?? 0.45));
  formData.append('imgsz', String(options.imgsz ?? 800));
  formData.append('agnostic_nms', String(Boolean(options.agnostic_nms ?? true)));

  const response = await fetch(`${API_BASE_URL}/api/detect`, {
    method: 'POST',
    body: formData,
  });

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    let message = 'Waste detection failed.';
    if (data && data.detail) {
      message = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    } else if (response.status === 413) {
      message = 'Image file is too large. Maximum allowed size is 15 MB.';
    } else if (response.status === 503) {
      message = 'Inference service is temporarily unavailable or model is still loading.';
    } else {
      message = `Server returned error (${response.status}: ${response.statusText})`;
    }
    const err = new Error(message);
    err.status = response.status;
    err.payload = data;
    throw err;
  }

  return data;
}

/**
 * Retrieves disposal guidance for a waste category.
 * @param {string} className - One of: plastic, metal, cardboard, glass, paper, general_trash.
 * @returns {Promise<Object>}
 */
export async function fetchDisposalGuidance(className) {
  const response = await fetch(`${API_BASE_URL}/api/disposal/${encodeURIComponent(className)}`, {
    headers: { 'Accept': 'application/json' },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new Error(errorData?.detail || `Disposal info not found for ${className}`);
  }

  return await response.json();
}

