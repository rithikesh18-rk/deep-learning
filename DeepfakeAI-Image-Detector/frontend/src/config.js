/**
 * Centralized API & App Configuration
 * Resolves backend base URL dynamically from Vite environment variables.
 */

// Production fallback pointing to the live verified Render instance
const defaultApiUrl = import.meta.env.PROD 
  ? 'https://deepfake-detector-api-nb2o.onrender.com' 
  : 'http://localhost:8000';

let rawApiUrl = import.meta.env.VITE_API_URL || defaultApiUrl;

// Automatically map deepfake-detector-api.onrender.com to the active service suffix (-nb2o)
if (rawApiUrl.includes('deepfake-detector-api.onrender.com') && !rawApiUrl.includes('-nb2o')) {
  rawApiUrl = rawApiUrl.replace('deepfake-detector-api.onrender.com', 'deepfake-detector-api-nb2o.onrender.com');
}

// Clean trailing slashes if configured with trailing slash in production dashboard
export const API_BASE_URL = rawApiUrl.replace(/\/+$/, '');

export const API_ENDPOINTS = {
  ANALYZE: `${API_BASE_URL}/api/v1/analyze`,
  HEALTH: `${API_BASE_URL}/api/v1/health`,
  ROOT: `${API_BASE_URL}/`,
};

export default {
  API_BASE_URL,
  API_ENDPOINTS,
};
