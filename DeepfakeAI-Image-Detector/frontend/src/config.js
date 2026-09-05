/**
 * Application Configuration
 * Resolves backend base URL dynamically from Vite environment variables.
 */

// Primary active Render deployment for service srv-dabfn5qjnfac73aj0ab0
const defaultApiUrl = import.meta.env.PROD 
  ? 'https://deepfake-detector-api-nb2o.onrender.com' 
  : 'http://localhost:8000';

let rawApiUrl = import.meta.env.VITE_API_URL || defaultApiUrl;

// Automatically map deepfake-detector-api.onrender.com to the active Render service suffix (-nb2o)
if (rawApiUrl.includes('deepfake-detector-api.onrender.com') && !rawApiUrl.includes('-nb2o')) {
  rawApiUrl = rawApiUrl.replace('deepfake-detector-api.onrender.com', 'deepfake-detector-api-nb2o.onrender.com');
}

// Clean trailing slashes if configured with trailing slash in production dashboard
export const API_BASE_URL = rawApiUrl.replace(/\/+$/, '');

export const API_ENDPOINTS = {
  HEALTH: `${API_BASE_URL}/api/v1/health`,
  ANALYZE: `${API_BASE_URL}/api/v1/analyze`,
  ROOT: `${API_BASE_URL}/`,
  DOCS: `${API_BASE_URL}/docs`,
};

export const SCANNER_CONFIG = {
  MAX_FILE_SIZE_MB: 50,
  ALLOWED_TYPES: ['image/jpeg', 'image/png', 'image/webp'],
  TIMEOUT_MS: 90000,
};
