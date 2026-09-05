/**
 * Application Configuration
 * Resolves backend base URL dynamically from Vite environment variables.
 */

// Production fallback pointing to the live Render instance
const defaultApiUrl = import.meta.env.PROD 
  ? 'https://deepfake-detector-api.onrender.com' 
  : 'http://localhost:8000';

const rawApiUrl = import.meta.env.VITE_API_URL || defaultApiUrl;

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
