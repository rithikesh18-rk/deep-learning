/**
 * Centralized API & App Configuration
 * Resolves backend base URL dynamically from Vite environment variables.
 */

// Fallback to local FastAPI server on port 8000 during local development
const rawApiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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
