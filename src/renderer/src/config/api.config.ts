export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:8001',
  ENDPOINTS: {
    HEALTH: '/health',
    TEST: '/test',
    PROCESS_IMAGES: '/api/v1/process-images'
  }
} as const

export const getApiUrl = (endpoint: string): string => {
  return `${API_CONFIG.BASE_URL}${endpoint}`
}
