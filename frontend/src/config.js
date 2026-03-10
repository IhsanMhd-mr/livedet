// frontend/src/config.js
// Path and Configuration Management for Frontend

// ============================================================================
// ENVIRONMENT DETECTION
// ============================================================================

const isDevelopment = import.meta.env.DEV
const isProduction = import.meta.env.PROD
const environment = isDevelopment ? 'development' : 'production'

// ============================================================================
// API CONFIGURATION
// ============================================================================

// Backend API Base URL - Auto-detect or use environment variables
const getAPIBaseURL = () => {
  // Check for environment variable first
  const envURL = import.meta.env.VITE_API_BASE_URL
  if (envURL) {
    console.log('[CONFIG] Using VITE_API_BASE_URL from .env:', envURL)
    return envURL
  }

  // Auto-detect based on hostname
  const hostname = window.location.hostname
  const protocol = window.location.protocol

  // Development: localhost
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    const devURL = `${protocol}//localhost:8000`
    console.log('[CONFIG] Development environment - using:', devURL)
    return devURL
  }

  // Production: same origin
  if (isProduction) {
    const prodURL = `${protocol}//${hostname}:8000`
    console.log('[CONFIG] Production environment - using:', prodURL)
    return prodURL
  }

  // Fallback
  const fallbackURL = 'http://localhost:8000'
  console.log('[CONFIG] Fallback URL:', fallbackURL)
  return fallbackURL
}

const API_BASE_URL = getAPIBaseURL()

// ============================================================================
// WEBSOCKET CONFIGURATION
// ============================================================================

const getWSURL = () => {
  const envWS = import.meta.env.VITE_WS_URL
  if (envWS) {
    console.log('[CONFIG] Using VITE_WS_URL from .env:', envWS)
    return envWS
  }

  const hostname = window.location.hostname
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return 'ws://localhost:8765'
  }

  return `ws://${hostname}:8765`
}

const WS_URL = getWSURL()

// ============================================================================
// PATH CONFIGURATION
// ============================================================================

const PATHS = {
  // Prediction endpoint
  predict: `${API_BASE_URL}/predict`,
  
  // Results endpoints
  results: `${API_BASE_URL}/results`,
  resultById: (resultId) => `${API_BASE_URL}/results/${resultId}`,
  
  // Health/Status endpoints
  health: `${API_BASE_URL}/health`,
  status: `${API_BASE_URL}/status`,
  
  // Admin endpoints
  stats: `${API_BASE_URL}/stats`,
  cleanup: `${API_BASE_URL}/cleanup`,
}

// ============================================================================
// FRONTEND CONFIGURATION
// ============================================================================

const FRONTEND_CONFIG = {
  // Image display settings
  imageDisplay: {
    maxWidth: '100%',
    maxHeight: '500px',
    borderRadius: '8px',
  },
  
  // Upload settings
  upload: {
    maxFileSize: 10 * 1024 * 1024, // 10 MB
    acceptedFormats: ['image/jpeg', 'image/png', 'image/jpg'],
    acceptedExtensions: ['.jpg', '.jpeg', '.png'],
  },
  
  // Detection settings
  detection: {
    confidenceThreshold: 0.5,
    timeout: 60000, // 60 seconds
  },
  
  // UI settings
  ui: {
    showDebugLogs: isDevelopment,
    enableConsoleOutput: isDevelopment,
  },
}

// ============================================================================
// LOGGER UTILITY
// ============================================================================

const createLogger = (component) => {
  return {
    log: (step, message, data = null) => {
      if (FRONTEND_CONFIG.ui.enableConsoleOutput) {
        const prefix = `[${component}] ${step}:`
        if (data) {
          console.log(prefix, message, data)
        } else {
          console.log(prefix, message)
        }
      }
    },
    error: (step, message, data = null) => {
      const prefix = `[${component}:ERROR] ${step}:`
      if (data) {
        console.error(prefix, message, data)
      } else {
        console.error(prefix, message)
      }
    },
    warn: (step, message, data = null) => {
      const prefix = `[${component}:WARNING] ${step}:`
      if (data) {
        console.warn(prefix, message, data)
      } else {
        console.warn(prefix, message)
      }
    },
    debug: (step, message, data = null) => {
      if (FRONTEND_CONFIG.ui.showDebugLogs) {
        const prefix = `[${component}:DEBUG] ${step}:`
        if (data) {
          console.debug(prefix, message, data)
        } else {
          console.debug(prefix, message)
        }
      }
    },
  }
}

// ============================================================================
// API CLIENT
// ============================================================================

const apiClient = {
  /**
   * Upload image for detection
   */
  predict: async (file) => {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await fetch(PATHS.predict, {
      method: 'POST',
      body: formData,
    })
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`)
    }
    
    return response.json()
  },

  /**
   * Retrieve result by ID
   */
  getResultById: async (resultId) => {
    const response = await fetch(PATHS.resultById(resultId))
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`)
    }
    
    return response.json()
  },

  /**
   * List all results
   */
  getAllResults: async () => {
    const response = await fetch(PATHS.results)
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`)
    }
    
    return response.json()
  },

  /**
   * Check backend health
   */
  health: async () => {
    try {
      const response = await fetch(PATHS.health, { timeout: 5000 })
      return response.ok
    } catch {
      return false
    }
  },

  /**
   * Get backend statistics
   */
  stats: async () => {
    const response = await fetch(PATHS.stats)
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`)
    }
    
    return response.json()
  },
}

// ============================================================================
// EXPORT CONFIGURATION
// ============================================================================

export {
  // Environment
  isDevelopment,
  isProduction,
  environment,
  
  // API
  API_BASE_URL,
  WS_URL,
  PATHS,
  apiClient,
  
  // Configuration
  FRONTEND_CONFIG,
  
  // Logger
  createLogger,
}

// ============================================================================
// CONFIGURATION SUMMARY
// ============================================================================

console.log('═'.repeat(80))
console.log('FRONTEND CONFIGURATION LOADED')
console.log('═'.repeat(80))
console.log('Environment:', environment)
console.log('API Base URL:', API_BASE_URL)
console.log('WebSocket URL:', WS_URL)
console.log('Debug Enabled:', FRONTEND_CONFIG.ui.showDebugLogs)
console.log('═'.repeat(80))
