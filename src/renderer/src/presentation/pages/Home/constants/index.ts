// src/renderer/src/presentation/pages/Home/constants/index.ts

export const SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp', 'tiff'] as const

export const MAX_FILE_SIZE = 50 * 1024 * 1024 // 50MB

export const DEFAULT_CANVAS_STATE = {
  zoom: 1,
  rotation: 0,
  position: { x: 0, y: 0 }
} as const

export const ZOOM_LEVELS = {
  MIN: 0.1,
  MAX: 5,
  STEP: 0.1,
  DEFAULT: 1
} as const

export const DEFAULT_PROCESSING_OPTIONS = {
  format: 'jpeg' as const,
  quality: 90,
  resize: {
    maintainAspectRatio: true
  }
} as const

export const KEYBOARD_SHORTCUTS = {
  OPEN_FOLDER: 'ctrl+o',
  RESET_VIEW: 'ctrl+r',
  ZOOM_IN: 'ctrl++',
  ZOOM_OUT: 'ctrl+-',
  FIT_TO_SCREEN: 'ctrl+0'
} as const
