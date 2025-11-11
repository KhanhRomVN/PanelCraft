// src/renderer/src/presentation/pages/Home/utils/index.ts
import { SUPPORTED_IMAGE_FORMATS, MAX_FILE_SIZE } from '../constants'

/**
 * Check if file is a supported image format
 */
export const isImageFile = (fileName: string): boolean => {
  const extension = fileName.split('.').pop()?.toLowerCase()
  return SUPPORTED_IMAGE_FORMATS.includes(extension as any)
}

/**
 * Format file size to human readable string
 */
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes'

  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))

  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i]
}

/**
 * Validate file size
 */
export const isFileSizeValid = (size: number): boolean => {
  return size <= MAX_FILE_SIZE
}

/**
 * Calculate zoom to fit image in container
 */
export const calculateFitZoom = (
  imageWidth: number,
  imageHeight: number,
  containerWidth: number,
  containerHeight: number
): number => {
  const widthRatio = containerWidth / imageWidth
  const heightRatio = containerHeight / imageHeight

  return Math.min(widthRatio, heightRatio, 1) // Never zoom in, only zoom out
}

/**
 * Clamp value between min and max
 */
export const clamp = (value: number, min: number, max: number): number => {
  return Math.min(Math.max(value, min), max)
}

/**
 * Generate unique ID
 */
export const generateId = (): string => {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}

/**
 * Get file extension
 */
export const getFileExtension = (fileName: string): string => {
  return fileName.split('.').pop()?.toLowerCase() || ''
}

/**
 * Sort images by name, date, or size
 */
export const sortImages = (
  images: any[],
  sortBy: 'name' | 'date' | 'size',
  order: 'asc' | 'desc' = 'asc'
) => {
  const sorted = [...images].sort((a, b) => {
    let comparison = 0

    switch (sortBy) {
      case 'name':
        comparison = a.name.localeCompare(b.name)
        break
      case 'date':
        comparison = a.lastModified - b.lastModified
        break
      case 'size':
        comparison = a.size - b.size
        break
    }

    return order === 'asc' ? comparison : -comparison
  })

  return sorted
}
