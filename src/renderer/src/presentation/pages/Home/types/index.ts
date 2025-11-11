// src/renderer/src/presentation/pages/Home/types/index.ts

export interface ImageFile {
  id: string
  name: string
  path: string
  size: number
  type: string
  thumbnail?: string
  lastModified: number
}

export interface FolderData {
  path: string
  name: string
  images: ImageFile[]
  totalSize: number
}

export interface CanvasState {
  zoom: number
  rotation: number
  position: { x: number; y: number }
}

export interface ProcessingOptions {
  resize?: {
    width?: number
    height?: number
    maintainAspectRatio: boolean
  }
  format?: 'jpeg' | 'png' | 'webp'
  quality?: number
  filters?: {
    brightness?: number
    contrast?: number
    saturation?: number
  }
}

export interface ProcessedImage extends ImageFile {
  originalId: string
  processedPath: string
  options: ProcessingOptions
  processedAt: number
}
