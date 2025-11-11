// src/renderer/src/presentation/pages/Home/hooks/index.ts
import { useState, useCallback, useEffect } from 'react'
import { CanvasState, ImageFile } from '../types'
import { DEFAULT_CANVAS_STATE, ZOOM_LEVELS } from '../constants'
import { clamp } from '../utils'

/**
 * Hook to manage canvas state (zoom, rotation, position)
 */
export const useCanvasState = () => {
  const [canvasState, setCanvasState] = useState<CanvasState>(DEFAULT_CANVAS_STATE)

  const setZoom = useCallback((zoom: number) => {
    setCanvasState((prev) => ({
      ...prev,
      zoom: clamp(zoom, ZOOM_LEVELS.MIN, ZOOM_LEVELS.MAX)
    }))
  }, [])

  const zoomIn = useCallback(() => {
    setCanvasState((prev) => ({
      ...prev,
      zoom: clamp(prev.zoom + ZOOM_LEVELS.STEP, ZOOM_LEVELS.MIN, ZOOM_LEVELS.MAX)
    }))
  }, [])

  const zoomOut = useCallback(() => {
    setCanvasState((prev) => ({
      ...prev,
      zoom: clamp(prev.zoom - ZOOM_LEVELS.STEP, ZOOM_LEVELS.MIN, ZOOM_LEVELS.MAX)
    }))
  }, [])

  const rotate = useCallback((angle: number) => {
    setCanvasState((prev) => ({
      ...prev,
      rotation: (prev.rotation + angle) % 360
    }))
  }, [])

  const resetCanvas = useCallback(() => {
    setCanvasState(DEFAULT_CANVAS_STATE)
  }, [])

  const setPosition = useCallback((position: { x: number; y: number }) => {
    setCanvasState((prev) => ({
      ...prev,
      position
    }))
  }, [])

  return {
    canvasState,
    setZoom,
    zoomIn,
    zoomOut,
    rotate,
    resetCanvas,
    setPosition
  }
}

/**
 * Hook to manage selected images
 */
export const useImageSelection = () => {
  const [selectedImages, setSelectedImages] = useState<string[]>([])

  const selectImage = useCallback((id: string) => {
    setSelectedImages((prev) => {
      if (prev.includes(id)) {
        return prev.filter((imgId) => imgId !== id)
      }
      return [...prev, id]
    })
  }, [])

  const selectMultiple = useCallback((ids: string[]) => {
    setSelectedImages(ids)
  }, [])

  const clearSelection = useCallback(() => {
    setSelectedImages([])
  }, [])

  const isSelected = useCallback(
    (id: string) => {
      return selectedImages.includes(id)
    },
    [selectedImages]
  )

  return {
    selectedImages,
    selectImage,
    selectMultiple,
    clearSelection,
    isSelected
  }
}

/**
 * Hook to manage keyboard shortcuts
 */
export const useKeyboardShortcuts = (callbacks: {
  onOpenFolder?: () => void
  onReset?: () => void
  onZoomIn?: () => void
  onZoomOut?: () => void
  onFitToScreen?: () => void
}) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isCtrl = e.ctrlKey || e.metaKey

      if (isCtrl) {
        switch (e.key.toLowerCase()) {
          case 'o':
            e.preventDefault()
            callbacks.onOpenFolder?.()
            break
          case 'r':
            e.preventDefault()
            callbacks.onReset?.()
            break
          case '=':
          case '+':
            e.preventDefault()
            callbacks.onZoomIn?.()
            break
          case '-':
            e.preventDefault()
            callbacks.onZoomOut?.()
            break
          case '0':
            e.preventDefault()
            callbacks.onFitToScreen?.()
            break
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [callbacks])
}

/**
 * Hook to manage folder loading state
 */
export const useFolderLoader = () => {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [images, setImages] = useState<ImageFile[]>([])

  const loadFolder = useCallback(async (folderPath: string) => {
    setIsLoading(true)
    setError(null)

    try {
      // TODO: Implement actual folder loading logic
      // This is a placeholder
      await new Promise((resolve) => setTimeout(resolve, 1000))

      // Mock data
      const mockImages: ImageFile[] = []
      setImages(mockImages)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load folder')
    } finally {
      setIsLoading(false)
    }
  }, [])

  const clearImages = useCallback(() => {
    setImages([])
    setError(null)
  }, [])

  return {
    isLoading,
    error,
    images,
    loadFolder,
    clearImages
  }
}
