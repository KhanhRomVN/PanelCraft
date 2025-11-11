// src/renderer/src/presentation/pages/Home/components/Canvas/components/ProcessPanel/index.tsx
import { FC, useState, useEffect, useRef } from 'react'
import { API_CONFIG } from '../../../../../../config/api.config'
import { useProcessing } from '../../../../../../contexts/ProcessingContext'

interface Rectangle {
  id: number
  x: number
  y: number
  w: number
  h: number
}

interface ProcessPanelProps {
  folderPath?: string
  onScrollSync?: (scrollTop: number) => void
  syncScrollTop?: number
}

const ProcessPanel: FC<ProcessPanelProps> = ({ onScrollSync, syncScrollTop }) => {
  const [currentImage, setCurrentImage] = useState<string | null>(null)
  const [rectangles, setRectangles] = useState<Rectangle[]>([])
  const [shouldCenterImage, setShouldCenterImage] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const isScrollingSyncRef = useRef(false)

  const { processedResults, isProcessing, currentImageIndex, setCurrentImageIndex } =
    useProcessing()

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (processedResults.length === 0) return

      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setCurrentImageIndex(
          currentImageIndex > 0 ? currentImageIndex - 1 : processedResults.length - 1
        )
      } else if (e.key === 'ArrowDown') {
        e.preventDefault()
        setCurrentImageIndex(
          currentImageIndex < processedResults.length - 1 ? currentImageIndex + 1 : 0
        )
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [processedResults.length, currentImageIndex, setCurrentImageIndex])

  useEffect(() => {
    const loadProcessedImage = async () => {
      if (processedResults.length > 0 && processedResults[currentImageIndex]) {
        const result = processedResults[currentImageIndex]
        const imageUrl = result.cleaned_text_result

        if (imageUrl) {
          try {
            const absoluteUrl = `${API_CONFIG.BASE_URL}${imageUrl}`
            const dataUrl = await window.electronAPI.fetchImageFromBackend(absoluteUrl)
            setCurrentImage(dataUrl)

            // THÊM: Load rectangles metadata
            setRectangles(result.rectangles || [])
          } catch (error) {
            console.error('Failed to load processed image:', error)
            setCurrentImage(null)
            setRectangles([])
          }
        } else {
          setCurrentImage(null)
          setRectangles([])
        }
      } else {
        setCurrentImage(null)
        setRectangles([])
      }
    }

    loadProcessedImage()
  }, [currentImageIndex, processedResults])

  // THÊM: Draw rectangles overlay
  useEffect(() => {
    if (!currentImage || rectangles.length === 0) return

    const image = imageRef.current
    const canvas = canvasRef.current
    if (!image || !canvas) return

    const drawRectangles = () => {
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      // Set canvas size to match displayed image
      canvas.width = image.clientWidth
      canvas.height = image.clientHeight

      const scaleX = canvas.width / image.naturalWidth
      const scaleY = canvas.height / image.naturalHeight

      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Draw red rectangles
      ctx.strokeStyle = '#FF0000'
      ctx.lineWidth = 2

      rectangles.forEach((rect, index) => {
        const x = rect.x * scaleX
        const y = rect.y * scaleY
        const w = rect.w * scaleX
        const h = rect.h * scaleY

        ctx.strokeRect(x, y, w, h)

        // Draw number label
        ctx.fillStyle = '#FF0000'
        ctx.font = 'bold 14px sans-serif'
        ctx.fillText(`${index + 1}`, x + w + 5, y - 5)
      })
    }

    if (image.complete) {
      drawRectangles()
    } else {
      image.addEventListener('load', drawRectangles)
      return () => image.removeEventListener('load', drawRectangles)
    }
  }, [currentImage, rectangles])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const handleScroll = () => {
      if (isScrollingSyncRef.current) {
        isScrollingSyncRef.current = false
        return
      }
      onScrollSync?.(container.scrollTop)
    }

    container.addEventListener('scroll', handleScroll)
    return () => container.removeEventListener('scroll', handleScroll)
  }, [onScrollSync])

  useEffect(() => {
    const container = containerRef.current
    if (!container || syncScrollTop === undefined) return

    isScrollingSyncRef.current = true
    container.scrollTop = syncScrollTop
  }, [syncScrollTop])

  useEffect(() => {
    const checkImageSize = () => {
      const container = containerRef.current
      const image = imageRef.current
      if (!container || !image) return

      const containerHeight = container.clientHeight
      const imageHeight = image.naturalHeight

      setShouldCenterImage(imageHeight < containerHeight)
    }

    const image = imageRef.current
    if (image) {
      if (image.complete) {
        checkImageSize()
      } else {
        image.addEventListener('load', checkImageSize)
        return () => image.removeEventListener('load', checkImageSize)
      }
    }
  }, [currentImage])

  return (
    <div className="h-full w-full bg-background flex flex-col">
      <div
        ref={containerRef}
        className={`flex-1 overflow-auto ${shouldCenterImage ? 'flex items-center' : ''}`}
      >
        {isProcessing ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center space-y-3">
              <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto" />
              <p className="text-sm text-text-secondary">Processing images...</p>
            </div>
          </div>
        ) : currentImage ? (
          <div className="relative w-full">
            <img
              ref={imageRef}
              src={currentImage}
              alt={`Processed Image ${currentImageIndex + 1}`}
              className="w-full h-auto object-contain"
              style={{ display: 'block' }}
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 pointer-events-none"
              style={{ display: 'block' }}
            />
          </div>
        ) : (
          <div className="h-full flex items-center justify-center">
            <p className="text-sm text-text-secondary">
              {processedResults.length > 0
                ? 'No processed image available'
                : 'Click "Start Processing" to begin'}
            </p>
          </div>
        )}
      </div>

      {processedResults.length > 0 && (
        <div className="px-4 py-2 bg-card-background border-t border-border-default">
          <p className="text-xs text-text-secondary text-center">
            Image {currentImageIndex + 1} of {processedResults.length}
          </p>
        </div>
      )}
    </div>
  )
}

export default ProcessPanel
