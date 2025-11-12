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

type DragMode =
  | 'move'
  | 'resize-tl'
  | 'resize-tr'
  | 'resize-bl'
  | 'resize-br'
  | 'resize-t'
  | 'resize-b'
  | 'resize-l'
  | 'resize-r'
  | null

const ProcessPanel: FC<ProcessPanelProps> = ({ onScrollSync, syncScrollTop }) => {
  const [currentImage, setCurrentImage] = useState<string | null>(null)
  const [rectangles, setRectangles] = useState<Rectangle[]>([])
  const [shouldCenterImage, setShouldCenterImage] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const isScrollingSyncRef = useRef(false)
  const rectanglesRef = useRef<Rectangle[]>([])
  const animationFrameRef = useRef<number | null>(null)
  const resizeObserverRef = useRef<ResizeObserver | null>(null)
  const dragModeRef = useRef<DragMode>(null)

  // States cho drag & resize
  const [selectedRectIndex, setSelectedRectIndex] = useState<number | null>(null)
  const [dragMode, setDragMode] = useState<DragMode>(null)
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null)
  const [originalRect, setOriginalRect] = useState<Rectangle | null>(null)

  const { processedResults, isProcessing, currentImageIndex, setCurrentImageIndex } =
    useProcessing()

  // Sync rectanglesRef với rectangles state
  useEffect(() => {
    rectanglesRef.current = rectangles
  }, [rectangles])

  // Sync dragModeRef với dragMode state
  useEffect(() => {
    dragModeRef.current = dragMode
  }, [dragMode])

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

  // Function để draw rectangles (dùng ref thay vì state)
  const drawRectangles = () => {
    const image = imageRef.current
    const canvas = canvasRef.current
    if (!image || !canvas || rectanglesRef.current.length === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // CHỈ set canvas size khi thực sự thay đổi (tránh trigger ResizeObserver)
    const targetWidth = Math.round(image.clientWidth)
    const targetHeight = Math.round(image.clientHeight)

    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth
      canvas.height = targetHeight
    }

    const scaleX = canvas.width / image.naturalWidth
    const scaleY = canvas.height / image.naturalHeight

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    rectanglesRef.current.forEach((rect, index) => {
      const x = rect.x * scaleX
      const y = rect.y * scaleY
      const w = rect.w * scaleX
      const h = rect.h * scaleY

      const isSelected = selectedRectIndex === index

      // Draw rectangle
      ctx.strokeStyle = isSelected ? '#0000FF' : '#FF0000'
      ctx.lineWidth = isSelected ? 3 : 2
      ctx.strokeRect(x, y, w, h)

      // Draw number label
      ctx.fillStyle = isSelected ? '#0000FF' : '#FF0000'
      ctx.font = 'bold 14px sans-serif'
      ctx.fillText(`${index + 1}`, x + w + 5, y - 5)

      // Draw resize handles nếu selected
      if (isSelected) {
        const handleSize = 8
        ctx.fillStyle = '#0000FF'

        // 4 góc
        ctx.fillRect(x - handleSize / 2, y - handleSize / 2, handleSize, handleSize)
        ctx.fillRect(x + w - handleSize / 2, y - handleSize / 2, handleSize, handleSize)
        ctx.fillRect(x - handleSize / 2, y + h - handleSize / 2, handleSize, handleSize)
        ctx.fillRect(x + w - handleSize / 2, y + h - handleSize / 2, handleSize, handleSize)
      }
    })
  }

  // Setup ResizeObserver một lần
  useEffect(() => {
    if (!currentImage) return

    const image = imageRef.current
    const canvas = canvasRef.current
    if (!image || !canvas) return

    if (image.complete) {
      drawRectangles()
    } else {
      image.addEventListener('load', drawRectangles)
      return () => image.removeEventListener('load', drawRectangles)
    }

    // Setup ResizeObserver
    if (!resizeObserverRef.current) {
      resizeObserverRef.current = new ResizeObserver(() => {
        // CHỈ redraw khi KHÔNG đang drag (dùng ref để có giá trị realtime)
        if (!dragModeRef.current) {
          drawRectangles()
        }
      })
    }

    resizeObserverRef.current.observe(image)

    return () => {
      if (resizeObserverRef.current) {
        resizeObserverRef.current.disconnect()
      }
      image.removeEventListener('load', drawRectangles)
    }
  }, [currentImage, selectedRectIndex])

  // Cleanup animation frame khi unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [])

  // Helper: Get mouse position relative to canvas
  const getMousePos = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return null

    const rect = canvas.getBoundingClientRect()
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    }
  }

  // Helper: Check if point is in rectangle
  const isPointInRect = (
    px: number,
    py: number,
    rect: Rectangle,
    scaleX: number,
    scaleY: number
  ) => {
    const x = rect.x * scaleX
    const y = rect.y * scaleY
    const w = rect.w * scaleX
    const h = rect.h * scaleY

    return px >= x && px <= x + w && py >= y && py <= y + h
  }

  // Helper: Detect resize handle
  const detectResizeHandle = (
    px: number,
    py: number,
    rect: Rectangle,
    scaleX: number,
    scaleY: number
  ): DragMode => {
    const x = rect.x * scaleX
    const y = rect.y * scaleY
    const w = rect.w * scaleX
    const h = rect.h * scaleY
    const handleSize = 8

    // 4 góc
    if (Math.abs(px - x) < handleSize && Math.abs(py - y) < handleSize) return 'resize-tl'
    if (Math.abs(px - (x + w)) < handleSize && Math.abs(py - y) < handleSize) return 'resize-tr'
    if (Math.abs(px - x) < handleSize && Math.abs(py - (y + h)) < handleSize) return 'resize-bl'
    if (Math.abs(px - (x + w)) < handleSize && Math.abs(py - (y + h)) < handleSize)
      return 'resize-br'

    // 4 cạnh
    if (Math.abs(px - (x + w / 2)) < handleSize && Math.abs(py - y) < handleSize) return 'resize-t'
    if (Math.abs(px - (x + w / 2)) < handleSize && Math.abs(py - (y + h)) < handleSize)
      return 'resize-b'
    if (Math.abs(px - x) < handleSize && Math.abs(py - (y + h / 2)) < handleSize) return 'resize-l'
    if (Math.abs(px - (x + w)) < handleSize && Math.abs(py - (y + h / 2)) < handleSize)
      return 'resize-r'

    return null
  }

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = getMousePos(e)
    const image = imageRef.current
    if (!pos || !image) return

    const scaleX = image.clientWidth / image.naturalWidth
    const scaleY = image.clientHeight / image.naturalHeight

    // Check if clicking on selected rectangle's handle
    if (selectedRectIndex !== null) {
      const rect = rectangles[selectedRectIndex]
      const handle = detectResizeHandle(pos.x, pos.y, rect, scaleX, scaleY)

      if (handle) {
        setDragMode(handle)
        setDragStart(pos)
        setOriginalRect({ ...rect })
        return
      }
    }

    // Check if clicking on any rectangle
    for (let i = rectangles.length - 1; i >= 0; i--) {
      if (isPointInRect(pos.x, pos.y, rectangles[i], scaleX, scaleY)) {
        setSelectedRectIndex(i)
        setDragMode('move')
        setDragStart(pos)
        setOriginalRect({ ...rectangles[i] })
        return
      }
    }

    // Deselect
    setSelectedRectIndex(null)
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = getMousePos(e)
    const image = imageRef.current
    const canvas = canvasRef.current
    if (
      !pos ||
      !image ||
      !canvas ||
      !dragMode ||
      !dragStart ||
      !originalRect ||
      selectedRectIndex === null
    ) {
      // Update cursor
      if (pos && image && canvas && selectedRectIndex !== null) {
        const scaleX = image.clientWidth / image.naturalWidth
        const scaleY = image.clientHeight / image.naturalHeight
        const handle = detectResizeHandle(
          pos.x,
          pos.y,
          rectanglesRef.current[selectedRectIndex],
          scaleX,
          scaleY
        )

        if (handle) {
          const cursors: Record<string, string> = {
            'resize-tl': 'nwse-resize',
            'resize-tr': 'nesw-resize',
            'resize-bl': 'nesw-resize',
            'resize-br': 'nwse-resize',
            'resize-t': 'ns-resize',
            'resize-b': 'ns-resize',
            'resize-l': 'ew-resize',
            'resize-r': 'ew-resize'
          }
          canvas.style.cursor = cursors[handle] || 'default'
        } else if (
          isPointInRect(pos.x, pos.y, rectanglesRef.current[selectedRectIndex], scaleX, scaleY)
        ) {
          canvas.style.cursor = 'move'
        } else {
          canvas.style.cursor = 'default'
        }
      } else if (canvas) {
        canvas.style.cursor = 'default'
      }
      return
    }

    const scaleX = image.clientWidth / image.naturalWidth
    const scaleY = image.clientHeight / image.naturalHeight

    const dx = (pos.x - dragStart.x) / scaleX
    const dy = (pos.y - dragStart.y) / scaleY

    // Cancel previous animation frame
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
    }

    // Update rectanglesRef trực tiếp (không trigger re-render)
    const newRectangles = [...rectanglesRef.current]

    if (dragMode === 'move') {
      newRectangles[selectedRectIndex] = {
        ...originalRect,
        x: originalRect.x + dx,
        y: originalRect.y + dy
      }
    } else if (dragMode.startsWith('resize-')) {
      let newX = originalRect.x
      let newY = originalRect.y
      let newW = originalRect.w
      let newH = originalRect.h

      if (dragMode.includes('l')) {
        newX = originalRect.x + dx
        newW = originalRect.w - dx
      }
      if (dragMode.includes('r')) {
        newW = originalRect.w + dx
      }
      if (dragMode.includes('t')) {
        newY = originalRect.y + dy
        newH = originalRect.h - dy
      }
      if (dragMode.includes('b')) {
        newH = originalRect.h + dy
      }

      // Minimum size
      if (newW < 10) newW = 10
      if (newH < 10) newH = 10

      newRectangles[selectedRectIndex] = {
        ...originalRect,
        x: newX,
        y: newY,
        w: newW,
        h: newH
      }
    }

    rectanglesRef.current = newRectangles

    // Redraw trong animation frame
    animationFrameRef.current = requestAnimationFrame(() => {
      drawRectangles()
    })
  }

  const handleMouseUp = () => {
    // Cancel animation frame nếu có
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }

    // Commit thay đổi vào state
    if (dragMode) {
      setRectangles([...rectanglesRef.current])
      // Redraw một lần cuối sau khi commit
      requestAnimationFrame(() => {
        drawRectangles()
      })
    }

    setDragMode(null)
    setDragStart(null)
    setOriginalRect(null)
  }

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
              style={{ display: 'block', userSelect: 'none' }}
              draggable={false}
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0"
              style={{ display: 'block' }}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
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
    </div>
  )
}

export default ProcessPanel
