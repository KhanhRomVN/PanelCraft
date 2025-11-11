// src/renderer/src/presentation/pages/Home/components/Canvas/components/OriginalPanel/index.tsx
import { FC, useState, useEffect, useRef } from 'react'
import { useProcessing } from '../../../../../../contexts/ProcessingContext'

interface OriginalPanelProps {
  folderPath?: string
  onScrollSync?: (scrollTop: number) => void
  syncScrollTop?: number
}

const OriginalPanel: FC<OriginalPanelProps> = ({ folderPath, onScrollSync, syncScrollTop }) => {
  const [images, setImages] = useState<string[]>([])
  const [currentImage, setCurrentImage] = useState<string | null>(null)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const [shouldCenterImage, setShouldCenterImage] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const isScrollingSyncRef = useRef(false)

  const { setImagePaths } = useProcessing()

  useEffect(() => {
    if (folderPath) {
      loadImages(folderPath)
    } else {
      setImages([])
      setCurrentIndex(0)
    }
  }, [folderPath])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (images.length === 0) return

      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setCurrentIndex((prev) => (prev > 0 ? prev - 1 : images.length - 1))
      } else if (e.key === 'ArrowDown') {
        e.preventDefault()
        setCurrentIndex((prev) => (prev < images.length - 1 ? prev + 1 : 0))
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [images.length])

  const loadImages = async (path: string) => {
    setIsLoading(true)
    try {
      const result = await window.electronAPI.readFolderImages(path)
      setImages(result.valid)
      setImagePaths(result.valid) // Lưu danh sách ảnh vào context
      setCurrentIndex(0)
      if (result.valid.length > 0) {
        loadCurrentImage(result.valid[0])
      }
    } catch (error) {
      console.error('Failed to load images:', error)
      setImages([])
      setImagePaths([]) // Clear context khi có lỗi
      setCurrentImage(null)
    } finally {
      setIsLoading(false)
    }
  }

  const loadCurrentImage = async (imagePath: string) => {
    try {
      const dataUrl = await window.electronAPI.readImage(imagePath)
      setCurrentImage(dataUrl)
    } catch (error) {
      console.error('Failed to load current image:', error)
      setCurrentImage(null)
    }
  }

  useEffect(() => {
    if (images.length > 0) {
      loadCurrentImage(images[currentIndex])
    }
  }, [currentIndex, images])

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
        {isLoading ? (
          <div className="h-full flex items-center justify-center">
            <p className="text-sm text-text-secondary">Loading images...</p>
          </div>
        ) : currentImage ? (
          <img
            ref={imageRef}
            src={currentImage}
            alt={`Image ${currentIndex + 1}`}
            className="w-full h-auto object-contain"
            style={{ display: 'block' }}
          />
        ) : (
          <div className="h-full flex items-center justify-center">
            <p className="text-sm text-text-secondary">No images loaded</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default OriginalPanel
