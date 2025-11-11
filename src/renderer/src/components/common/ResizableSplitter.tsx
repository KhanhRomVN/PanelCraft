import { FC, useState, useCallback, useEffect } from 'react'

interface ResizableSplitterProps {
  onResize?: (leftWidth: number) => void
  minSize?: number
  maxSize?: number
  orientation?: 'horizontal' | 'vertical'
}

const ResizableSplitter: FC<ResizableSplitterProps> = ({
  onResize,
  minSize = 20,
  maxSize = 80,
  orientation = 'horizontal'
}) => {
  const [isResizing, setIsResizing] = useState(false)
  const [isHovering, setIsHovering] = useState(false)

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    setIsResizing(true)
  }, [])

  useEffect(() => {
    if (!isResizing) return

    const handleMouseMove = (e: MouseEvent) => {
      if (orientation === 'horizontal') {
        const container = document.querySelector('.h-screen')
        if (!container) return
        const containerRect = container.getBoundingClientRect()
        const leftWidth = ((e.clientX - containerRect.left) / containerRect.width) * 100
        const clampedWidth = Math.max(minSize, Math.min(maxSize, leftWidth))
        onResize?.(clampedWidth)
      }
    }

    const handleMouseUp = () => {
      setIsResizing(false)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
  }, [isResizing, onResize, minSize, maxSize, orientation])

  return (
    <div
      className={`relative flex-shrink-0 ${
        orientation === 'horizontal' ? 'w-1 cursor-col-resize' : 'h-1 cursor-row-resize'
      } bg-border-default hover:bg-blue-500 transition-colors ${
        isResizing ? 'bg-blue-500' : ''
      } ${isHovering ? 'bg-blue-400' : ''}`}
      onMouseDown={handleMouseDown}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
    >
      <div
        className={`absolute ${
          orientation === 'horizontal' ? 'inset-y-0 -left-2 -right-2' : 'inset-x-0 -top-2 -bottom-2'
        }`}
      />
    </div>
  )
}

export default ResizableSplitter
