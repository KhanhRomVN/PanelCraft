// src/renderer/src/presentation/pages/Home/index.tsx
import { useState, useEffect } from 'react'
import CanvasPanel from './components/Canvas'
import ControlPanel from './components/Control'
import ResizableSplitter from '../../../components/common/ResizableSplitter'

const HomePage = () => {
  const [selectedFolder, setSelectedFolder] = useState<string | null>(null)
  const [canvasWidth, setCanvasWidth] = useState(50)

  // Handle Ctrl+O keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'o') {
        e.preventDefault()
        handleOpenFolder()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const handleOpenFolder = async () => {
    try {
      const folderPath = await window.electronAPI.openFolderDialog()
      if (folderPath) {
        setSelectedFolder(folderPath)
      }
    } catch (error) {
      console.error('Failed to open folder:', error)
    }
  }

  const handleCanvasResize = (width: number) => {
    const clampedWidth = Math.max(20, Math.min(80, width))
    setCanvasWidth(clampedWidth)
  }

  return (
    <div className="h-screen bg-background flex overflow-hidden">
      {/* Canvas Panel */}
      <div style={{ width: `${canvasWidth}%` }}>
        <CanvasPanel selectedFolder={selectedFolder} onOpenFolder={handleOpenFolder} />
      </div>

      {/* Resizable Splitter */}
      <ResizableSplitter onResize={handleCanvasResize} />

      {/* Control Panel */}
      <div style={{ width: `${100 - canvasWidth}%` }}>
        <ControlPanel />
      </div>
    </div>
  )
}

export default HomePage
