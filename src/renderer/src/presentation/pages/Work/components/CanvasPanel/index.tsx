// src/renderer/src/presentation/pages/Work/components/CanvasPanel/index.tsx
import React, { useState, useEffect, useRef } from 'react'
import { ImageFile } from '../../types/canvas.types'
import CustomModal from '../../../../../components/common/CustomModal'
import CustomButton from '../../../../../components/common/CustomButton'
import { FolderOpen, AlertTriangle, ImageIcon } from 'lucide-react'
import { validateFolder } from '../../utils/imageUtils'
import ControlPanel from '../ControlPanel'
import {
  BubbleSegmentationService,
  SegmentationResult,
  ProcessingProgress
} from '../../../../../services/bubbleSegmentationService'

const CanvasPanel: React.FC = () => {
  const [rawImages, setRawImages] = useState<ImageFile[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isDragging, setIsDragging] = useState(false)

  // Modal states
  const [showInvalidModal, setShowInvalidModal] = useState(false)
  const [invalidFiles, setInvalidFiles] = useState<string[]>([])
  const [showFormatModal, setShowFormatModal] = useState(false)
  const [detectedFormats, setDetectedFormats] = useState<string[]>([])
  const [selectedFormat, setSelectedFormat] = useState<string>('')
  const [pendingImages, setPendingImages] = useState<ImageFile[]>([])
  const [isConverting, setIsConverting] = useState(false)

  // Processing states
  const [processedResults, setProcessedResults] = useState<SegmentationResult[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingProgress, setProcessingProgress] = useState<ProcessingProgress | null>(null)
  const [selectedBubbleIndex, setSelectedBubbleIndex] = useState<number | null>(null)
  const [isModelLoading, setIsModelLoading] = useState(false)

  const canvasRef = useRef<HTMLDivElement>(null)
  const segmentationService = BubbleSegmentationService.getInstance()

  // Load model on mount
  useEffect(() => {
    const loadModel = async () => {
      setIsModelLoading(true)
      const success = await segmentationService.loadModel()
      if (!success) {
        console.error('Failed to load segmentation model')
      }
      setIsModelLoading(false)
    }

    loadModel()
  }, [])

  // Handle Ctrl+O
  useEffect(() => {
    const handleKeyDown = async (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'o') {
        e.preventDefault()
        await handleSelectFolder()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  // Handle Arrow Keys for navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (rawImages.length === 0) return

      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setCurrentIndex((prev) => Math.max(0, prev - 1))
      } else if (e.key === 'ArrowDown') {
        e.preventDefault()
        setCurrentIndex((prev) => Math.min(rawImages.length - 1, prev + 1))
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [rawImages.length])

  const handleSelectFolder = async () => {
    try {
      const result = await (window as any).api.folder.select()

      if (!result.success || result.canceled) {
        return
      }

      if (result.folderPath) {
        await processFolder(result.folderPath)
      }
    } catch (error) {
      console.error('Error selecting folder:', error)
    }
  }

  const processFolder = async (folderPath: string) => {
    try {
      const result = await (window as any).api.folder.read(folderPath)

      if (!result.success || !result.files) {
        return
      }

      const validation = validateFolder(result.files)

      if (validation.invalidFiles.length > 0) {
        setInvalidFiles(validation.invalidFiles)
        setPendingImages(validation.validImages)
        setShowInvalidModal(true)
        return
      }

      if (validation.formats.length > 1) {
        setDetectedFormats(validation.formats)
        setPendingImages(validation.validImages)
        setShowFormatModal(true)
        return
      }

      setRawImages(validation.validImages)
      setCurrentIndex(0)
      setProcessedResults([])
    } catch (error) {
      console.error('Error processing folder:', error)
    }
  }

  const handleInvalidModalContinue = () => {
    setShowInvalidModal(false)

    if (detectedFormats.length > 1) {
      const formats = Array.from(new Set(pendingImages.map((img) => img.format)))
      setDetectedFormats(formats)
      setShowFormatModal(true)
    } else {
      setRawImages(pendingImages)
      setCurrentIndex(0)
      setPendingImages([])
      setProcessedResults([])
    }
  }

  const handleFormatConvert = async () => {
    if (!selectedFormat) return

    setIsConverting(true)

    try {
      const filesToConvert = pendingImages
        .filter((img) => img.format !== selectedFormat)
        .map((img) => img.path)

      if (filesToConvert.length > 0) {
        const result = await (window as any).api.image.batchConvert(filesToConvert, selectedFormat)

        if (result.success && result.results) {
          const updatedImages = pendingImages.map((img) => {
            if (img.format !== selectedFormat) {
              const converted = result.results?.find(
                (r: { sourcePath: string }) => r.sourcePath === img.path
              )
              if (converted) {
                return {
                  ...img,
                  path: converted.targetPath,
                  format: selectedFormat,
                  name: img.name.replace(/\.[^.]+$/, `.${selectedFormat}`)
                }
              }
            }
            return img
          })

          setRawImages(updatedImages)
        }
      } else {
        setRawImages(pendingImages)
      }

      setCurrentIndex(0)
      setShowFormatModal(false)
      setPendingImages([])
      setSelectedFormat('')
      setProcessedResults([])
    } catch (error) {
      console.error('Error converting images:', error)
    } finally {
      setIsConverting(false)
    }
  }

  const handleStartProcessing = async (mode: 'sequential' | 'parallel') => {
    if (rawImages.length === 0 || isProcessing) return

    setIsProcessing(true)
    setProcessedResults([])
    setSelectedBubbleIndex(null)

    try {
      const imagesToProcess = rawImages.map((img) => ({
        path: img.path,
        name: img.name
      }))

      const results = await segmentationService.processImages(imagesToProcess, mode, (progress) => {
        setProcessingProgress(progress)
      })

      setProcessedResults(results)
    } catch (error) {
      console.error('Error processing images:', error)
    } finally {
      setIsProcessing(false)
      setProcessingProgress(null)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const items = Array.from(e.dataTransfer.items)

    for (const item of items) {
      const entry = item.webkitGetAsEntry()

      if (entry && entry.isDirectory) {
        // @ts-ignore
        const folderPath = entry.fullPath
        await processFolder(folderPath)
        break
      }
    }
  }

  const currentRawImage = rawImages[currentIndex]
  const currentProcessedResult = processedResults[currentIndex]

  return (
    <>
      <div
        ref={canvasRef}
        className="h-full w-full flex relative"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* Control Panel */}
        {rawImages.length > 0 && (
          <ControlPanel
            hasImages={rawImages.length > 0}
            isProcessing={isProcessing}
            onStart={handleStartProcessing}
          />
        )}

        {rawImages.length === 0 ? (
          <div
            className={`flex-1 flex flex-col items-center justify-center border-2 border-dashed rounded-lg transition-colors ${
              isDragging
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/10'
                : 'border-gray-300 dark:border-gray-600'
            }`}
          >
            <FolderOpen className="w-16 h-16 text-gray-400 dark:text-gray-500 mb-4" />
            <p className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-2">
              No folder selected
            </p>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
              Press <kbd className="px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded">Ctrl+O</kbd> to
              select folder or drag & drop folder here
            </p>
            <CustomButton variant="primary" size="md" onClick={handleSelectFolder}>
              Select Folder
            </CustomButton>
          </div>
        ) : (
          <>
            {/* Left Panel - Raw Images */}
            <div className="flex-1 border-r border-border-default overflow-auto">
              <div className="h-full flex flex-col">
                <div className="flex-shrink-0 px-4 py-3 border-b border-border-default bg-card-background">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-text-primary">Raw Images</h3>
                    <span className="text-xs text-text-secondary">
                      {currentIndex + 1} / {rawImages.length}
                    </span>
                  </div>
                </div>

                <div className="flex-1 overflow-auto">
                  {currentRawImage && (
                    <img
                      src={`local-image://${encodeURIComponent(currentRawImage.path)}`}
                      alt={currentRawImage.name}
                      className="w-full h-auto object-contain"
                    />
                  )}
                </div>

                <div className="flex-shrink-0 px-4 py-2 border-t border-border-default bg-card-background">
                  <p className="text-xs text-text-secondary truncate">{currentRawImage?.name}</p>
                </div>
              </div>
            </div>

            {/* Right Panel - Processed Images */}
            <div className="flex-1 overflow-auto">
              <div className="h-full flex flex-col">
                <div className="flex-shrink-0 px-4 py-3 border-b border-border-default bg-card-background">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-text-primary">Processed Images</h3>
                    {currentProcessedResult && (
                      <span className="text-xs text-text-secondary">
                        {currentProcessedResult.boxes.length} bubbles detected
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex-1 overflow-auto">
                  {isProcessing && processingProgress ? (
                    <div className="h-full flex flex-col items-center justify-center p-6">
                      <div className="w-full max-w-md">
                        <div className="mb-4">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                              Processing...
                            </span>
                            <span className="text-sm text-gray-500 dark:text-gray-400">
                              {processingProgress.currentIndex + 1} / {processingProgress.total}
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                              style={{
                                width: `${((processingProgress.currentIndex + 1) / processingProgress.total) * 100}%`
                              }}
                            />
                          </div>
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400 text-center">
                          {processingProgress.currentImage}
                        </p>
                      </div>
                    </div>
                  ) : currentProcessedResult ? (
                    <div className="relative">
                      <img
                        src={currentProcessedResult.canvas.toDataURL()}
                        alt={currentProcessedResult.imageName}
                        className="w-full h-auto object-contain"
                      />
                    </div>
                  ) : (
                    <div className="h-full flex items-center justify-center text-gray-400 dark:text-gray-500">
                      <div className="text-center">
                        <ImageIcon className="w-12 h-12 mx-auto mb-2 opacity-50" />
                        <p className="text-sm">No processed images yet</p>
                        <p className="text-xs mt-1">Click "Start Segmentation" to begin</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Invalid Files Modal */}
      <CustomModal
        isOpen={showInvalidModal}
        onClose={() => setShowInvalidModal(false)}
        title="Invalid Files Detected"
        size="md"
        actionText="Continue"
        cancelText="Cancel"
        onAction={handleInvalidModalContinue}
      >
        <div className="px-6 py-4">
          <div className="flex items-start gap-3 mb-4">
            <AlertTriangle className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm text-text-primary mb-2">
                The selected folder contains {invalidFiles.length} file(s) that are not supported
                image formats.
              </p>
              <p className="text-sm text-text-secondary">
                Click "Continue" to proceed with only valid image files (jpg, jpeg, png).
              </p>
            </div>
          </div>

          <div className="max-h-48 overflow-auto bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
            <p className="text-xs font-medium text-text-secondary mb-2">Invalid files:</p>
            <ul className="space-y-1">
              {invalidFiles.map((file, index) => (
                <li key={index} className="text-xs text-text-primary font-mono">
                  {file}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </CustomModal>

      {/* Format Conversion Modal */}
      <CustomModal
        isOpen={showFormatModal}
        onClose={() => setShowFormatModal(false)}
        title="Multiple Image Formats Detected"
        size="md"
        actionText="Convert"
        cancelText="Cancel"
        onAction={handleFormatConvert}
        actionDisabled={!selectedFormat}
        actionLoading={isConverting}
      >
        <div className="px-6 py-4">
          <div className="mb-4">
            <p className="text-sm text-text-primary mb-2">
              The folder contains images in multiple formats:{' '}
              <strong>{detectedFormats.join(', ')}</strong>
            </p>
            <p className="text-sm text-text-secondary">
              Select a target format to convert all images for consistency.
            </p>
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-text-primary mb-2">
              Target Format:
            </label>
            {detectedFormats.map((format) => (
              <label
                key={format}
                className="flex items-center gap-3 px-4 py-3 border border-border-default rounded-lg hover:bg-sidebar-itemHover cursor-pointer transition-colors"
              >
                <input
                  type="radio"
                  name="format"
                  value={format}
                  checked={selectedFormat === format}
                  onChange={(e) => setSelectedFormat(e.target.value)}
                  className="w-4 h-4 text-blue-600"
                />
                <span className="text-sm font-medium text-text-primary uppercase">{format}</span>
                <span className="text-xs text-text-secondary ml-auto">
                  {pendingImages.filter((img) => img.format === format).length} files
                </span>
              </label>
            ))}
          </div>
        </div>
      </CustomModal>
    </>
  )
}

export default CanvasPanel
