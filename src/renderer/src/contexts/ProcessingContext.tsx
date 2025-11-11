// src/renderer/src/contexts/ProcessingContext.tsx

import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react'
import { ProcessedImageResult } from '../services/backend.service'

interface ProcessingContextType {
  imagePaths: string[]
  setImagePaths: (paths: string[]) => void
  processedResults: ProcessedImageResult[]
  setProcessedResults: (results: ProcessedImageResult[]) => void
  isProcessing: boolean
  setIsProcessing: (value: boolean) => void
  currentImageIndex: number
  setCurrentImageIndex: (index: number) => void
}

const ProcessingContext = createContext<ProcessingContextType | undefined>(undefined)

export const useProcessing = () => {
  const context = useContext(ProcessingContext)
  if (!context) {
    throw new Error('useProcessing must be used within ProcessingProvider')
  }
  return context
}

interface ProcessingProviderProps {
  children: ReactNode
}

export const ProcessingProvider: React.FC<ProcessingProviderProps> = ({ children }) => {
  const [imagePaths, setImagePaths] = useState<string[]>([])
  const [processedResults, setProcessedResults] = useState<ProcessedImageResult[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [currentImageIndex, setCurrentImageIndex] = useState(0)

  // THÊM: Reset currentImageIndex khi processedResults thay đổi
  useEffect(() => {
    if (processedResults.length === 0) {
      setCurrentImageIndex(0)
    } else if (currentImageIndex >= processedResults.length) {
      setCurrentImageIndex(processedResults.length - 1)
    }
  }, [processedResults.length])

  return (
    <ProcessingContext.Provider
      value={{
        imagePaths,
        setImagePaths,
        processedResults,
        setProcessedResults,
        isProcessing,
        setIsProcessing,
        currentImageIndex,
        setCurrentImageIndex
      }}
    >
      {children}
    </ProcessingContext.Provider>
  )
}
