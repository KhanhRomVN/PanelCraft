// src/renderer/src/presentation/pages/Work/components/ControlPanel/index.tsx
import React, { useState } from 'react'
import CustomButton from '../../../../../components/common/CustomButton'
import { Play, Settings } from 'lucide-react'

export interface ControlPanelProps {
  hasImages: boolean
  isProcessing: boolean
  onStart: (mode: 'sequential' | 'parallel') => void
}

const ControlPanel: React.FC<ControlPanelProps> = ({ hasImages, isProcessing, onStart }) => {
  const [processingMode, setProcessingMode] = useState<'sequential' | 'parallel'>('sequential')

  const handleStart = () => {
    onStart(processingMode)
  }

  return (
    <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-10">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 p-4 min-w-[300px]">
        <div className="flex items-center gap-2 mb-4">
          <Settings className="w-5 h-5 text-gray-600 dark:text-gray-400" />
          <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
            Processing Control
          </h3>
        </div>

        <div className="space-y-3">
          {/* Processing Mode Selection */}
          <div>
            <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-2">
              Processing Mode:
            </label>
            <div className="space-y-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  value="sequential"
                  checked={processingMode === 'sequential'}
                  onChange={(e) => setProcessingMode(e.target.value as 'sequential' | 'parallel')}
                  disabled={isProcessing}
                  className="w-4 h-4 text-blue-600"
                />
                <div className="flex-1">
                  <span className="text-sm text-gray-700 dark:text-gray-300">Sequential</span>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Process images one by one (stable)
                  </p>
                </div>
              </label>

              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  value="parallel"
                  checked={processingMode === 'parallel'}
                  onChange={(e) => setProcessingMode(e.target.value as 'sequential' | 'parallel')}
                  disabled={isProcessing}
                  className="w-4 h-4 text-blue-600"
                />
                <div className="flex-1">
                  <span className="text-sm text-gray-700 dark:text-gray-300">Parallel</span>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Process all images at once (faster, uses more RAM)
                  </p>
                </div>
              </label>
            </div>
          </div>

          {/* Start Button */}
          <CustomButton
            variant="primary"
            size="md"
            onClick={handleStart}
            disabled={!hasImages || isProcessing}
            className="w-full"
          >
            <Play className="w-4 h-4 mr-2" />
            {isProcessing ? 'Processing...' : 'Start Segmentation'}
          </CustomButton>
        </div>
      </div>
    </div>
  )
}

export default ControlPanel
