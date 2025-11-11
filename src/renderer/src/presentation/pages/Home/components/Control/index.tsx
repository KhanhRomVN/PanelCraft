// src/renderer/src/presentation/pages/Home/components/Control/index.tsx
import { FC, useState } from 'react'
import CustomButton from '../../../../../components/common/CustomButton'
import { Play, FlaskConical } from 'lucide-react'
import { useProcessing } from '../../../../../contexts/ProcessingContext'
import { useModels } from '../../../../../contexts/ModelsContext'
import { BackendService } from '../../../../../services/backend.service'

const ControlPanel: FC = () => {
  const [testResult, setTestResult] = useState<string>('')
  const [isTestingBackend, setIsTestingBackend] = useState(false)
  const [error, setError] = useState<string>('')

  const {
    imagePaths,
    isProcessing,
    setIsProcessing,
    setProcessedResults,
    currentImageIndex,
    processedResults
  } = useProcessing()

  const { modelsPath } = useModels()

  const currentOCRResults = processedResults[currentImageIndex]?.ocr_results || []

  const handleTest = async () => {
    setIsTestingBackend(true)
    setTestResult('')

    try {
      const result = await window.electronAPI.testBackend()
      setTestResult(JSON.stringify(result, null, 2))
    } catch (error) {
      if (error instanceof TypeError && error.message.includes('fetch')) {
        setTestResult('Error: Backend server is not running. Please start the backend first.')
      } else {
        setTestResult(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`)
      }
    } finally {
      setIsTestingBackend(false)
    }
  }

  const handleStartProcessing = async () => {
    if (!modelsPath) {
      setError('Models path not configured')
      return
    }

    if (imagePaths.length === 0) {
      setError('No images to process')
      return
    }

    setIsProcessing(true)
    setError('')

    try {
      const response = await BackendService.processImages(imagePaths, modelsPath)

      if (response.success && response.data?.results) {
        setProcessedResults(response.data.results)
      } else {
        setError(response.error || 'Processing failed')
      }
    } catch (error) {
      console.error('Processing error:', error)
      setError(error instanceof Error ? error.message : 'Unknown error occurred')
    } finally {
      setIsProcessing(false)
    }
  }

  const hasImages = imagePaths.length > 0

  return (
    <div className="h-full bg-card-background flex flex-col items-center justify-center p-8">
      <div className="text-center space-y-6 max-w-md w-full">
        <h3 className="text-xl font-semibold text-text-primary">Control Panel</h3>

        {hasImages && (
          <div className="space-y-4">
            <p className="text-sm text-text-secondary">
              {imagePaths.length} image{imagePaths.length > 1 ? 's' : ''} loaded
            </p>

            <CustomButton
              variant="primary"
              size="lg"
              icon={Play}
              onClick={handleStartProcessing}
              disabled={isProcessing || !modelsPath}
              className="w-full"
            >
              {isProcessing ? 'Processing...' : 'Start Processing'}
            </CustomButton>

            {error && (
              <div className="p-3 bg-red-100 dark:bg-red-900/20 border border-red-300 dark:border-red-800 rounded-lg">
                <p className="text-sm text-red-700 dark:text-red-400">{error}</p>
              </div>
            )}
          </div>
        )}

        {processedResults.length > 0 && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium text-text-primary">OCR Results</h4>
              <span className="text-xs text-text-secondary">
                Image {currentImageIndex + 1} / {processedResults.length}
              </span>
            </div>

            {currentOCRResults.length > 0 ? (
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {currentOCRResults.map((ocr, index) => (
                  <div
                    key={`${ocr.segment_id}-${index}`}
                    className="p-3 bg-gray-100 dark:bg-gray-800 rounded-lg"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-text-secondary">
                        Segment {ocr.segment_id + 1}
                      </span>
                      <span className="text-xs text-text-secondary">
                        {(ocr.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <p className="text-sm text-text-primary break-words">{ocr.original_text}</p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
                <p className="text-sm text-text-secondary text-center">
                  {processedResults[currentImageIndex]?.cleaned_text_result
                    ? 'No text detected'
                    : 'Process images to see OCR results'}
                </p>
              </div>
            )}
          </div>
        )}

        <div className="pt-8 border-t border-border-default">
          <p className="text-sm text-text-secondary mb-4">Test backend connection</p>

          <CustomButton
            variant="secondary"
            size="md"
            icon={FlaskConical}
            onClick={handleTest}
            disabled={isTestingBackend}
            className="w-fit mx-auto"
          >
            {isTestingBackend ? 'Testing...' : 'Test Backend'}
          </CustomButton>

          {testResult && (
            <div className="mt-4 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg text-left">
              <pre className="text-xs text-text-primary overflow-auto max-h-40">{testResult}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ControlPanel
