// src/renderer/src/components/common/ModelDownloadModal.tsx
import React, { useEffect, useState } from 'react'
import { Download, Check, AlertCircle, Loader2, FolderOpen } from 'lucide-react'
import CustomModal from '../common/CustomModal'
import { ModelService, DownloadProgress, REQUIRED_MODELS } from '../../services/modelService'
import { Progress } from '../ui/progress'

interface ModelDownloadModalProps {
  onComplete: () => void
}

const ModelDownloadModal: React.FC<ModelDownloadModalProps> = ({ onComplete }) => {
  const [isOpen, setIsOpen] = useState(false)
  const [isChecking, setIsChecking] = useState(true)
  const [isDownloading, setIsDownloading] = useState(false)
  const [progress, setProgress] = useState<DownloadProgress[]>([])
  const [error, setError] = useState<string | null>(null)
  const [selectedFolder, setSelectedFolder] = useState<string | null>(null)
  const [hasExistingModels, setHasExistingModels] = useState(false)

  const modelService = ModelService.getInstance()

  useEffect(() => {
    checkModels()
  }, [])

  useEffect(() => {
    const unsubscribe = modelService.onProgress((newProgress) => {
      setProgress(newProgress)
    })

    return unsubscribe
  }, [])

  const checkModels = async () => {
    setIsChecking(true)
    try {
      const result = await modelService.checkModelsExist()
      if (!result.exists) {
        setIsOpen(true)
        // Lấy đường dẫn đã lưu nếu có
        const savedPath = await window.api.storage.get('models_folder_path')
        if (savedPath) {
          setSelectedFolder(savedPath)
        }
      } else {
        onComplete()
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to check models')
      setIsOpen(true)
    } finally {
      setIsChecking(false)
    }
  }

  const handleSelectFolder = async () => {
    try {
      const result = await window.api.folder.select()
      if (result.success && result.folderPath) {
        setSelectedFolder(result.folderPath)
        setError(null)

        // Kiểm tra xem folder có models sẵn không
        const existingModels = await modelService.checkFolderForModels(result.folderPath)
        setHasExistingModels(existingModels)

        if (existingModels) {
          // Lưu path và thông báo tìm thấy models
          await window.api.storage.set('models_folder_path', result.folderPath)
        }
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to select folder')
    }
  }

  const handleUseExistingModels = async () => {
    if (!selectedFolder) return

    try {
      await window.api.storage.set('models_folder_path', selectedFolder)
      setIsOpen(false)
      onComplete()
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to save folder path')
    }
  }

  const handleDownload = async () => {
    if (!selectedFolder) {
      setError('Vui lòng chọn thư mục để lưu models')
      return
    }

    setIsDownloading(true)
    setError(null)

    try {
      const success = await modelService.downloadAllModels(selectedFolder)
      if (success) {
        // Lưu path sau khi download thành công
        await window.api.storage.set('models_folder_path', selectedFolder)
        setTimeout(() => {
          setIsOpen(false)
          onComplete()
        }, 1000)
      } else {
        setError('Some models failed to download. Please try again.')
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Download failed')
    } finally {
      setIsDownloading(false)
    }
  }

  const getOverallProgress = () => {
    if (progress.length === 0) return 0
    const completed = progress.filter((p) => p.status === 'completed').length
    return Math.round((completed / progress.length) * 100)
  }

  const getTotalSize = () => {
    let total = 0
    REQUIRED_MODELS.forEach((model) => {
      model.files.forEach((file) => {
        if (file.size) {
          const sizeStr = file.size.replace('~', '').replace('MB', '')
          total += parseFloat(sizeStr)
        }
      })
    })
    return `~${Math.round(total)}MB`
  }

  if (isChecking) {
    return (
      <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
        <div className="bg-card-background rounded-xl shadow-xl border border-border-default p-8">
          <div className="flex flex-col items-center gap-4">
            <Loader2 className="w-12 h-12 text-blue-500 animate-spin" />
            <p className="text-text-primary font-medium">Đang kiểm tra models...</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <CustomModal isOpen={isOpen} onClose={() => {}} title="Cấu hình Models" size="xl" hideFooter>
      <div className="p-6 space-y-6">
        {/* Warning Message */}
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
          <div className="flex gap-3">
            <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h4 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-1">
                Models chưa được cấu hình
              </h4>
              <p className="text-sm text-yellow-700 dark:text-yellow-400">
                Chọn thư mục để lưu models ({REQUIRED_MODELS.length} models, {getTotalSize()}) hoặc
                chọn thư mục đã có models sẵn.
              </p>
            </div>
          </div>
        </div>

        {/* Folder Selection */}
        <div className="space-y-3">
          <label className="text-sm font-medium text-text-primary">Thư mục lưu Models:</label>
          <div className="flex gap-2">
            <input
              type="text"
              value={selectedFolder || ''}
              readOnly
              placeholder="Chưa chọn thư mục..."
              className="flex-1 px-3 py-2 bg-background-secondary border border-border-default rounded-lg text-text-primary placeholder:text-text-secondary"
            />
            <button
              onClick={handleSelectFolder}
              disabled={isDownloading}
              className="px-4 py-2 bg-background-secondary hover:bg-background-tertiary border border-border-default text-text-primary rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <FolderOpen className="w-4 h-4" />
              Chọn thư mục
            </button>
          </div>
        </div>

        {/* Existing Models Found */}
        {hasExistingModels && (
          <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
            <div className="flex gap-3">
              <Check className="w-5 h-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h4 className="font-semibold text-green-800 dark:text-green-300 mb-1">
                  Tìm thấy models trong thư mục!
                </h4>
                <p className="text-sm text-green-700 dark:text-green-400 mb-3">
                  Thư mục này đã chứa các models cần thiết. Bạn có thể sử dụng ngay hoặc tải lại.
                </p>
                <div className="flex gap-2">
                  <button
                    onClick={handleUseExistingModels}
                    className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium transition-colors"
                  >
                    Sử dụng models có sẵn
                  </button>
                  <button
                    onClick={() => {
                      setHasExistingModels(false)
                    }}
                    className="px-4 py-2 bg-background-secondary hover:bg-background-tertiary border border-border-default text-text-primary rounded-lg text-sm font-medium transition-colors"
                  >
                    Tải lại models
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Model List */}
        <div className="space-y-4">
          <h3 className="font-semibold text-text-primary">Danh sách Models:</h3>
          {REQUIRED_MODELS.map((model) => {
            const modelProgress = progress.filter((p) => p.modelId === model.id)
            const completed = modelProgress.filter((p) => p.status === 'completed').length
            const total = model.files.length
            const modelStatus = completed === total ? 'completed' : 'pending'

            return (
              <div
                key={model.id}
                className="border border-border-default rounded-lg p-4 bg-card-background"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <h4 className="font-medium text-text-primary flex items-center gap-2">
                      {model.name}
                      {modelStatus === 'completed' && <Check className="w-4 h-4 text-green-500" />}
                    </h4>
                    <p className="text-sm text-text-secondary mt-1">{model.description}</p>
                  </div>
                </div>

                {/* Files Progress */}
                {isDownloading && (
                  <div className="mt-3 space-y-2">
                    {model.files.map((file) => {
                      const fileProgress = modelProgress.find((p) => p.fileName === file.name)
                      return (
                        <div key={file.name} className="text-xs text-text-secondary">
                          <div className="flex items-center justify-between mb-1">
                            <span className="truncate flex-1">{file.name}</span>
                            <span className="ml-2">
                              {fileProgress?.status === 'completed' && (
                                <Check className="w-3 h-3 text-green-500" />
                              )}
                              {fileProgress?.status === 'downloading' && (
                                <Loader2 className="w-3 h-3 text-blue-500 animate-spin" />
                              )}
                              {fileProgress?.status === 'error' && (
                                <AlertCircle className="w-3 h-3 text-red-500" />
                              )}
                            </span>
                          </div>
                          {file.size && <span className="text-gray-400">{file.size}</span>}
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* Overall Progress */}
        {isDownloading && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-text-secondary">Tiến độ tổng thể</span>
              <span className="text-text-primary font-medium">{getOverallProgress()}%</span>
            </div>
            <Progress value={getOverallProgress()} className="h-2" />
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
            <div className="flex gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0" />
              <p className="text-sm text-red-700 dark:text-red-400">{error}</p>
            </div>
          </div>
        )}

        {/* Action Button */}
        {!hasExistingModels && (
          <div className="flex justify-end pt-4 border-t border-border-default">
            <button
              onClick={handleDownload}
              disabled={isDownloading || !selectedFolder}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {isDownloading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Đang tải xuống...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4" />
                  Tải xuống Models
                </>
              )}
            </button>
          </div>
        )}
      </div>
    </CustomModal>
  )
}

export default ModelDownloadModal
