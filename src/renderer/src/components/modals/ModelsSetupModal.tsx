// src/renderer/src/components/modals/ModelsSetupModal.tsx

import React, { useState, useEffect } from 'react'
import CustomModal from '../common/CustomModal'
import CustomButton from '../common/CustomButton'
import { FolderOpen, Download, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { MODELS_LIST, MODEL_CATEGORIES, ModelDownloadProgress } from '../../types/models'
import { cn } from '../../shared/lib/utils'

interface ModelsSetupModalProps {
  isOpen: boolean
  onComplete: (modelsPath: string) => void
}

type SetupStep = 'select' | 'downloading' | 'completed' | 'error'

const ModelsSetupModal: React.FC<ModelsSetupModalProps> = ({ isOpen, onComplete }) => {
  const [step, setStep] = useState<SetupStep>('select')
  const [selectedPath, setSelectedPath] = useState<string>('')
  const [downloadProgress, setDownloadProgress] = useState<ModelDownloadProgress[]>([])
  const [errorMessage, setErrorMessage] = useState<string>('')
  const [isChecking, setIsChecking] = useState(false)

  // Khởi tạo download progress
  useEffect(() => {
    if (step === 'downloading') {
      const initialProgress: ModelDownloadProgress[] = MODELS_LIST.map((model) => ({
        modelName: model.name,
        category: model.category,
        progress: 0,
        status: 'pending'
      }))
      setDownloadProgress(initialProgress)
    }
  }, [step])

  const handleSelectExistingFolder = async () => {
    setIsChecking(true)
    setErrorMessage('')

    try {
      const folderPath = await window.electronAPI.openFolderDialog()

      if (!folderPath) {
        setIsChecking(false)
        return
      }

      // Kiểm tra folder có hợp lệ không
      const result = await window.electronAPI.checkModelsFolder(folderPath)

      if (result.isValid) {
        setSelectedPath(folderPath)
        setStep('completed')
        setTimeout(() => {
          onComplete(folderPath)
        }, 1000)
      } else {
        const missingFolders = Object.entries(result.folders)
          .filter(([_, exists]) => !exists)
          .map(([name]) => MODEL_CATEGORIES[name as keyof typeof MODEL_CATEGORIES])

        setErrorMessage(`Folder không hợp lệ. Thiếu các thư mục: ${missingFolders.join(', ')}`)
      }
    } catch (error) {
      console.error('Error selecting folder:', error)
      setErrorMessage('Có lỗi xảy ra khi kiểm tra folder')
    } finally {
      setIsChecking(false)
    }
  }

  const handleCreateNewFolder = async () => {
    setIsChecking(true)
    setErrorMessage('')

    try {
      const folderPath = await window.electronAPI.openFolderDialog()

      if (!folderPath) {
        setIsChecking(false)
        return
      }

      // Tạo folder structure
      await window.electronAPI.createModelsFolder(folderPath)

      setSelectedPath(folderPath)
      setStep('downloading')

      // Bắt đầu tải models
      await downloadModels(folderPath)
    } catch (error) {
      console.error('Error creating folder:', error)
      setErrorMessage('Có lỗi xảy ra khi tạo folder')
      setStep('error')
    } finally {
      setIsChecking(false)
    }
  }

  const downloadModels = async (basePath: string) => {
    try {
      for (let i = 0; i < MODELS_LIST.length; i++) {
        const model = MODELS_LIST[i]

        // Update status to downloading
        setDownloadProgress((prev) =>
          prev.map((p, idx) => (idx === i ? { ...p, status: 'downloading', progress: 0 } : p))
        )

        try {
          const destPath = `${basePath}/${model.category}/${model.filename}`

          // Simulate progress updates (vì không có streaming progress từ Electron)
          const progressInterval = setInterval(() => {
            setDownloadProgress((prev) =>
              prev.map((p, idx) =>
                idx === i && p.progress < 90 ? { ...p, progress: p.progress + 10 } : p
              )
            )
          }, 500)

          // Tải file
          await window.electronAPI.downloadModelFile(model.url, destPath)

          clearInterval(progressInterval)

          // Update status to completed
          setDownloadProgress((prev) =>
            prev.map((p, idx) => (idx === i ? { ...p, status: 'completed', progress: 100 } : p))
          )
        } catch (error) {
          console.error(`Failed to download ${model.name}:`, error)

          setDownloadProgress((prev) =>
            prev.map((p, idx) =>
              idx === i ? { ...p, status: 'error', error: 'Tải xuống thất bại' } : p
            )
          )

          throw new Error(`Failed to download ${model.name}`)
        }
      }

      // Tất cả đã tải xong
      setStep('completed')
      setTimeout(() => {
        onComplete(basePath)
      }, 2000)
    } catch (error) {
      console.error('Error downloading models:', error)
      setStep('error')
      setErrorMessage('Có lỗi xảy ra trong quá trình tải models')
    }
  }

  const renderSelectStep = () => (
    <div className="p-6 space-y-6">
      <div className="text-center space-y-3">
        <div className="flex justify-center">
          <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/20 rounded-full flex items-center justify-center">
            <FolderOpen className="w-8 h-8 text-blue-600 dark:text-blue-400" />
          </div>
        </div>
        <h3 className="text-lg font-semibold text-text-primary">Thiết lập thư mục Models</h3>
        <p className="text-sm text-text-secondary max-w-md mx-auto">
          Để sử dụng ứng dụng, bạn cần chọn hoặc tạo thư mục chứa các AI models cần thiết.
        </p>
      </div>

      {errorMessage && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-red-600 dark:text-red-400">{errorMessage}</p>
        </div>
      )}

      <div className="space-y-3">
        <CustomButton
          variant="primary"
          size="md"
          onClick={handleSelectExistingFolder}
          disabled={isChecking}
          loading={isChecking}
          icon={FolderOpen}
          className="w-full"
        >
          Chọn thư mục có sẵn
        </CustomButton>

        <CustomButton
          variant="secondary"
          size="md"
          onClick={handleCreateNewFolder}
          disabled={isChecking}
          icon={Download}
          className="w-full"
        >
          Tạo mới và tải models
        </CustomButton>
      </div>
    </div>
  )

  const renderDownloadingStep = () => {
    const totalModels = MODELS_LIST.length
    const completedModels = downloadProgress.filter((p) => p.status === 'completed').length
    const overallProgress = Math.round((completedModels / totalModels) * 100)

    return (
      <div className="p-6 space-y-6">
        <div className="text-center space-y-3">
          <div className="flex justify-center">
            <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/20 rounded-full flex items-center justify-center">
              <Loader2 className="w-8 h-8 text-blue-600 dark:text-blue-400 animate-spin" />
            </div>
          </div>
          <h3 className="text-lg font-semibold text-text-primary">Đang tải models...</h3>
          <p className="text-sm text-text-secondary">
            {completedModels} / {totalModels} models đã hoàn thành
          </p>
        </div>

        {/* Overall Progress */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-text-secondary">Tiến độ tổng thể</span>
            <span className="font-medium text-text-primary">{overallProgress}%</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div
              className="bg-blue-600 dark:bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${overallProgress}%` }}
            />
          </div>
        </div>

        {/* Models List */}
        <div className="space-y-2 max-h-[400px] overflow-y-auto">
          {downloadProgress.map((model, index) => (
            <div
              key={index}
              className="bg-card-background border border-border-default rounded-lg p-3 space-y-2"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {model.status === 'completed' && (
                    <CheckCircle className="w-4 h-4 text-green-600 dark:text-green-400" />
                  )}
                  {model.status === 'downloading' && (
                    <Loader2 className="w-4 h-4 text-blue-600 dark:text-blue-400 animate-spin" />
                  )}
                  {model.status === 'error' && (
                    <AlertCircle className="w-4 h-4 text-red-600 dark:text-red-400" />
                  )}
                  {model.status === 'pending' && (
                    <div className="w-4 h-4 rounded-full border-2 border-gray-300 dark:border-gray-600" />
                  )}
                  <div>
                    <p className="text-sm font-medium text-text-primary">{model.modelName}</p>
                    <p className="text-xs text-text-secondary">
                      {MODEL_CATEGORIES[model.category]}
                    </p>
                  </div>
                </div>
                <span
                  className={cn(
                    'text-xs font-medium',
                    model.status === 'completed' && 'text-green-600 dark:text-green-400',
                    model.status === 'downloading' && 'text-blue-600 dark:text-blue-400',
                    model.status === 'error' && 'text-red-600 dark:text-red-400',
                    model.status === 'pending' && 'text-gray-400 dark:text-gray-500'
                  )}
                >
                  {model.status === 'completed' && 'Hoàn thành'}
                  {model.status === 'downloading' && `${model.progress}%`}
                  {model.status === 'error' && 'Lỗi'}
                  {model.status === 'pending' && 'Đang chờ...'}
                </span>
              </div>

              {model.status === 'downloading' && (
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                  <div
                    className="bg-blue-600 dark:bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                    style={{ width: `${model.progress}%` }}
                  />
                </div>
              )}

              {model.status === 'error' && model.error && (
                <p className="text-xs text-red-600 dark:text-red-400">{model.error}</p>
              )}
            </div>
          ))}
        </div>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
          <p className="text-xs text-yellow-600 dark:text-yellow-400">
            <strong>Lưu ý:</strong> Quá trình tải xuống có thể mất vài phút. Vui lòng không đóng ứng
            dụng.
          </p>
        </div>
      </div>
    )
  }

  const renderCompletedStep = () => (
    <div className="p-6 space-y-6">
      <div className="text-center space-y-3">
        <div className="flex justify-center">
          <div className="w-16 h-16 bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center">
            <CheckCircle className="w-8 h-8 text-green-600 dark:text-green-400" />
          </div>
        </div>
        <h3 className="text-lg font-semibold text-text-primary">Thiết lập hoàn tất!</h3>
        <p className="text-sm text-text-secondary">
          Các models đã được cài đặt thành công. Ứng dụng sẽ tự động khởi động...
        </p>
      </div>

      <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
        <p className="text-xs text-green-600 dark:text-green-400">
          <strong>Thư mục models:</strong>
          <br />
          {selectedPath}
        </p>
      </div>
    </div>
  )

  const renderErrorStep = () => (
    <div className="p-6 space-y-6">
      <div className="text-center space-y-3">
        <div className="flex justify-center">
          <div className="w-16 h-16 bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center">
            <AlertCircle className="w-8 h-8 text-red-600 dark:text-red-400" />
          </div>
        </div>
        <h3 className="text-lg font-semibold text-text-primary">Có lỗi xảy ra</h3>
        <p className="text-sm text-text-secondary">
          {errorMessage || 'Không thể hoàn tất quá trình thiết lập'}
        </p>
      </div>

      <div className="flex justify-center">
        <CustomButton
          variant="primary"
          size="md"
          onClick={() => {
            setStep('select')
            setErrorMessage('')
            setDownloadProgress([])
          }}
        >
          Thử lại
        </CustomButton>
      </div>
    </div>
  )

  return (
    <CustomModal
      isOpen={isOpen}
      onClose={() => {}} // Không cho đóng
      title="Thiết lập Models"
      hideFooter
      size="lg"
      className="select-none"
    >
      {step === 'select' && renderSelectStep()}
      {step === 'downloading' && renderDownloadingStep()}
      {step === 'completed' && renderCompletedStep()}
      {step === 'error' && renderErrorStep()}
    </CustomModal>
  )
}

export default ModelsSetupModal
