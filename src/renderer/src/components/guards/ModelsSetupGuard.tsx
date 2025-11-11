// src/renderer/src/components/guards/ModelsSetupGuard.tsx

import React, { ReactNode, useEffect, useState } from 'react'
import { useModels } from '../../contexts/ModelsContext'
import ModelsSetupModal from '../modals/ModelsSetupModal'

interface ModelsSetupGuardProps {
  children: ReactNode
}

const ModelsSetupGuard: React.FC<ModelsSetupGuardProps> = ({ children }) => {
  const { isModelsReady, setModelsPath, checkModelsConfig } = useModels()
  const [showSetupModal, setShowSetupModal] = useState(false)
  const [isChecking, setIsChecking] = useState(true)

  useEffect(() => {
    const checkSetup = async () => {
      setIsChecking(true)

      // Đợi một chút để UI render
      await new Promise((resolve) => setTimeout(resolve, 500))

      const isReady = await checkModelsConfig()

      if (!isReady) {
        setShowSetupModal(true)
      }

      setIsChecking(false)
    }

    checkSetup()
  }, [])

  const handleSetupComplete = (modelsPath: string) => {
    setModelsPath(modelsPath)
    setShowSetupModal(false)
  }

  // Đang kiểm tra
  if (isChecking) {
    return (
      <div className="fixed inset-0 bg-background flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="text-text-secondary">Đang kiểm tra cấu hình...</p>
        </div>
      </div>
    )
  }

  // Hiển thị setup modal nếu chưa có models
  if (showSetupModal || !isModelsReady) {
    return (
      <>
        <div className="fixed inset-0 bg-background flex items-center justify-center">
          <div className="text-center space-y-4">
            <p className="text-text-secondary">Đang thiết lập models...</p>
          </div>
        </div>
        <ModelsSetupModal isOpen={true} onComplete={handleSetupComplete} />
      </>
    )
  }

  // Models đã sẵn sàng, render children
  return <>{children}</>
}

export default ModelsSetupGuard
