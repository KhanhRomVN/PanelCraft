// src/renderer/src/contexts/ModelsContext.tsx

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { MODELS_CONFIG_KEY, ModelsConfig } from '../types/models'

interface ModelsContextType {
  modelsPath: string | null
  isModelsReady: boolean
  setModelsPath: (path: string) => void
  checkModelsConfig: () => Promise<boolean>
}

const ModelsContext = createContext<ModelsContextType | undefined>(undefined)

export const useModels = () => {
  const context = useContext(ModelsContext)
  if (!context) {
    throw new Error('useModels must be used within ModelsProvider')
  }
  return context
}

interface ModelsProviderProps {
  children: ReactNode
}

export const ModelsProvider: React.FC<ModelsProviderProps> = ({ children }) => {
  const [modelsPath, setModelsPathState] = useState<string | null>(null)
  const [isModelsReady, setIsModelsReady] = useState(false)

  // Load models config từ localStorage
  useEffect(() => {
    const loadModelsConfig = async () => {
      try {
        const configStr = localStorage.getItem(MODELS_CONFIG_KEY)

        if (!configStr) {
          setIsModelsReady(false)
          return
        }

        const config: ModelsConfig = JSON.parse(configStr)

        // Kiểm tra xem folder còn hợp lệ không
        const result = await window.electronAPI.checkModelsFolder(config.modelsPath)

        if (result.isValid) {
          setModelsPathState(config.modelsPath)
          setIsModelsReady(true)
        } else {
          // Folder không còn hợp lệ, xóa config
          localStorage.removeItem(MODELS_CONFIG_KEY)
          setIsModelsReady(false)
        }
      } catch (error) {
        console.error('Error loading models config:', error)
        setIsModelsReady(false)
      }
    }

    loadModelsConfig()
  }, [])

  const setModelsPath = (path: string) => {
    const config: ModelsConfig = {
      modelsPath: path,
      lastChecked: new Date().toISOString(),
      isValid: true
    }

    localStorage.setItem(MODELS_CONFIG_KEY, JSON.stringify(config))
    setModelsPathState(path)
    setIsModelsReady(true)
  }

  const checkModelsConfig = async (): Promise<boolean> => {
    try {
      const configStr = localStorage.getItem(MODELS_CONFIG_KEY)

      if (!configStr) {
        return false
      }

      const config: ModelsConfig = JSON.parse(configStr)
      const result = await window.electronAPI.checkModelsFolder(config.modelsPath)

      if (!result.isValid) {
        localStorage.removeItem(MODELS_CONFIG_KEY)
        setModelsPathState(null)
        setIsModelsReady(false)
        return false
      }

      return true
    } catch (error) {
      console.error('Error checking models config:', error)
      return false
    }
  }

  return (
    <ModelsContext.Provider
      value={{
        modelsPath,
        isModelsReady,
        setModelsPath,
        checkModelsConfig
      }}
    >
      {children}
    </ModelsContext.Provider>
  )
}
