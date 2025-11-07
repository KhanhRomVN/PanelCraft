// src/renderer/src/services/modelService.ts
export interface ModelFile {
  name: string
  url: string
  size?: string // Kích thước file để hiển thị (optional)
}

export interface ModelGroup {
  id: string
  name: string
  description: string
  files: ModelFile[]
  localPath: string // Đường dẫn lưu trong app
}

export const REQUIRED_MODELS: ModelGroup[] = [
  {
    id: 'text-detector',
    name: 'Comic Text Detector',
    description: 'Model phát hiện và làm sạch text trong bubble',
    localPath: 'models/text-detector',
    files: [
      {
        name: 'comictextdetector.pt.onnx',
        url: 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx',
        size: '~50MB'
      }
    ]
  },
  {
    id: 'manga-ocr',
    name: 'Manga OCR',
    description: 'Model nhận dạng chữ trong manga',
    localPath: 'models/manga-ocr',
    files: [
      {
        name: 'config.json',
        url: 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/config.json'
      },
      {
        name: 'preprocessor_config.json',
        url: 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/preprocessor_config.json'
      },
      {
        name: 'pytorch_model.bin',
        url: 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/pytorch_model.bin',
        size: '~400MB'
      },
      {
        name: 'special_tokens_map.json',
        url: 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/special_tokens_map.json'
      },
      {
        name: 'tokenizer_config.json',
        url: 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/tokenizer_config.json'
      }
    ]
  },
  {
    id: 'bubble-segmentation',
    name: 'Speech Bubble Segmentation',
    description: 'Model phân đoạn vùng bubble',
    localPath: 'models/bubble-seg',
    files: [
      {
        name: 'config.yaml',
        url: 'https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble/resolve/main/config.yaml'
      },
      {
        name: 'model.pt',
        url: 'https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble/resolve/main/model.pt',
        size: '~52MB'
      },
      {
        name: 'model_dynamic.onnx',
        url: 'https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble/resolve/main/model_dynamic.onnx',
        size: '~52MB'
      }
    ]
  }
]

export interface DownloadProgress {
  modelId: string
  fileName: string
  loaded: number
  total: number
  percentage: number
  status: 'pending' | 'downloading' | 'completed' | 'error'
  error?: string
}

export class ModelService {
  private static instance: ModelService
  private progressCallbacks: ((progress: DownloadProgress[]) => void)[] = []

  private constructor() {}

  static getInstance(): ModelService {
    if (!ModelService.instance) {
      ModelService.instance = new ModelService()
    }
    return ModelService.instance
  }

  onProgress(callback: (progress: DownloadProgress[]) => void) {
    this.progressCallbacks.push(callback)
    return () => {
      this.progressCallbacks = this.progressCallbacks.filter((cb) => cb !== callback)
    }
  }

  private notifyProgress(progress: DownloadProgress[]) {
    this.progressCallbacks.forEach((callback) => callback(progress))
  }

  async checkModelsExist(): Promise<{ exists: boolean; missing: ModelGroup[] }> {
    const missing: ModelGroup[] = []

    const savedPath = await window.api.storage.get('models_folder_path')
    if (!savedPath) {
      return {
        exists: false,
        missing: REQUIRED_MODELS
      }
    }

    for (const model of REQUIRED_MODELS) {
      const result = await window.api.model.check(model.id, savedPath)
      if (!result.exists) {
        missing.push(model)
      }
    }

    return {
      exists: missing.length === 0,
      missing
    }
  }

  async checkFolderForModels(folderPath: string): Promise<boolean> {
    try {
      for (const model of REQUIRED_MODELS) {
        const result = await window.api.model.check(model.id, folderPath)
        console.log(`[checkFolderForModels] Model ${model.id}:`, result)
        if (!result.exists || !result.files || result.files.length === 0) {
          return false
        }

        const requiredFileCount = model.files.length
        if (result.files.length < requiredFileCount) {
          console.log(
            `[checkFolderForModels] Model ${model.id} incomplete: ${result.files.length}/${requiredFileCount} files`
          )
          return false
        }
      }
      return true
    } catch (error) {
      console.error('Error checking folder for models:', error)
      return false
    }
  }

  async downloadAllModels(targetFolder: string): Promise<boolean> {
    const allProgress: Map<string, DownloadProgress> = new Map()

    // Khởi tạo progress cho tất cả files
    for (const model of REQUIRED_MODELS) {
      for (const file of model.files) {
        const key = `${model.id}-${file.name}`
        allProgress.set(key, {
          modelId: model.id,
          fileName: file.name,
          loaded: 0,
          total: 0,
          percentage: 0,
          status: 'pending'
        })
      }
    }

    try {
      // Download từng model
      for (const model of REQUIRED_MODELS) {
        for (const file of model.files) {
          const key = `${model.id}-${file.name}`

          // Update status to downloading
          const progress = allProgress.get(key)!
          progress.status = 'downloading'
          this.notifyProgress(Array.from(allProgress.values()))

          try {
            // Download file
            const result = await window.api.model.download({
              modelId: model.id,
              fileName: file.name,
              url: file.url,
              localPath: model.localPath,
              customBasePath: targetFolder
            })

            if (result.success) {
              progress.status = 'completed'
              progress.percentage = 100
              progress.loaded = progress.total || 100
              progress.total = progress.total || 100
            } else {
              progress.status = 'error'
              progress.error = result.error || 'Download failed'
            }
          } catch (error) {
            progress.status = 'error'
            progress.error = error instanceof Error ? error.message : 'Unknown error'
          }

          this.notifyProgress(Array.from(allProgress.values()))
        }
      }

      // Check if all completed successfully
      const allCompleted = Array.from(allProgress.values()).every((p) => p.status === 'completed')
      return allCompleted
    } catch (error) {
      console.error('Error downloading models:', error)
      return false
    }
  }
}
