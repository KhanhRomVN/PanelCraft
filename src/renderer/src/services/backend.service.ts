import { API_CONFIG, getApiUrl } from '../config/api.config'

export interface SegmentData {
  id: number
  box: [number, number, number, number] // [x1, y1, x2, y2]
  score: number
  rectangle?: [number, number, number, number] // [x, y, w, h]
}

export interface OCRResult {
  segment_id: number
  original_text: string
  confidence: number
}

export interface ProcessedImageResult {
  image_index: number
  original_path: string
  original_dimensions: [number, number]
  segmentation_result?: string
  cleaned_text_result?: string
  segments: SegmentData[]
  rectangles: Array<{ id: number; x: number; y: number; w: number; h: number }>
  ocr_results: OCRResult[]
}

export interface ProcessImagesRequest {
  image_paths: string[]
  model_base_path: string
  steps?: string[]
  options?: Record<string, any>
}

export interface ProcessImagesResponse {
  success: boolean
  message: string
  data?: {
    results: ProcessedImageResult[]
  }
  error?: string
}

export class BackendService {
  private static readonly BASE_URL = API_CONFIG.BASE_URL

  static async processImages(
    imagePaths: string[],
    modelsPath: string
  ): Promise<ProcessImagesResponse> {
    try {
      const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.PROCESS_IMAGES), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          image_paths: imagePaths,
          model_base_path: modelsPath,
          steps: ['full_pipeline']
        } as ProcessImagesRequest)
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data: ProcessImagesResponse = await response.json()
      return data
    } catch (error) {
      console.error('Failed to process images:', error)
      throw error
    }
  }

  static async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.HEALTH))
      return response.ok
    } catch (error) {
      return false
    }
  }
}
