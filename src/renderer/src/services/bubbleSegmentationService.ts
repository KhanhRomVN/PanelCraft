// src/renderer/src/services/bubbleSegmentationService.ts
import * as ort from 'onnxruntime-web'

ort.env.wasm.numThreads = 1
ort.env.wasm.simd = false
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/'

export interface SegmentationResult {
  imagePath: string
  imageName: string
  masks: Float32Array[]
  boxes: number[][]
  scores: number[]
  canvas: HTMLCanvasElement
}

export interface ProcessingProgress {
  currentIndex: number
  total: number
  currentImage: string
  status: 'processing' | 'completed' | 'error'
  error?: string
}

export class BubbleSegmentationService {
  private static instance: BubbleSegmentationService
  private session: ort.InferenceSession | null = null
  private modelPath: string = ''

  private constructor() {}

  static getInstance(): BubbleSegmentationService {
    if (!BubbleSegmentationService.instance) {
      BubbleSegmentationService.instance = new BubbleSegmentationService()
    }
    return BubbleSegmentationService.instance
  }

  async loadModel(): Promise<boolean> {
    try {
      const savedPath = await window.api.storage.get('models_folder_path')
      if (!savedPath) {
        throw new Error('Models folder not configured')
      }

      const modelPath = `${savedPath}/models/bubble-segmentation/model_dynamic.onnx`
      console.log('[BubbleSegmentation] Loading model from:', modelPath)

      const response = await fetch(`local-resource://${encodeURIComponent(modelPath)}`)
      if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.statusText}`)
      }

      const modelBuffer = await response.arrayBuffer()
      console.log('[BubbleSegmentation] Model loaded, size:', modelBuffer.byteLength, 'bytes')

      this.session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      })

      console.log('[BubbleSegmentation] Model session created successfully')
      return true
    } catch (error) {
      console.error('[BubbleSegmentation] Error loading model:', error)
      return false
    }
  }

  private async preprocessImage(imagePath: string): Promise<ort.Tensor> {
    return new Promise((resolve, reject) => {
      const img = new Image()
      img.crossOrigin = 'anonymous'

      img.onload = () => {
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d')!

        const targetSize = 640
        canvas.width = targetSize
        canvas.height = targetSize

        ctx.drawImage(img, 0, 0, targetSize, targetSize)

        const imageData = ctx.getImageData(0, 0, targetSize, targetSize)
        const data = imageData.data

        const float32Data = new Float32Array(3 * targetSize * targetSize)

        for (let i = 0; i < targetSize * targetSize; i++) {
          float32Data[i] = data[i * 4] / 255.0
          float32Data[targetSize * targetSize + i] = data[i * 4 + 1] / 255.0
          float32Data[2 * targetSize * targetSize + i] = data[i * 4 + 2] / 255.0
        }

        const tensor = new ort.Tensor('float32', float32Data, [1, 3, targetSize, targetSize])
        resolve(tensor)
      }

      img.onerror = reject
      img.src = `local-image://${encodeURIComponent(imagePath)}`
    })
  }

  private generateRandomColor(): string {
    const hue = Math.floor(Math.random() * 360)
    return `hsla(${hue}, 70%, 60%, 0.5)`
  }

  private async drawSegmentation(
    imagePath: string,
    boxes: number[][],
    masks: Float32Array[]
  ): Promise<HTMLCanvasElement> {
    return new Promise((resolve, reject) => {
      const img = new Image()
      img.crossOrigin = 'anonymous'

      img.onload = () => {
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d')!

        canvas.width = img.width
        canvas.height = img.height

        ctx.drawImage(img, 0, 0)

        boxes.forEach((box, idx) => {
          const [x1, y1, x2, y2] = box
          const color = this.generateRandomColor()

          ctx.fillStyle = color
          ctx.fillRect(x1, y1, x2 - x1, y2 - y1)

          ctx.strokeStyle = color.replace('0.5', '1.0')
          ctx.lineWidth = 2
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
        })

        resolve(canvas)
      }

      img.onerror = reject
      img.src = `local-image://${encodeURIComponent(imagePath)}`
    })
  }

  async processImage(imagePath: string, imageName: string): Promise<SegmentationResult> {
    if (!this.session) {
      throw new Error('Model not loaded')
    }

    const inputTensor = await this.preprocessImage(imagePath)

    const results = await this.session.run({
      images: inputTensor
    })

    const boxes: number[][] = []
    const masks: Float32Array[] = []
    const scores: number[] = []

    const outputBoxes = results.output0.data as Float32Array
    const numDetections = outputBoxes.length / 6

    for (let i = 0; i < numDetections; i++) {
      const offset = i * 6
      const x1 = outputBoxes[offset]
      const y1 = outputBoxes[offset + 1]
      const x2 = outputBoxes[offset + 2]
      const y2 = outputBoxes[offset + 3]
      const score = outputBoxes[offset + 4]

      if (score > 0.5) {
        boxes.push([x1, y1, x2, y2])
        scores.push(score)
        masks.push(new Float32Array(0))
      }
    }

    const canvas = await this.drawSegmentation(imagePath, boxes, masks)

    return {
      imagePath,
      imageName,
      masks,
      boxes,
      scores,
      canvas
    }
  }

  async processImages(
    imagePaths: Array<{ path: string; name: string }>,
    mode: 'sequential' | 'parallel',
    onProgress?: (progress: ProcessingProgress) => void
  ): Promise<SegmentationResult[]> {
    const results: SegmentationResult[] = []

    if (mode === 'sequential') {
      for (let i = 0; i < imagePaths.length; i++) {
        const { path, name } = imagePaths[i]

        onProgress?.({
          currentIndex: i,
          total: imagePaths.length,
          currentImage: name,
          status: 'processing'
        })

        try {
          const result = await this.processImage(path, name)
          results.push(result)
        } catch (error) {
          console.error(`Error processing ${name}:`, error)
          onProgress?.({
            currentIndex: i,
            total: imagePaths.length,
            currentImage: name,
            status: 'error',
            error: error instanceof Error ? error.message : 'Unknown error'
          })
        }
      }
    } else {
      const promises = imagePaths.map(async ({ path, name }, index) => {
        onProgress?.({
          currentIndex: index,
          total: imagePaths.length,
          currentImage: name,
          status: 'processing'
        })

        try {
          return await this.processImage(path, name)
        } catch (error) {
          console.error(`Error processing ${name}:`, error)
          throw error
        }
      })

      results.push(...(await Promise.all(promises)))
    }

    onProgress?.({
      currentIndex: imagePaths.length,
      total: imagePaths.length,
      currentImage: '',
      status: 'completed'
    })

    return results
  }
}
