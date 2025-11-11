export interface ModelInfo {
  name: string
  url: string
  filename: string
  category: 'text_detection' | 'ocr' | 'segmentation'
}

export interface ModelDownloadProgress {
  modelName: string
  category: string
  progress: number
  status: 'pending' | 'downloading' | 'completed' | 'error'
  error?: string
}

export interface ModelsConfig {
  modelsPath: string
  lastChecked: string
  isValid: boolean
}

export const MODELS_LIST: ModelInfo[] = [
  // Text Detection
  {
    name: 'Comic Text Detector',
    url: 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx',
    filename: 'comictextdetector.pt.onnx',
    category: 'text_detection'
  },

  // OCR Models
  {
    name: 'Manga OCR Config',
    url: 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/config.json',
    filename: 'config.json',
    category: 'ocr'
  },
  {
    name: 'Manga OCR Preprocessor Config',
    url: 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/preprocessor_config.json',
    filename: 'preprocessor_config.json',
    category: 'ocr'
  },
  {
    name: 'Manga OCR Model',
    url: 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/pytorch_model.bin',
    filename: 'pytorch_model.bin',
    category: 'ocr'
  },
  {
    name: 'Manga OCR Special Tokens',
    url: 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/special_tokens_map.json',
    filename: 'special_tokens_map.json',
    category: 'ocr'
  },
  {
    name: 'Manga OCR Tokenizer Config',
    url: 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/tokenizer_config.json',
    filename: 'tokenizer_config.json',
    category: 'ocr'
  },
  {
    name: 'Manga OCR Vocab',
    url: 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/vocab.txt',
    filename: 'vocab.txt',
    category: 'ocr'
  },

  // Segmentation Models
  {
    name: 'YOLOv8 Bubble Segmentation Config',
    url: 'https://huggingface.co/khanhromvn/manga_bubble_seg/resolve/main/config.yaml',
    filename: 'config.yaml',
    category: 'segmentation'
  },
  {
    name: 'YOLOv8 Bubble Segmentation Model',
    url: 'https://huggingface.co/khanhromvn/manga_bubble_seg/resolve/main/manga_bubble_seg.onnx',
    filename: 'manga_bubble_seg.onnx',
    category: 'segmentation'
  }
]

export const MODEL_CATEGORIES = {
  text_detection: 'Text Detection',
  ocr: 'OCR',
  segmentation: 'Segmentation'
} as const

export const MODELS_CONFIG_KEY = 'panelcraft-models-config'
