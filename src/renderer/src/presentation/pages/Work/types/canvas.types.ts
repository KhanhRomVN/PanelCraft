export interface ImageFile {
  name: string
  path: string
  format: string
}

export interface FolderValidation {
  validImages: ImageFile[]
  invalidFiles: string[]
  formats: string[]
}

export type PanelType = 'raw' | 'processed'
