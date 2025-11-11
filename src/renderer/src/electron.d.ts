// src/renderer/src/electron.d.ts

export interface ElectronAPI {
  testBackend: () => Promise<{ message: string }>
  openFolderDialog: () => Promise<string | null>
  readFolderImages: (folderPath: string) => Promise<{ valid: string[]; invalid: string[] }>
  readImage: (imagePath: string) => Promise<string>

  // Models Management APIs
  checkModelsFolder: (folderPath: string) => Promise<{
    isValid: boolean
    folders: {
      text_detection: boolean
      ocr: boolean
      segmentation: boolean
    }
  }>
  createModelsFolder: (folderPath: string) => Promise<{ success: boolean }>
  downloadModelFile: (fileUrl: string, destPath: string) => Promise<{ success: boolean }>
  checkFileExists: (filePath: string) => Promise<boolean>
  fetchImageFromBackend: (imageUrl: string) => Promise<string>
}

declare global {
  interface Window {
    electronAPI: ElectronAPI
    electron: {
      ipcRenderer: {
        on: (channel: string, listener: (...args: any[]) => void) => void
        removeListener: (channel: string, listener: (...args: any[]) => void) => void
        send: (channel: string, ...args: any[]) => void
        invoke: (channel: string, ...args: any[]) => Promise<any>
      }
    }
    api: ElectronAPI
  }
}

export {}
