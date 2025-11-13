import { ElectronAPI } from '@electron-toolkit/preload'

// Define the API interface
interface API {
  testBackend: () => Promise<{ message: string }>
  openFolderDialog: () => Promise<string | null>
  readFolderImages: (folderPath: string) => Promise<{ valid: string[]; invalid: string[] }>
  fetchImageFromBackend: (imageUrl: string) => Promise<string>
  readImage: (imagePath: string) => Promise<string>

  // Models Management APIs
  checkModelsFolder: (folderPath: string) => Promise<{
    isValid: boolean
    folders: { text_detection: boolean; ocr: boolean; segmentation: boolean }
  }>
  createModelsFolder: (folderPath: string) => Promise<{ success: boolean }>
  downloadModelFile: (fileUrl: string, destPath: string) => Promise<{ success: boolean }>
  checkFileExists: (filePath: string) => Promise<boolean>
  getSystemFonts: () => Promise<string[]>

  // Storage API
  storage: {
    get: (key: string) => Promise<any>
    set: (key: string, value: any) => Promise<void>
    delete: (key: string) => Promise<void>
    clear: () => Promise<void>
  }
}

interface ElectronIpcRenderer {
  on: (channel: string, listener: (...args: any[]) => void) => void
  removeListener: (channel: string, listener: (...args: any[]) => void) => void
  send: (channel: string, ...args: any[]) => void
  invoke: (channel: string, ...args: any[]) => Promise<any>
}

// Extend the Window interface to include our APIs
declare global {
  interface Window {
    electron: ElectronAPI & {
      ipcRenderer: ElectronIpcRenderer
    }
    api: API
    electronAPI: API
  }
}
