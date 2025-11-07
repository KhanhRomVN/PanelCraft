import { ElectronAPI } from '@electron-toolkit/preload'

// Define the API interface
interface API {
  storage: {
    set: (key: string, value: any) => Promise<void>
    get: (key: string) => Promise<any>
    remove: (key: string) => Promise<void>
  }
  folder: {
    select: () => Promise<{
      success: boolean
      folderPath?: string
      canceled?: boolean
      error?: string
    }>
    read: (folderPath: string) => Promise<{
      success: boolean
      files?: Array<{ name: string; path: string }>
      error?: string
    }>
  }
  image: {
    convert: (
      sourcePath: string,
      targetFormat: string
    ) => Promise<{ success: boolean; targetPath?: string; error?: string }>
    batchConvert: (
      files: string[],
      targetFormat: string
    ) => Promise<{ success: boolean; results?: any[]; error?: string }>
  }
  model: {
    check: (modelId: string, customPath?: string) => Promise<{ exists: boolean; files?: string[] }>
    download: (params: {
      modelId: string
      fileName: string
      url: string
      localPath: string
      customBasePath?: string
    }) => Promise<{ success: boolean; path?: string; error?: string }>
    getPath: (modelId: string) => Promise<{ path: string }>
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
