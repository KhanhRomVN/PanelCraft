import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

// Export API type for TypeScript
export interface API {
  // Empty API - no custom methods
}

// Custom APIs for renderer
// Lấy API URL từ environment variable
const API_URL = process.env.VITE_API_URL || 'http://localhost:8001'

const api: API = {
  storage: {
    get: (key: string) => ipcRenderer.invoke('storage:get', key),
    set: (key: string, value: any) => ipcRenderer.invoke('storage:set', key, value),
    delete: (key: string) => ipcRenderer.invoke('storage:delete', key),
    clear: () => ipcRenderer.invoke('storage:clear')
  },
  testBackend: async () => {
    try {
      const response = await fetch(`${API_URL}/test`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      return await response.json()
    } catch (error) {
      console.error('Backend test failed:', error)
      throw error
    }
  },
  openFolderDialog: () => ipcRenderer.invoke('dialog:openFolder'),
  readFolderImages: (folderPath: string) => ipcRenderer.invoke('folder:readImages', folderPath),
  readImage: (imagePath: string) => ipcRenderer.invoke('image:read', imagePath),
  fetchImageFromBackend: (imageUrl: string) =>
    ipcRenderer.invoke('image:fetchFromBackend', imageUrl),

  // Models Management APIs
  checkModelsFolder: (folderPath: string) => ipcRenderer.invoke('models:checkFolder', folderPath),
  createModelsFolder: (folderPath: string) => ipcRenderer.invoke('models:createFolder', folderPath),
  downloadModelFile: (fileUrl: string, destPath: string) =>
    ipcRenderer.invoke('models:downloadFile', fileUrl, destPath),
  checkFileExists: (filePath: string) => ipcRenderer.invoke('models:checkFileExists', filePath),
  getSystemFonts: () => ipcRenderer.invoke('fonts:getSystemFonts')
}

// Use `contextBridge` APIs to expose Electron APIs to
// renderer only if context isolation is enabled, otherwise
// just add to the DOM global.
if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', {
      ...electronAPI,
      ipcRenderer: {
        on: (channel: string, listener: (...args: any[]) => void) => {
          ipcRenderer.on(channel, listener)
        },
        removeListener: (channel: string, listener: (...args: any[]) => void) => {
          ipcRenderer.removeListener(channel, listener)
        },
        send: (channel: string, ...args: any[]) => {
          ipcRenderer.send(channel, ...args)
        },
        invoke: (channel: string, ...args: any[]) => {
          return ipcRenderer.invoke(channel, ...args)
        }
      }
    })
    contextBridge.exposeInMainWorld('api', api)
    contextBridge.exposeInMainWorld('electronAPI', api) // For compatibility
  } catch (error) {
    console.error(error)
  }
} else {
  // TypeScript now knows these properties exist on Window
  ;(window as any).electron = {
    ...electronAPI,
    ipcRenderer: {
      on: ipcRenderer.on.bind(ipcRenderer),
      removeListener: ipcRenderer.removeListener.bind(ipcRenderer),
      send: ipcRenderer.send.bind(ipcRenderer),
      invoke: ipcRenderer.invoke.bind(ipcRenderer)
    }
  }
  ;(window as any).api = api
  ;(window as any).electronAPI = api
}
