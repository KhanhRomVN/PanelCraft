import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

// Export API type for TypeScript
export interface API {
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

// Custom APIs for renderer
const api: API = {
  storage: {
    set: (key: string, value: any) => ipcRenderer.invoke('storage:set', key, value),
    get: (key: string) => ipcRenderer.invoke('storage:get', key),
    remove: (key: string) => ipcRenderer.invoke('storage:remove', key)
  },
  folder: {
    select: () => ipcRenderer.invoke('folder:select'),
    read: (folderPath: string) => ipcRenderer.invoke('folder:read', folderPath)
  },
  image: {
    convert: (sourcePath: string, targetFormat: string) =>
      ipcRenderer.invoke('image:convert', sourcePath, targetFormat),
    batchConvert: (files: string[], targetFormat: string) =>
      ipcRenderer.invoke('image:batch-convert', files, targetFormat)
  },
  model: {
    check: (modelId: string, customPath?: string) =>
      ipcRenderer.invoke('model:check', modelId, customPath),
    download: (params: {
      modelId: string
      fileName: string
      url: string
      localPath: string
      customBasePath?: string
    }) => ipcRenderer.invoke('model:download', params),
    getPath: (modelId: string) => ipcRenderer.invoke('model:get-path', modelId)
  }
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
