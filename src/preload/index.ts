import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

// Export API type for TypeScript
export interface API {
  cloudDatabase: {
    testConnection: (connectionString: string) => Promise<{ success: boolean; error?: string }>
    connect: (connectionString: string) => Promise<{ success: boolean; error?: string }>
    disconnect: () => Promise<{ success: boolean; error?: string }>
    initializeSchema: () => Promise<{ success: boolean; error?: string }>
    query: (
      query: string,
      params?: any[]
    ) => Promise<{ success: boolean; rows: any[]; rowCount: number; error?: string }>
    status: () => Promise<{ isConnected: boolean }>
  }
  vocabulary: {
    update: (item: any) => Promise<{ success: boolean; error?: string }>
  }
  storage: {
    set: (key: string, value: any) => Promise<void>
    get: (key: string) => Promise<any>
    remove: (key: string) => Promise<void>
  }
  popup: {
    showSession: (sessionData: any) => Promise<{ success: boolean; error?: string }>
    hideAndFocusMain: (sessionId: string) => Promise<{ success: boolean; error?: string }>
  }
}

// Custom APIs for renderer
const api: API = {
  cloudDatabase: {
    testConnection: (connectionString: string) =>
      ipcRenderer.invoke('cloud-db:test-connection', connectionString),
    connect: (connectionString: string) => ipcRenderer.invoke('cloud-db:connect', connectionString),
    disconnect: () => ipcRenderer.invoke('cloud-db:disconnect'),
    initializeSchema: () => ipcRenderer.invoke('cloud-db:initialize-schema'),
    query: (query: string, params?: any[]) => ipcRenderer.invoke('cloud-db:query', query, params),
    status: () => ipcRenderer.invoke('cloud-db:status')
  },
  vocabulary: {
    update: async (item: any) => {
      try {
        const { getCloudDatabase } = await import('../renderer/src/services/CloudDatabaseService')
        const db = getCloudDatabase()
        if (!db) throw new Error('Database not connected')

        if (item.item_type === 'word' || item.item_type === 'phrase') {
          await db.updateVocabularyItem(item)
        } else if (
          item.item_type === 'tense' ||
          item.item_type === 'structure' ||
          item.item_type === 'rule' ||
          item.item_type === 'pattern'
        ) {
          await db.updateGrammarItem(item)
        }

        return { success: true }
      } catch (error) {
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Update failed'
        }
      }
    }
  },
  popup: {
    showSession: (sessionData: any) => ipcRenderer.invoke('popup:show-session', sessionData),
    hideAndFocusMain: (sessionId: string) =>
      ipcRenderer.invoke('popup:hide-and-focus-main', sessionId)
  },
  storage: {
    set: (key: string, value: any) => ipcRenderer.invoke('storage:set', key, value),
    get: (key: string) => ipcRenderer.invoke('storage:get', key),
    remove: (key: string) => ipcRenderer.invoke('storage:remove', key)
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
