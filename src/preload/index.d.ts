import { ElectronAPI } from '@electron-toolkit/preload'

// Define the API interface
interface API {
  sqlite: {
    createDatabase: (path: string) => Promise<any>
    openDatabase: (path: string) => Promise<any>
    closeDatabase: () => Promise<any>
    runQuery: (query: string, params?: any[]) => Promise<any>
    getAllRows: (query: string, params?: any[]) => Promise<any>
    getOneRow: (query: string, params?: any[]) => Promise<any>
    status: () => Promise<{ isConnected: boolean; message: string }>
  }
  storage: {
    set: (key: string, value: any) => Promise<void>
    get: (key: string) => Promise<any>
    remove: (key: string) => Promise<void>
  }
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
  popup: {
    showSession: (sessionData: any) => Promise<{ success: boolean; error?: string }>
    hideAndFocusMain: (sessionId: string) => Promise<{ success: boolean; error?: string }>
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
