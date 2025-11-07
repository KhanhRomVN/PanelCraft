import * as dotenv from 'dotenv'
dotenv.config()
import { app, shell, BrowserWindow, ipcMain, dialog } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import * as fs from 'fs'
import * as path from 'path'
import { protocol } from 'electron'

let mainWindow: BrowserWindow | null = null

// Storage file path
const getStorageFilePath = () => {
  const userDataPath = app.getPath('userData')
  return path.join(userDataPath, 'email-manager-config.json')
}

// Register custom protocol for loading local images
function registerLocalImageProtocol(): void {
  protocol.registerFileProtocol('local-image', (request, callback) => {
    const url = request.url.replace('local-image://', '')
    const decodedPath = decodeURIComponent(url)
    callback({ path: decodedPath })
  })
}

function registerLocalResourceProtocol(): void {
  protocol.registerFileProtocol('local-resource', (request, callback) => {
    const url = request.url.replace('local-resource://', '')
    const decodedPath = decodeURIComponent(url)

    let mimeType = 'application/octet-stream'
    if (decodedPath.endsWith('.wasm')) {
      mimeType = 'application/wasm'
    } else if (decodedPath.endsWith('.onnx')) {
      mimeType = 'application/octet-stream'
    }

    callback({
      path: decodedPath,
      headers: {
        'Content-Type': mimeType
      }
    })
  })
}
// Setup IPC handlers for model operations
function setupModelHandlers() {
  const https = require('https')
  const http = require('http')

  // Check if model exists
  ipcMain.handle('model:check', async (_event, modelId: string, customPath?: string) => {
    try {
      const basePath = customPath || app.getPath('userData')
      const modelPath = path.join(basePath, 'models', modelId)

      console.log('[model:check] Checking path:', modelPath)
      console.log('[model:check] Custom path:', customPath)
      console.log('[model:check] Base path:', basePath)

      if (!fs.existsSync(modelPath)) {
        console.log('[model:check] Path does not exist')
        return { exists: false, files: [] }
      }

      const allFiles = fs.readdirSync(modelPath)
      const files = allFiles.filter((file) => !file.startsWith('.'))
      console.log('[model:check] Found files:', files)
      console.log('[model:check] All files (including hidden):', allFiles)
      return { exists: files.length > 0, files }
    } catch (error) {
      console.error('[model:check] Error:', error)
      return { exists: false, files: [] }
    }
  })

  // Download model file
  ipcMain.handle(
    'model:download',
    async (
      event,
      params: {
        modelId: string
        fileName: string
        url: string
        localPath: string
        customBasePath?: string
      }
    ) => {
      try {
        const basePath = params.customBasePath || app.getPath('userData')
        const modelDir = path.join(basePath, 'models', params.modelId)
        const filePath = path.join(modelDir, params.fileName)

        console.log('[model:download] ==================')
        console.log('[model:download] Base path:', basePath)
        console.log('[model:download] Model ID:', params.modelId)
        console.log('[model:download] Target directory:', modelDir)
        console.log('[model:download] File path:', filePath)
        console.log('[model:download] Download URL:', params.url)

        if (!fs.existsSync(modelDir)) {
          console.log('[model:download] Creating directory:', modelDir)
          fs.mkdirSync(modelDir, { recursive: true })
        }

        const protocol = params.url.startsWith('https') ? https : http

        return new Promise((resolve, reject) => {
          const request = protocol.get(params.url, (response) => {
            if (response.statusCode === 302 || response.statusCode === 301) {
              console.log('[model:download] Following redirect to:', response.headers.location)
              const redirectProtocol = response.headers.location?.startsWith('https') ? https : http
              redirectProtocol
                .get(response.headers.location!, (redirectResponse) => {
                  handleDownloadResponse(redirectResponse, filePath, event, params, resolve, reject)
                })
                .on('error', (error) => {
                  console.error('[model:download] Redirect request error:', error.message)
                  reject(error)
                })
              return
            }

            handleDownloadResponse(response, filePath, event, params, resolve, reject)
          })

          request.on('error', (error) => {
            console.error('[model:download] Request error:', error.message)
            fs.unlink(filePath, () => {})
            resolve({
              success: false,
              error: error.message
            })
          })

          request.setTimeout(300000, () => {
            request.destroy()
            fs.unlink(filePath, () => {})
            resolve({
              success: false,
              error: 'Download timeout (5 minutes)'
            })
          })
        })

        function handleDownloadResponse(
          response: any,
          filePath: string,
          event: any,
          params: any,
          resolve: any,
          reject: any
        ) {
          const totalSize = parseInt(response.headers['content-length'] || '0', 10)
          let downloadedSize = 0
          let lastProgressUpdate = Date.now()

          console.log('[model:download] Total size:', totalSize, 'bytes')

          const file = fs.createWriteStream(filePath)

          response.on('data', (chunk: Buffer) => {
            downloadedSize += chunk.length

            const now = Date.now()
            if (now - lastProgressUpdate > 500 || downloadedSize === totalSize) {
              const percentage = totalSize > 0 ? Math.round((downloadedSize / totalSize) * 100) : 0
              console.log(
                `[model:download] Progress: ${percentage}% (${downloadedSize}/${totalSize} bytes)`
              )

              event.sender.send('model:download-progress', {
                modelId: params.modelId,
                fileName: params.fileName,
                loaded: downloadedSize,
                total: totalSize,
                percentage
              })

              lastProgressUpdate = now
            }
          })

          response.pipe(file)

          file.on('finish', () => {
            file.close((err) => {
              if (err) {
                console.error('[model:download] Error closing file:', err.message)
                fs.unlink(filePath, () => {})
                resolve({
                  success: false,
                  error: err.message
                })
                return
              }

              const stats = fs.statSync(filePath)
              console.log('[model:download] Download completed:', filePath)
              console.log('[model:download] File size:', stats.size, 'bytes')

              if (totalSize > 0 && stats.size !== totalSize) {
                console.warn(
                  `[model:download] Size mismatch: expected ${totalSize}, got ${stats.size}`
                )
              }

              resolve({ success: true, path: filePath })
            })
          })

          file.on('error', (error) => {
            console.error('[model:download] File write error:', error.message)
            fs.unlink(filePath, () => {})
            resolve({
              success: false,
              error: error.message
            })
          })

          response.on('error', (error: any) => {
            console.error('[model:download] Response error:', error.message)
            fs.unlink(filePath, () => {})
            resolve({
              success: false,
              error: error.message
            })
          })
        }
      } catch (error) {
        console.error('[model:download] Error:', error)
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Failed to download model'
        }
      }
    }
  )

  // Get model path
  ipcMain.handle('model:get-path', async (_event, modelId: string) => {
    const userDataPath = app.getPath('userData')
    const modelPath = path.join(userDataPath, 'models', modelId)
    return { path: modelPath }
  })
}

function createWindow(): void {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false,
      nodeIntegration: false,
      contextIsolation: true
    }
  })

  // Configure CSP to allow custom protocols
  mainWindow.webContents.session.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          "default-src 'self' 'unsafe-inline' 'unsafe-eval'; " +
            "connect-src 'self' http: https: ws: wss: local-image: local-resource:; " +
            "img-src 'self' data: blob: local-image: https:; " +
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; " +
            "style-src 'self' 'unsafe-inline';"
        ]
      }
    })
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow?.maximize()
    mainWindow?.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  // HMR for renderer base on electron-vite cli.
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])

    // Enable auto-reload in development
    mainWindow.webContents.on('did-fail-load', () => {
      if (mainWindow) {
        mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'] as string)
      }
    })

    if (is.dev && mainWindow) {
      mainWindow.webContents.on('did-finish-load', () => {
        if (mainWindow) {
          mainWindow.blur()
          mainWindow.webContents.openDevTools({ mode: 'detach' })
        }
      })
    }
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

// Setup IPC handlers for storage operations
function setupStorageHandlers() {
  // Set storage value
  ipcMain.handle('storage:set', async (_event, key: string, value: any) => {
    try {
      const storagePath = getStorageFilePath()
      let data: Record<string, any> = {}

      // Read existing data if file exists
      if (fs.existsSync(storagePath)) {
        const fileContent = fs.readFileSync(storagePath, 'utf8')
        try {
          data = JSON.parse(fileContent)
        } catch (parseError) {
          console.warn('Failed to parse storage file, creating new one')
          data = {}
        }
      }

      // Update data
      data[key] = value

      // Write back to file
      fs.writeFileSync(storagePath, JSON.stringify(data, null, 2), 'utf8')
    } catch (error) {
      console.error('Error setting storage value:', error)
      throw error
    }
  })

  // Get storage value
  ipcMain.handle('storage:get', async (_event, key: string) => {
    try {
      const storagePath = getStorageFilePath()

      if (!fs.existsSync(storagePath)) {
        return null
      }

      const fileContent = fs.readFileSync(storagePath, 'utf8')

      if (!fileContent || fileContent.trim().length === 0) {
        console.warn(`[storage:get] Empty file for key: ${key}`)
        return null
      }

      try {
        const data = JSON.parse(fileContent)
        const result = data[key] || null
        return result
      } catch (parseError) {
        console.error(`[storage:get] Failed to parse storage file for key: ${key}`, parseError)
        return null
      }
    } catch (error) {
      console.error(`[storage:get] Error getting storage value for key: ${key}`, error)
      return null
    }
  })

  // Remove storage value
  ipcMain.handle('storage:remove', async (_event, key: string) => {
    try {
      const storagePath = getStorageFilePath()

      if (!fs.existsSync(storagePath)) {
        return
      }

      const fileContent = fs.readFileSync(storagePath, 'utf8')
      try {
        const data = JSON.parse(fileContent)
        delete data[key]
        fs.writeFileSync(storagePath, JSON.stringify(data, null, 2), 'utf8')
      } catch (parseError) {
        console.warn('Failed to parse storage file')
      }
    } catch (error) {
      console.error('Error removing storage value:', error)
      throw error
    }
  })
}

// Setup IPC handlers for folder and image operations
function setupFolderHandlers() {
  // Select folder dialog
  ipcMain.handle('folder:select', async () => {
    try {
      const result = await dialog.showOpenDialog({
        properties: ['openDirectory']
      })

      if (result.canceled || result.filePaths.length === 0) {
        return { success: false, canceled: true }
      }

      const folderPath = result.filePaths[0]
      return { success: true, folderPath }
    } catch (error) {
      console.error('[folder:select] Error:', error)
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to select folder'
      }
    }
  })

  // Read folder contents
  ipcMain.handle('folder:read', async (_event, folderPath: string) => {
    try {
      const files = await fs.promises.readdir(folderPath, { withFileTypes: true })

      const fileList = files
        .filter((file) => file.isFile())
        .map((file) => ({
          name: file.name,
          path: path.join(folderPath, file.name)
        }))

      return { success: true, files: fileList }
    } catch (error) {
      console.error('[folder:read] Error:', error)
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to read folder'
      }
    }
  })

  // Convert image format
  ipcMain.handle('image:convert', async (_event, sourcePath: string, targetFormat: string) => {
    try {
      const sharp = require('sharp')
      const ext = path.extname(sourcePath)
      const targetPath = sourcePath.replace(ext, `.${targetFormat}`)

      await sharp(sourcePath).toFormat(targetFormat).toFile(targetPath)

      return { success: true, targetPath }
    } catch (error) {
      console.error('[image:convert] Error:', error)
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to convert image'
      }
    }
  })

  // Batch convert images
  ipcMain.handle('image:batch-convert', async (_event, files: string[], targetFormat: string) => {
    try {
      const sharp = require('sharp')
      const results = []

      for (const sourcePath of files) {
        const ext = path.extname(sourcePath)
        const targetPath = sourcePath.replace(ext, `.${targetFormat}`)

        await sharp(sourcePath).toFormat(targetFormat).toFile(targetPath)
        results.push({ sourcePath, targetPath, success: true })
      }

      return { success: true, results }
    } catch (error) {
      console.error('[image:batch-convert] Error:', error)
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to batch convert images'
      }
    }
  })
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
app.whenReady().then(async () => {
  // Set app user model id for windows
  electronApp.setAppUserModelId('com.electron')

  // Register custom protocols
  registerLocalImageProtocol()
  registerLocalResourceProtocol()

  // Setup IPC handlers
  setupStorageHandlers()
  setupFolderHandlers()
  setupModelHandlers()

  // Default open or close DevTools by F12 in development
  // and ignore CommandOrControl + R in production.
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  createWindow()

  // Hot reload in development
  if (is.dev) {
    app.on('activate', () => {
      if (mainWindow === null) createWindow()
    })

    if (mainWindow) {
      mainWindow.webContents.on('destroyed', () => {
        mainWindow = null
      })
    }
  }

  app.on('activate', function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
