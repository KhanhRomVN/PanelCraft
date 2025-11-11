import * as dotenv from 'dotenv'
dotenv.config()
import { app, shell, BrowserWindow, ipcMain, dialog } from 'electron'
import * as fs from 'fs'
import * as path from 'path'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { spawn, ChildProcess } from 'child_process'
import { getBackendPath, BACKEND_CONFIG } from './config/backend.config'
import fetch from 'node-fetch'

let mainWindow: BrowserWindow | null = null
let backendProcess: ChildProcess | null = null

// Khởi động backend server
function startBackend(): Promise<void> {
  return new Promise((resolve, reject) => {
    const backendPath = getBackendPath()

    console.log('[Backend] Starting backend at:', backendPath)

    try {
      backendProcess = spawn(backendPath, [], {
        env: {
          ...process.env,
          HOST: BACKEND_CONFIG.HOST,
          PORT: BACKEND_CONFIG.PORT.toString()
        }
      })

      backendProcess.stdout?.on('data', (data) => {
        console.log('[Backend]', data.toString())
      })

      backendProcess.stderr?.on('data', (data) => {
        console.error('[Backend Error]', data.toString())
      })

      backendProcess.on('error', (error) => {
        console.error('[Backend] Failed to start:', error)
        reject(error)
      })

      backendProcess.on('close', (code) => {
        console.log('[Backend] Process exited with code:', code)
        backendProcess = null
      })

      // Đợi backend khởi động và kiểm tra health
      let retries = 0
      const maxRetries = 30 // 30 seconds
      const checkBackend = setInterval(async () => {
        try {
          const healthUrl = `http://${BACKEND_CONFIG.HOST}:${BACKEND_CONFIG.PORT}/health`
          const response = await fetch(healthUrl)
          if (response.ok) {
            clearInterval(checkBackend)
            console.log('[Backend] Started successfully')
            resolve()
          }
        } catch (error) {
          retries++
          if (retries >= maxRetries) {
            clearInterval(checkBackend)
            console.error('[Backend] Failed to start after 30s')
            reject(new Error('Backend startup timeout'))
          }
        }
      }, 1000)
    } catch (error) {
      console.error('[Backend] Spawn error:', error)
      reject(error)
    }
  })
}

// Dừng backend server
function stopBackend(): void {
  if (backendProcess) {
    console.log('[Backend] Stopping backend...')
    backendProcess.kill()
    backendProcess = null
  }
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
      contextIsolation: true,
      webSecurity: false // Tắt web security để load resource từ localhost
    }
  })

  // Thêm CSP header để cho phép load resource từ localhost
  mainWindow.webContents.session.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob: http://localhost:* ws://localhost:*; " +
            "img-src 'self' data: https: blob: http://localhost:*; " +
            "media-src 'self' data: https: blob: http://localhost:*; " +
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; " +
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

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
app.whenReady().then(async () => {
  // Set app user model id for windows
  electronApp.setAppUserModelId('com.electron')

  // Default open or close DevTools by F12 in development
  // and ignore CommandOrControl + R in production.
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  // Khởi động backend trước khi tạo window
  // TEMPORARY DISABLED: Backend startup
  // try {
  //   await startBackend()
  // } catch (error) {
  //   console.error('[App] Failed to start backend:', error)
  // }

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

app.on('window-all-closed', async () => {
  stopBackend()
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

// IPC Handlers
ipcMain.handle('dialog:openFolder', async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    properties: ['openDirectory']
  })
  return result.canceled ? null : result.filePaths[0]
})

ipcMain.handle('folder:readImages', async (_, folderPath: string) => {
  try {
    const files = fs.readdirSync(folderPath)
    const validExtensions = ['.jpg', '.jpeg', '.png']

    const valid: string[] = []
    const invalid: string[] = []

    for (const file of files) {
      const fullPath = path.join(folderPath, file)
      const stat = fs.statSync(fullPath)

      if (stat.isDirectory()) {
        invalid.push(file)
        continue
      }

      const ext = path.extname(file).toLowerCase()
      if (validExtensions.includes(ext)) {
        valid.push(fullPath)
      } else {
        invalid.push(file)
      }
    }

    return { valid, invalid }
  } catch (error) {
    console.error('Failed to read folder:', error)
    throw error
  }
})

ipcMain.handle('image:read', async (_, imagePath: string) => {
  try {
    const imageBuffer = fs.readFileSync(imagePath)
    const ext = path.extname(imagePath).toLowerCase()
    let mimeType = 'image/jpeg'

    if (ext === '.png') mimeType = 'image/png'
    else if (ext === '.gif') mimeType = 'image/gif'
    else if (ext === '.webp') mimeType = 'image/webp'
    else if (ext === '.bmp') mimeType = 'image/bmp'

    const base64 = imageBuffer.toString('base64')
    return `data:${mimeType};base64,${base64}`
  } catch (error) {
    console.error('Failed to read image:', error)
    throw error
  }
})

// IPC Handlers for Models Management
ipcMain.handle('models:checkFolder', async (_, folderPath: string) => {
  try {
    const textDetectionPath = path.join(folderPath, 'text_detection')
    const ocrPath = path.join(folderPath, 'ocr')
    const segmentationPath = path.join(folderPath, 'segmentation')

    // Kiểm tra các folder con
    const hasTextDetection =
      fs.existsSync(textDetectionPath) && fs.statSync(textDetectionPath).isDirectory()
    const hasOcr = fs.existsSync(ocrPath) && fs.statSync(ocrPath).isDirectory()
    const hasSegmentation =
      fs.existsSync(segmentationPath) && fs.statSync(segmentationPath).isDirectory()

    const isValid = hasTextDetection && hasOcr && hasSegmentation

    return {
      isValid,
      folders: {
        text_detection: hasTextDetection,
        ocr: hasOcr,
        segmentation: hasSegmentation
      }
    }
  } catch (error) {
    console.error('Failed to check models folder:', error)
    return {
      isValid: false,
      folders: {
        text_detection: false,
        ocr: false,
        segmentation: false
      }
    }
  }
})

ipcMain.handle('models:createFolder', async (_, folderPath: string) => {
  try {
    // Tạo folder chính
    if (!fs.existsSync(folderPath)) {
      fs.mkdirSync(folderPath, { recursive: true })
    }

    // Tạo các folder con
    const textDetectionPath = path.join(folderPath, 'text_detection')
    const ocrPath = path.join(folderPath, 'ocr')
    const segmentationPath = path.join(folderPath, 'segmentation')

    fs.mkdirSync(textDetectionPath, { recursive: true })
    fs.mkdirSync(ocrPath, { recursive: true })
    fs.mkdirSync(segmentationPath, { recursive: true })

    return { success: true }
  } catch (error) {
    console.error('Failed to create models folder:', error)
    throw error
  }
})

ipcMain.handle('models:downloadFile', async (_, fileUrl: string, destPath: string) => {
  try {
    // Tạo folder nếu chưa tồn tại
    const dir = path.dirname(destPath)
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true })
    }

    const response = await fetch(fileUrl, {
      method: 'GET',
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
      },
      redirect: 'follow'
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status} for URL: ${fileUrl}`)
    }

    const buffer = await response.arrayBuffer()

    fs.writeFileSync(destPath, Buffer.from(buffer))

    return { success: true }
  } catch (error) {
    throw error
  }
})

ipcMain.handle('models:checkFileExists', async (_, filePath: string) => {
  try {
    return fs.existsSync(filePath)
  } catch (error) {
    return false
  }
})

// Cleanup khi app quit
app.on('before-quit', () => {
  stopBackend()
})

ipcMain.handle('image:fetchFromBackend', async (_, imageUrl: string) => {
  try {
    const response = await fetch(imageUrl)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    const arrayBuffer = await response.arrayBuffer()
    const buffer = Buffer.from(arrayBuffer)
    const base64 = buffer.toString('base64')

    // Xác định mime type từ URL
    const ext = imageUrl.split('.').pop()?.toLowerCase()
    let mimeType = 'image/jpeg'
    if (ext === 'png') mimeType = 'image/png'
    else if (ext === 'gif') mimeType = 'image/gif'
    else if (ext === 'webp') mimeType = 'image/webp'

    return `data:${mimeType};base64,${base64}`
  } catch (error) {
    console.error('Failed to fetch image from backend:', error)
    throw error
  }
})
