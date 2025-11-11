import { app } from 'electron'
import { join } from 'path'
import { is } from '@electron-toolkit/utils'
import * as dotenv from 'dotenv'

dotenv.config()

export const getBackendPath = (): string => {
  if (is.dev) {
    // Development: sử dụng backend từ resources trong project Electron
    return join(app.getAppPath(), 'resources', 'backend', 'manga_backend')
  } else {
    // Production: backend nằm trong resources
    return join(process.resourcesPath, 'backend', 'manga_backend')
  }
}

const HOST = process.env.BACKEND_HOST || 'localhost'
const PORT = parseInt(process.env.BACKEND_PORT || '8001', 10)

export const BACKEND_CONFIG = {
  HOST,
  PORT,
  BASE_URL: `http://${HOST}:${PORT}`
}
