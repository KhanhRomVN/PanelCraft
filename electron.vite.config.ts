import { resolve } from 'path'
import { defineConfig, externalizeDepsPlugin } from 'electron-vite'
import { createRequire } from 'module'

const require = createRequire(import.meta.url)
const reactPlugin = require('@vitejs/plugin-react')

export default defineConfig({
  main: {
    plugins: [externalizeDepsPlugin()],
    build: {
      watch: {}
    }
  },
  preload: {
    plugins: [externalizeDepsPlugin()],
    build: {
      watch: {}
    }
  },
  renderer: {
    resolve: {
      alias: {
        '@renderer': resolve('src/renderer/src')
      }
    },
    plugins: [reactPlugin()],
    server: {
      hmr: true,
      watch: {
        usePolling: true
      }
    }
  }
})
