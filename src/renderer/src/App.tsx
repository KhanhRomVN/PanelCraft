import { RouterProvider, createHashRouter } from 'react-router-dom'
import { routes } from './presentation/routes'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from './presentation/providers/theme-provider'
import { useState } from 'react'
import ModelDownloadModal from './components/modal/ModelDownloadModal'

function App() {
  const router = createHashRouter(routes)
  const queryClient = new QueryClient()
  const [modelsReady, setModelsReady] = useState(false)

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider
        defaultTheme="light"
        storageKey="updaterCacheDirName: panelcraft-updater
-theme"
      >
        {!modelsReady && <ModelDownloadModal onComplete={() => setModelsReady(true)} />}
        {modelsReady && <RouterProvider router={router} />}
      </ThemeProvider>
    </QueryClientProvider>
  )
}

export default App
