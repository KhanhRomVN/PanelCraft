import { RouterProvider, createHashRouter } from 'react-router-dom'
import { routes } from './presentation/routes'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from './presentation/providers/theme-provider'
import { ProcessingProvider } from './contexts/ProcessingContext'
import { ModelsProvider } from './contexts/ModelsContext'
import ModelsSetupGuard from './components/guards/ModelsSetupGuard'

function App() {
  const router = createHashRouter(routes)
  const queryClient = new QueryClient()

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="light" storageKey="panelcraft-theme">
        <ModelsProvider>
          <ProcessingProvider>
            <ModelsSetupGuard>
              <RouterProvider router={router} />
            </ModelsSetupGuard>
          </ProcessingProvider>
        </ModelsProvider>
      </ThemeProvider>
    </QueryClientProvider>
  )
}

export default App
