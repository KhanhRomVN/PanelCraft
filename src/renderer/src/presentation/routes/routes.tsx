import { RouteObject } from 'react-router-dom'
import MainLayout from '../layouts/MainLayout'
import HomePage from '../pages/Home'
import NotFoundPage from '../pages/Other/NotFoundPage'
import SettingPage from '../pages/Setting'

export const routes: RouteObject[] = [
  {
    path: '/',
    element: <MainLayout />,
    children: [
      {
        path: '',
        element: <HomePage />
      },
      {
        path: 'analytics',
        element: (
          <div className="p-6">
            <h1 className="text-2xl font-bold">Analytics</h1>
          </div>
        )
      },
      {
        path: 'settings',
        element: <SettingPage />
      }
    ]
  },
  {
    path: '*',
    element: <NotFoundPage />
  }
]
