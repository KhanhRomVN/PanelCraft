import { RouteObject } from 'react-router-dom'
import MainLayout from '../layouts/MainLayout'
import WorkPage from '../pages/Work'

export const routes: RouteObject[] = [
  {
    path: '/',
    element: <MainLayout />,
    children: [
      {
        path: '',
        element: <WorkPage />
      }
    ]
  }
]
