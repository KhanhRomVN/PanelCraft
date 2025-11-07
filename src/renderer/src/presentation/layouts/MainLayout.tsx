import { Outlet } from 'react-router-dom'

const MainLayout = () => {
  return (
    <div className="flex min-h-screen bg-sidebar-background">
      <div className="flex-1 first:flex flex-col h-screen overflow-hidden">
        <div className="flex-1 min-h-0 bg-background rounded-xl overflow-hidden">
          <Outlet />
        </div>
      </div>
    </div>
  )
}

export default MainLayout
