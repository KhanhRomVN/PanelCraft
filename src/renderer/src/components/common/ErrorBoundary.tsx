import { useRouteError, isRouteErrorResponse, Link } from 'react-router-dom'
import { Button } from '../ui/button'
import { Home } from 'lucide-react'

const ErrorBoundary = () => {
  const error = useRouteError()

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-destructive">Đã xảy ra lỗi</h1>
        <p className="mt-4 text-muted-foreground">
          {isRouteErrorResponse(error)
            ? `Lỗi ${error.status}: ${error.statusText}`
            : 'Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại sau.'}
        </p>
        <Link to="/">
          <Button className="mt-8 gap-2">
            <Home className="h-4 w-4" />
            Trở về trang chủ
          </Button>
        </Link>
      </div>
    </div>
  )
}

export default ErrorBoundary