import { Link } from 'react-router-dom'
import { Button } from '../../../components/ui/button'
import { Home } from 'lucide-react'

const NotFoundPage = () => {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-[#f6f6f0] p-4">
      <div className="relative z-10 text-center">
        <h1 className="text-9xl font-bold text-[#52aaa5]">404</h1>
        <div className="mt-4 space-y-2">
          <h2 className="text-2xl font-semibold text-[#2D3748]">
            Trang không tồn tại
          </h2>
          <p className="text-[#718096]">
            Xin lỗi, trang bạn đang tìm kiếm không tồn tại hoặc đã bị di chuyển.
          </p>
        </div>
        
        <Link to="/">
          <Button 
            className="mt-8 gap-2 bg-[#52aaa5] text-white transition-colors hover:bg-[#52aaa5]/90"
          >
            <Home className="h-4 w-4" />
            Trở về trang chủ
          </Button>
        </Link>
      </div>

      {/* Decorative elements */}
      <div className="absolute inset-0 -z-0 overflow-hidden">
        {/* Large center circle */}
        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
          <div className="h-[400px] w-[400px] rounded-full bg-[#52aaa5]/5" />
        </div>
        
        {/* Top left elements */}
        <div className="absolute left-1/4 top-1/4 -translate-x-1/2 -translate-y-1/2">
          <div className="h-[200px] w-[200px] rounded-full bg-[#52aaa5]/3" />
        </div>
        <div className="absolute left-1/3 top-1/3 -translate-x-1/2 -translate-y-1/2">
          <div className="h-[100px] w-[100px] rounded-2xl bg-[#52aaa5]/4 transform rotate-45" />
        </div>
        
        {/* Bottom right elements */}
        <div className="absolute right-1/4 bottom-1/4 translate-x-1/2 translate-y-1/2">
          <div className="h-[300px] w-[300px] rounded-full bg-[#52aaa5]/4" />
        </div>
        <div className="absolute right-1/3 bottom-1/3 translate-x-1/2 translate-y-1/2">
          <div className="h-[150px] w-[150px] rounded-2xl bg-[#52aaa5]/3 transform -rotate-12" />
        </div>
        
        {/* Additional decorative elements */}
        <div className="absolute left-1/2 bottom-1/4">
          <div className="h-[80px] w-[80px] rounded-full border-2 border-[#52aaa5]/10" />
        </div>
        <div className="absolute right-1/2 top-1/4">
          <div className="h-[60px] w-[60px] rounded-2xl border-2 border-[#52aaa5]/10 transform rotate-45" />
        </div>
        
        {/* Small scattered elements */}
        <div className="absolute left-1/4 top-1/2">
          <div className="h-4 w-4 rounded-full bg-[#52aaa5]/20" />
        </div>
        <div className="absolute right-1/4 bottom-1/2">
          <div className="h-6 w-6 rounded-full bg-[#52aaa5]/15" />
        </div>
        <div className="absolute right-1/3 top-1/3">
          <div className="h-3 w-12 rounded-full bg-[#52aaa5]/10" />
        </div>
        <div className="absolute left-1/3 bottom-1/3">
          <div className="h-12 w-3 rounded-full bg-[#52aaa5]/10" />
        </div>
      </div>
    </div>
  )
}

export default NotFoundPage