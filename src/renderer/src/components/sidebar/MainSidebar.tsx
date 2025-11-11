import { useState } from 'react'
import { cn } from '../../shared/lib/utils'
import { Button } from '../ui/button'
import { motion } from 'framer-motion'
import { BookOpen, ChevronDown, Database, Home, Settings, FileText, Tag, Clock } from 'lucide-react'
import { useNavigate, useLocation } from 'react-router-dom'

interface MainSidebarProps extends React.HTMLAttributes<HTMLDivElement> {}

interface MenuItem {
  icon: React.ComponentType<{ className?: string }>
  label: string
  href?: string
  description: string
  submenu?: SubMenuItem[]
}

interface SubMenuItem {
  label: string
  href: string
  icon?: React.ComponentType<{ className?: string }>
}

const menuItems: MenuItem[] = [
  {
    icon: Home,
    label: 'Dashboard',
    href: '/',
    description: 'Tổng quan & thống kê'
  },
  { icon: Clock, label: 'Session', href: '/session-manager', description: 'Tổng quan & thống kê' },
  {
    icon: BookOpen,
    label: 'Collection',
    href: '/collections',
    description: 'Quản lý từ vựng'
  },
  {
    icon: Settings,
    label: 'Settings',
    href: '/settings',

    description: 'Cài đặt ứng dụng'
  }
]

const sidebarVariants = {
  hidden: { x: -300, opacity: 0 },
  visible: {
    x: 0,
    opacity: 1,
    transition: {
      type: 'spring' as const,
      stiffness: 100,
      damping: 20
    }
  }
}

const submenuVariants = {
  hidden: { height: 0, opacity: 0 },
  visible: {
    height: 'auto',
    opacity: 1,
    transition: {
      duration: 0.3,
      ease: 'easeInOut'
    }
  }
}

export function MainSidebar({ className }: MainSidebarProps) {
  const navigate = useNavigate()
  const location = useLocation()
  const [expandedItems, setExpandedItems] = useState<string[]>([])

  const toggleSubmenu = (label: string) => {
    setExpandedItems((prev) =>
      prev.includes(label) ? prev.filter((item) => item !== label) : [...prev, label]
    )
  }

  const isItemActive = (href?: string, submenu?: SubMenuItem[]): boolean => {
    if (href) {
      return location.pathname === href || (href !== '/' && location.pathname.startsWith(href))
    }
    if (submenu) {
      return submenu.some((item) => location.pathname === item.href)
    }
    return false
  }

  const isSubitemActive = (href: string): boolean => {
    return location.pathname === href
  }

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={sidebarVariants}
      className={cn('fixed inset-y-0 left-0 w-72', 'flex flex-col p-6', 'bg-background', className)}
    >
      {/* Logo */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-text-primary">PanelCraft</h1>
        <p className="text-sm text-text-secondary mt-1">Management System</p>
      </div>

      {/* Menu Items */}
      <div className="flex-1 space-y-2 overflow-y-auto">
        {menuItems.map((item) => {
          const isActive = isItemActive(item.href, item.submenu)
          const isExpanded = expandedItems.includes(item.label)

          return (
            <div key={item.label}>
              {/* Main Menu Item */}
              <Button
                variant="ghost"
                className={cn(
                  'w-full justify-start gap-3 rounded-xl py-6 text-base transition-all duration-200',
                  'hover:bg-sidebar-item-hover hover:shadow-sm',
                  isActive
                    ? 'bg-primary text-white hover:bg-primary/90 shadow-md'
                    : 'text-text-primary hover:text-text-primary'
                )}
                onClick={() => {
                  if (item.submenu) {
                    toggleSubmenu(item.label)
                  } else if (item.href) {
                    navigate(item.href)
                  }
                }}
              >
                <item.icon
                  className={cn(
                    'h-5 w-5 transition-colors flex-shrink-0',
                    isActive ? 'text-white' : 'text-primary'
                  )}
                />
                <div className="flex flex-col items-start flex-1">
                  <span className="font-medium">{item.label}</span>
                  <span
                    className={cn('text-xs', isActive ? 'text-white/80' : 'text-text-secondary')}
                  >
                    {item.description}
                  </span>
                </div>
                {item.submenu && (
                  <ChevronDown
                    className={cn(
                      'h-5 w-5 transition-transform duration-200 flex-shrink-0',
                      isExpanded ? 'rotate-180' : '',
                      isActive ? 'text-white' : 'text-text-secondary'
                    )}
                  />
                )}
              </Button>

              {/* Submenu Items */}
              {item.submenu && (
                <motion.div
                  initial="hidden"
                  animate={isExpanded ? 'visible' : 'hidden'}
                  variants={submenuVariants}
                  className="overflow-hidden"
                >
                  <div className="pl-4 space-y-1 mt-1">
                    {item.submenu.map((subitem) => {
                      const isSubitemActiveState = isSubitemActive(subitem.href)
                      const SubIcon = subitem.icon

                      return (
                        <Button
                          key={subitem.href}
                          variant="ghost"
                          className={cn(
                            'w-full justify-start gap-3 rounded-lg py-4 text-sm transition-all duration-200',
                            'hover:bg-sidebar-item-hover hover:shadow-sm',
                            isSubitemActiveState
                              ? 'bg-primary/20 text-primary font-medium'
                              : 'text-text-secondary hover:text-text-primary'
                          )}
                          onClick={() => navigate(subitem.href)}
                        >
                          {SubIcon && (
                            <SubIcon
                              className={cn(
                                'h-4 w-4 transition-colors flex-shrink-0',
                                isSubitemActiveState ? 'text-primary' : 'text-text-secondary'
                              )}
                            />
                          )}
                          <span>{subitem.label}</span>
                        </Button>
                      )
                    })}
                  </div>
                </motion.div>
              )}
            </div>
          )
        })}
      </div>
    </motion.div>
  )
}
