// src/renderer/src/components/common/CustomOverlay.tsx
import React, { useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X } from 'lucide-react'
import { cn } from '../../shared/lib/utils'
import { createPortal } from 'react-dom'

interface CustomOverlayProps {
  isOpen: boolean
  onClose: () => void
  title?: string
  subtitle?: string
  position?: 'left' | 'right' | 'top' | 'bottom'
  width?: string // e.g., '400px', '30%', 'auto'
  height?: string // e.g., '500px', '60%', 'auto'
  children: React.ReactNode
  headerActions?: React.ReactNode
  footerActions?: React.ReactNode
  className?: string
  hideHeader?: boolean
  animationType?: 'slide' | 'scale' | 'fade' | 'bounce'
  showCloseButton?: boolean
  gap?: number // Khoảng cách với màn hình (tailwind units: 1-12)
}

const getContentVariants = (position: string, animationType: string) => {
  const baseVariants = {
    slide: {
      hidden: {
        x: position === 'right' ? '100%' : position === 'left' ? '-100%' : 0,
        y: position === 'top' ? '-100%' : position === 'bottom' ? '100%' : 0,
        opacity: 0
      },
      visible: { x: 0, y: 0, opacity: 1 }
    },
    scale: {
      hidden: { scale: 0.8, opacity: 0 },
      visible: { scale: 1, opacity: 1 }
    },
    fade: {
      hidden: { opacity: 0 },
      visible: { opacity: 1 }
    },
    bounce: {
      hidden: {
        x: position === 'right' ? '100%' : position === 'left' ? '-100%' : 0,
        y: position === 'top' ? '-100%' : position === 'bottom' ? '100%' : 0,
        scale: 0.9
      },
      visible: {
        x: 0,
        y: 0,
        scale: 1,
        transition: {
          type: 'spring',
          damping: 15,
          stiffness: 300
        }
      }
    }
  }

  return baseVariants[animationType] || baseVariants.slide
}

const CustomOverlay: React.FC<CustomOverlayProps> = ({
  isOpen,
  onClose,
  title,
  subtitle,
  position = 'right',
  width = 'auto',
  height = 'auto',
  children,
  headerActions,
  footerActions,
  className,
  hideHeader = false,
  animationType = 'slide',
  showCloseButton = true,
  gap = 4 // Default gap = 4 (1rem)
}) => {
  // Handle ESC key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose()
      }
    }

    document.addEventListener('keydown', handleEscape)
    return () => document.removeEventListener('keydown', handleEscape)
  }, [isOpen, onClose])

  // Convert gap number to rem
  const gapValue = `${gap * 0.25}rem`

  const getContentPosition = () => {
    const baseStyle: React.CSSProperties = {
      position: 'fixed',
      backgroundColor: 'var(--card-background)',
      borderRadius: '0.75rem',
      boxShadow: '0 0 0 1px var(--primary), 0 25px 50px -12px rgba(0, 0, 0, 0.25)',
      display: 'flex',
      flexDirection: 'column',
      border: '1px solid var(--border-default)',
      maxWidth: '100%',
      maxHeight: '100%',
      pointerEvents: 'auto'
    }

    // Calculate dimensions
    const dimensionStyle: React.CSSProperties = {}

    if (position === 'left' || position === 'right') {
      dimensionStyle.width = width
      dimensionStyle.height = height === 'auto' ? `calc(100vh - ${gapValue} * 2)` : height
      dimensionStyle.top = gapValue
      dimensionStyle.bottom = gapValue
    } else {
      dimensionStyle.width = width === 'auto' ? `calc(100vw - ${gapValue} * 2)` : width
      dimensionStyle.height = height
      dimensionStyle.left = gapValue
      dimensionStyle.right = gapValue
    }

    switch (position) {
      case 'right':
        return {
          ...baseStyle,
          ...dimensionStyle,
          right: gapValue
        }
      case 'left':
        return {
          ...baseStyle,
          ...dimensionStyle,
          left: gapValue
        }
      case 'top':
        return {
          ...baseStyle,
          ...dimensionStyle,
          top: gapValue
        }
      case 'bottom':
        return {
          ...baseStyle,
          ...dimensionStyle,
          bottom: gapValue
        }
      default:
        return baseStyle
    }
  }

  const overlayContent = (
    <AnimatePresence mode="wait">
      {isOpen && (
        <>
          {/* Overlay Content - No background overlay, just the content box */}
          <motion.div
            variants={getContentVariants(position, animationType)}
            initial="hidden"
            animate="visible"
            exit="hidden"
            transition={{
              duration: animationType === 'bounce' ? undefined : 0.4,
              ease: animationType === 'bounce' ? undefined : 'easeInOut'
            }}
            style={getContentPosition()}
            className={cn('z-[999]', className)}
          >
            {/* Header */}
            {!hideHeader && (
              <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2, duration: 0.4 }}
                className="flex-shrink-0 px-4 py-3 border-b border-border-default bg-card-background rounded-t-xl"
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    {title && (
                      <h3 className="text-lg font-bold text-text-primary truncate">{title}</h3>
                    )}
                    {subtitle && (
                      <p className="text-sm text-text-secondary mt-1 truncate">{subtitle}</p>
                    )}
                  </div>
                  <div className="flex items-center gap-2 ml-4">
                    {headerActions && (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.3, duration: 0.3 }}
                      >
                        {headerActions}
                      </motion.div>
                    )}
                    {showCloseButton && (
                      <motion.button
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.9 }}
                        transition={{ delay: 0.4, duration: 0.3 }}
                        onClick={onClose}
                        className={cn(
                          'p-2 text-text-secondary hover:text-text-primary hover:bg-sidebar-itemHover rounded-lg',
                          'transition-colors focus:outline-none focus:ring-2 focus:ring-primary/20'
                        )}
                      >
                        <X className="h-5 w-5" />
                      </motion.button>
                    )}
                  </div>
                </div>
              </motion.div>
            )}

            {/* Content - Scrollable */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.5 }}
              className="flex-1 overflow-y-auto overflow-x-hidden"
              style={{ minHeight: 0 }}
            >
              {children}
            </motion.div>

            {/* Footer */}
            {footerActions && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5, duration: 0.4 }}
                className="flex-shrink-0 p-4 border-t border-border-default bg-card-background rounded-b-xl"
              >
                <div className="flex items-center justify-end gap-3">{footerActions}</div>
              </motion.div>
            )}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )

  return createPortal(overlayContent, document.body)
}

export default CustomOverlay
