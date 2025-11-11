import React from 'react'
import { LucideIcon } from 'lucide-react'

interface CustomButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'error' | 'warning' | 'success' | 'loading' | 'ghost'
  size?: 'xs' | 'sm' | 'md' | 'lg'
  align?: 'left' | 'center' | 'right'
  children: React.ReactNode
  loading?: boolean
  icon?: LucideIcon
  iconClassName?: string // Custom class for icon styling
  emoji?: React.ReactNode // SVG emoji
}

const CustomButton: React.FC<CustomButtonProps> = ({
  variant = 'primary',
  size = 'md',
  align = 'center',
  children,
  loading = false,
  icon: Icon,
  iconClassName = '',
  emoji,
  className = '',
  disabled,
  ...props
}) => {
  // Kiểm tra xem button có text hay không
  const hasText = children !== undefined && children !== null && children !== ''

  const baseStyles = `
    inline-flex items-center rounded-md font-normal 
    transition-all duration-200 focus:outline-none
    disabled:opacity-50 disabled:cursor-not-allowed
    ${hasText ? 'gap-2' : 'gap-0'}
  `

  const alignStyles = {
    left: 'justify-start text-left',
    center: 'justify-center text-center',
    right: 'justify-end text-right'
  }

  // Size styles khi có text
  const sizeStylesWithText = {
    xs: 'px-2 py-1 text-xs h-[32px]',
    sm: 'px-3 py-2 text-sm h-[38px]',
    md: 'px-4 py-3 text-base h-[48px]',
    lg: 'px-5 py-4 text-lg h-[56px]'
  }

  // Size styles khi chỉ có icon (width = height)
  const sizeStylesIconOnly = {
    xs: 'p-1 text-xs w-[32px] h-[32px]',
    sm: 'p-2 text-sm w-[38px] h-[38px]',
    md: 'p-3 text-base w-[48px] h-[48px]',
    lg: 'p-4 text-lg w-[56px] h-[56px]'
  }

  // Chọn size phù hợp
  const sizeStyles = hasText ? sizeStylesWithText : sizeStylesIconOnly

  // Icon sizes based on button size (có text)
  const iconSizesWithText = {
    xs: 'h-2.5 w-2.5',
    sm: 'h-3 w-3',
    md: 'h-4 w-4',
    lg: 'h-5 w-5'
  }

  // Icon sizes khi chỉ có icon (không có text) - tăng size để vừa button
  const iconSizesIconOnly = {
    xs: 'h-3.5 w-3.5',
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-7 w-7'
  }

  // Chọn size phù hợp
  const iconSizes = hasText ? iconSizesWithText : iconSizesIconOnly

  // Emoji sizes based on button size (có text)
  const emojiSizesWithText = {
    xs: 'h-2.5 w-2.5',
    sm: 'h-3 w-3',
    md: 'h-4 w-4',
    lg: 'h-5 w-5'
  }

  // Emoji sizes khi chỉ có emoji (không có text) - tăng size để vừa button
  const emojiSizesIconOnly = {
    xs: 'h-3.5 w-3.5',
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-7 w-7'
  }

  // Chọn size phù hợp
  const emojiSizes = hasText ? emojiSizesWithText : emojiSizesIconOnly

  const variantStyles = {
    primary: `
      bg-button-bg text-button-bgText border border-button-border
      hover:bg-button-bgHover hover:border-button-borderHover
    `,
    secondary: `
      bg-gray-100 text-gray-700 border border-gray-300
      hover:bg-gray-200 hover:border-gray-400
      dark:bg-gray-700 dark:text-gray-300 dark:border-gray-600
      dark:hover:bg-gray-600 dark:hover:border-gray-500
    `,
    error: `
      bg-red-600 text-white border border-red-600
      hover:bg-red-700 hover:border-red-700
    `,
    warning: `
      bg-yellow-500 text-white border border-yellow-500
      hover:bg-yellow-600 hover:border-yellow-600
    `,
    success: `
      bg-green-600 text-white border border-green-600
      hover:bg-green-700 hover:border-green-700
    `,
    loading: `
      bg-gray-400 text-white border border-gray-400
      cursor-not-allowed
    `,
    ghost: `
      text-text-primary font-normal
      hover:bg-sidebar-itemHover
      active:bg-button-bg active:text-button-bgText
      border-none
    `
  }

  // Override size styles for ghost variant
  const ghostSizeStyles = {
    xs: 'w-full px-2 py-1 text-xs h-[32px]',
    sm: 'w-full px-3 py-2 text-sm h-[38px]',
    md: 'w-full px-4 py-3 text-base h-[48px]',
    lg: 'w-full px-5 py-4 text-lg h-[56px]'
  }

  const LoadingSpinner = () => (
    <svg
      className="animate-spin h-3 w-3"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      ></circle>
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      ></path>
    </svg>
  )

  const currentVariant = loading ? 'loading' : variant
  const isDisabled = disabled || loading || variant === 'loading'

  // Render icon or emoji based on what's provided (mutually exclusive)
  const renderLeftIcon = () => {
    if (loading || variant === 'loading') {
      return <LoadingSpinner />
    }

    // Only render icon if no emoji is provided
    if (Icon && !emoji) {
      return <Icon className={`${iconSizes[size]} ${iconClassName}`.trim()} />
    }

    // Only render emoji if no icon is provided
    if (emoji && !Icon) {
      return <span className={`inline-block ${emojiSizes[size]}`}>{emoji}</span>
    }

    // If both are provided, prioritize icon and ignore emoji
    if (Icon && emoji) {
      console.warn('CustomButton: Both icon and emoji props provided. Only icon will be rendered.')
      return <Icon className={`${iconSizes[size]} ${iconClassName}`.trim()} />
    }

    return null
  }

  return (
    <button
      className={`
        ${baseStyles}
        ${alignStyles[align]}
        ${variant === 'ghost' ? ghostSizeStyles[size] : sizeStyles[size]}
        ${variantStyles[currentVariant]}
        ${className}
      `
        .replace(/\s+/g, ' ')
        .trim()}
      disabled={isDisabled}
      {...props}
    >
      {renderLeftIcon()}
      {children}
    </button>
  )
}

export default CustomButton
