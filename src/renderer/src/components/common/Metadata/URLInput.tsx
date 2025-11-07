import React, { useState, useEffect } from 'react'
import CustomInput from '../CustomInput'
import CustomCombobox from '../CustomCombobox'
import { ExternalLink, Check, X, Globe } from 'lucide-react'
import { URLSubType, URL_SUBTYPE_OPTIONS } from './types'

interface URLInputProps {
  value: string
  onChange: (value: string) => void
  subType: URLSubType
  onSubTypeChange: (subType: URLSubType) => void
  disabled?: boolean
}

export const URLInput: React.FC<URLInputProps> = ({
  value,
  onChange,
  subType,
  onSubTypeChange,
  disabled = false
}) => {
  const [isValidating, setIsValidating] = useState(false)
  const [isValid, setIsValid] = useState<boolean | null>(null)

  const validateURL = async (url: string) => {
    if (!url.trim()) {
      setIsValid(null)
      return
    }

    try {
      new URL(url) // Basic URL validation
      setIsValidating(true)

      // Simple check - in real app, you might want to do actual HTTP request
      const isHttpUrl = url.startsWith('http://') || url.startsWith('https://')
      setIsValid(isHttpUrl)

      setTimeout(() => setIsValidating(false), 500) // Simulate network check
    } catch {
      setIsValid(false)
      setIsValidating(false)
    }
  }

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      validateURL(value)
    }, 500)

    return () => clearTimeout(timeoutId)
  }, [value])

  const getPlaceholderBySubType = (subType: URLSubType): string => {
    const placeholders = {
      video_url: 'https://youtube.com/watch?v=...',
      image_url: 'https://example.com/image.jpg',
      website_url: 'https://example.com',
      api_url: 'https://api.example.com/v1/...',
      custom_url: 'https://example.com'
    }
    return placeholders[subType]
  }

  const getSubTypeIcon = (subType: URLSubType) => {
    const icons = {
      video_url: '🎥',
      image_url: '🖼️',
      website_url: '🌐',
      api_url: '⚡',
      custom_url: '🔗'
    }
    return icons[subType]
  }

  return (
    <div className="space-y-3">
      {/* URL SubType Selection */}
      <CustomCombobox
        label="URL Type"
        value={subType}
        options={URL_SUBTYPE_OPTIONS}
        onChange={(value) => onSubTypeChange(value as URLSubType)}
        placeholder="Select URL type..."
        size="sm"
        disabled={disabled}
      />

      {/* URL Input */}
      <div className="space-y-2">
        <CustomInput
          label={`${getSubTypeIcon(subType)} URL`}
          value={value}
          onChange={onChange}
          placeholder={getPlaceholderBySubType(subType)}
          disabled={disabled}
          size="sm"
          leftIcon={<Globe className="h-3 w-3" />}
          rightIcon={
            <div className="flex items-center gap-1">
              {isValidating && (
                <div className="animate-spin h-3 w-3 border border-blue-500 border-t-transparent rounded-full" />
              )}
              {!isValidating && isValid === true && <Check className="h-3 w-3 text-green-500" />}
              {!isValidating && isValid === false && <X className="h-3 w-3 text-red-500" />}
              {value && isValid === true && (
                <a
                  href={value}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-0.5 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                  title="Open URL in new tab"
                >
                  <ExternalLink className="h-3 w-3 text-blue-500" />
                </a>
              )}
            </div>
          }
        />

        {/* URL Status Messages */}
        {!isValidating && isValid === false && (
          <p className="text-xs text-red-500 flex items-center gap-1">
            <X className="h-3 w-3" />
            Invalid URL format
          </p>
        )}

        {!isValidating && isValid === true && (
          <p className="text-xs text-green-500 flex items-center gap-1">
            <Check className="h-3 w-3" />
            Valid URL
          </p>
        )}

        {/* URL Preview for specific types */}
        {value && isValid === true && (subType === 'image_url' || subType === 'video_url') && (
          <div className="mt-2 p-2 bg-gray-50 dark:bg-gray-800 rounded-md">
            <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Preview:</p>
            {subType === 'image_url' && (
              <div className="max-w-xs">
                <img
                  src={value}
                  alt="URL preview"
                  className="max-w-full h-auto rounded border"
                  onError={(e) => {
                    const target = e.target as HTMLImageElement
                    target.style.display = 'none'
                  }}
                />
              </div>
            )}
            {subType === 'video_url' && (
              <div className="text-xs text-gray-500 italic">Video preview not available</div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
