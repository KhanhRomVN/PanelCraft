import React, { useState, useEffect } from 'react'
import CustomInput from '../CustomInput'
import CustomCombobox from '../CustomCombobox'
import { Folder, Check, X, AlertCircle } from 'lucide-react'
import { cn } from '../../../shared/lib/utils'
import {
  LocalFileSubType,
  LOCALFILE_SUBTYPE_OPTIONS,
  FILE_EXTENSIONS,
  getFileExtensionColor
} from './types'

interface LocalFileInputProps {
  value: string
  onChange: (value: string) => void
  subType: LocalFileSubType
  onSubTypeChange: (subType: LocalFileSubType) => void
  disabled?: boolean
  emailAddress?: string
}

export const LocalFileInput: React.FC<LocalFileInputProps> = ({
  value,
  onChange,
  subType,
  onSubTypeChange,
  disabled = false,
  emailAddress
}) => {
  const [isValid, setIsValid] = useState<boolean | null>(null)
  const [fileExtension, setFileExtension] = useState<string>('')
  const [validationMessage, setValidationMessage] = useState<string>('')

  const validateLocalFile = (filePath: string) => {
    if (!filePath.trim()) {
      setIsValid(null)
      setValidationMessage('')
      setFileExtension('')
      return
    }

    // Check if path starts with relative path indicators
    const isValidPath =
      filePath.startsWith('./') || filePath.startsWith('../') || filePath.startsWith('/')

    if (!isValidPath) {
      setIsValid(false)
      setValidationMessage('Path must be relative (start with ./ or ../)')
      setFileExtension('')
      return
    }

    // Extract file extension
    const extensionMatch = filePath.match(/\.([a-zA-Z0-9]+)$/)
    const extension = extensionMatch ? `.${extensionMatch[1].toLowerCase()}` : ''
    setFileExtension(extension)

    // Validate against subType
    if (subType !== 'custom_file' && extension) {
      const allowedExtensions = FILE_EXTENSIONS[subType as keyof typeof FILE_EXTENSIONS] || []
      if (!allowedExtensions.includes(extension)) {
        setIsValid(false)
        setValidationMessage(
          `Invalid extension for ${subType}. Expected: ${allowedExtensions.join(', ')}`
        )
        return
      }
    }

    setIsValid(true)
    setValidationMessage('Valid file path')
  }

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      validateLocalFile(value)
    }, 300)

    return () => clearTimeout(timeoutId)
  }, [value, subType])

  const getPlaceholderBySubType = (subType: LocalFileSubType): string => {
    const placeholders = {
      image: './assets/images/profile.jpg',
      video: './videos/demo.mp4',
      audio: './audio/music.mp3',
      document: './documents/report.pdf',
      archive: './backups/data.zip',
      custom_file: './path/to/file.ext'
    }
    return placeholders[subType]
  }

  const getSubTypeIcon = (subType: LocalFileSubType) => {
    const icons = {
      image: '🖼️',
      video: '🎥',
      audio: '🎵',
      document: '📄',
      archive: '📦',
      custom_file: '📁'
    }
    return icons[subType]
  }

  const generateSuggestedPath = () => {
    if (!emailAddress) return ''

    const emailPrefix = emailAddress.split('@')[0]
    const pathMap = {
      image: `./data/${emailPrefix}/images/`,
      video: `./data/${emailPrefix}/videos/`,
      audio: `./data/${emailPrefix}/audio/`,
      document: `./data/${emailPrefix}/documents/`,
      archive: `./data/${emailPrefix}/archives/`,
      custom_file: `./data/${emailPrefix}/files/`
    }

    return pathMap[subType]
  }

  return (
    <div className="space-y-3">
      {/* File SubType Selection */}
      <CustomCombobox
        label="File Type"
        value={subType}
        options={LOCALFILE_SUBTYPE_OPTIONS}
        onChange={(value) => onSubTypeChange(value as LocalFileSubType)}
        placeholder="Select file type..."
        size="sm"
        disabled={disabled}
      />

      {/* File Path Input */}
      <div className="space-y-2">
        <CustomInput
          label={`${getSubTypeIcon(subType)} File Path`}
          value={value}
          onChange={onChange}
          placeholder={getPlaceholderBySubType(subType)}
          disabled={disabled}
          size="sm"
          leftIcon={<Folder className="h-3 w-3" />}
          rightIcon={
            <div className="flex items-center gap-1">
              {isValid === true && <Check className="h-3 w-3 text-green-500" />}
              {isValid === false && <X className="h-3 w-3 text-red-500" />}
            </div>
          }
        />

        {/* File Extension Badge */}
        {fileExtension && (
          <div className="flex items-center gap-2">
            <span
              className={cn(
                'inline-flex items-center px-2 py-1 rounded text-xs font-medium',
                getFileExtensionColor(fileExtension)
              )}
            >
              {fileExtension.toUpperCase()}
            </span>
          </div>
        )}

        {/* Status Messages */}
        {validationMessage && (
          <p
            className={cn(
              'text-xs flex items-center gap-1',
              isValid === true ? 'text-green-500' : 'text-red-500'
            )}
          >
            {isValid === true ? <Check className="h-3 w-3" /> : <AlertCircle className="h-3 w-3" />}
            {validationMessage}
          </p>
        )}

        {/* Suggested Path Helper */}
        {emailAddress && !value && (
          <div className="p-2 bg-blue-50 dark:bg-blue-900/20 rounded-md border border-blue-200 dark:border-blue-700">
            <p className="text-xs text-blue-600 dark:text-blue-400 mb-1">Suggested path:</p>
            <button
              onClick={() => onChange(generateSuggestedPath())}
              className="text-xs text-blue-700 dark:text-blue-300 hover:underline"
              disabled={disabled}
              type="button"
            >
              {generateSuggestedPath()}
            </button>
          </div>
        )}

        {/* File Type Guidelines */}
        {subType !== 'custom_file' && (
          <div className="text-xs text-gray-500 dark:text-gray-400">
            <p className="font-medium mb-1">Supported extensions:</p>
            <div className="flex flex-wrap gap-1">
              {(FILE_EXTENSIONS[subType as keyof typeof FILE_EXTENSIONS] || []).map((ext) => (
                <span
                  key={ext}
                  className={cn('px-1.5 py-0.5 rounded text-xs', getFileExtensionColor(ext))}
                >
                  {ext}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
