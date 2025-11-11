import React from 'react'
import CustomInput from '../CustomInput'
import CustomButton from '../CustomButton'
import CustomCodeEditor from '../CustomCodeEditor'
import CustomTag from '../CustomTag'
import { URLInput } from './URLInput'
import { LocalFileInput } from './LocalFileInput'
import { cn } from '../../../shared/lib/utils'
import { Check, X, Calendar } from 'lucide-react'
import { MetadataFieldType, URLSubType, LocalFileSubType } from './types'

// Boolean Toggle Component
export const BooleanToggle: React.FC<{
  value: boolean
  onChange: (value: boolean) => void
  disabled?: boolean
}> = ({ value, onChange, disabled }) => (
  <div className="flex items-center gap-2">
    <button
      type="button"
      onClick={() => onChange(true)}
      disabled={disabled}
      className={cn(
        'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
        value
          ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
          : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400 hover:bg-green-50 dark:hover:bg-green-900/10',
        disabled && 'opacity-50 cursor-not-allowed'
      )}
    >
      <Check className="h-3 w-3 mr-1 inline" />
      True
    </button>
    <button
      type="button"
      onClick={() => onChange(false)}
      disabled={disabled}
      className={cn(
        'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
        !value
          ? 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
          : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400 hover:bg-red-50 dark:hover:bg-red-900/10',
        disabled && 'opacity-50 cursor-not-allowed'
      )}
    >
      <X className="h-3 w-3 mr-1 inline" />
      False
    </button>
  </div>
)

// DateTime Picker Component
export const DateTimePicker: React.FC<{
  value: string
  onChange: (value: string) => void
  disabled?: boolean
}> = ({ value, onChange, disabled }) => {
  const formatDateForInput = (isoString: string) => {
    if (!isoString) return ''
    try {
      const date = new Date(isoString)
      const year = date.getFullYear()
      const month = String(date.getMonth() + 1).padStart(2, '0')
      const day = String(date.getDate()).padStart(2, '0')
      const hours = String(date.getHours()).padStart(2, '0')
      const minutes = String(date.getMinutes()).padStart(2, '0')
      return `${year}-${month}-${day}T${hours}:${minutes}`
    } catch {
      return ''
    }
  }

  const handleDateChange = (inputValue: string) => {
    if (!inputValue) {
      onChange('')
      return
    }

    try {
      const date = new Date(inputValue)
      onChange(date.toISOString())
    } catch {
      onChange(inputValue) // Keep invalid value for user to fix
    }
  }

  const setToNow = () => {
    onChange(new Date().toISOString())
  }

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <input
          type="datetime-local"
          value={formatDateForInput(value)}
          onChange={(e) => handleDateChange(e.target.value)}
          disabled={disabled}
          className={cn(
            'flex-1 px-3 py-2 text-sm rounded-lg border',
            'bg-white dark:bg-gray-800',
            'border-gray-200 dark:border-gray-700',
            'focus:border-blue-500 dark:focus:border-blue-400',
            'focus:ring-1 focus:ring-blue-500 dark:focus:ring-blue-400',
            'transition-colors',
            disabled && 'opacity-50 cursor-not-allowed'
          )}
        />
        {!disabled && (
          <CustomButton
            variant="secondary"
            size="sm"
            onClick={setToNow}
            title="Set to current time"
          >
            <Calendar className="h-3 w-3" />
          </CustomButton>
        )}
      </div>
      {value && <p className="text-xs text-gray-500">ISO: {value}</p>}
    </div>
  )
}

// Field Input Renderer
export const renderFieldInput = (
  type: MetadataFieldType,
  value: string,
  onChange: (value: string) => void,
  extraProps: any,
  setExtraProps: (props: any) => void,
  disabled: boolean = false
) => {
  switch (type) {
    case 'boolean':
      return (
        <BooleanToggle
          value={value.toLowerCase() === 'true'}
          onChange={(boolValue) => onChange(String(boolValue))}
          disabled={disabled}
        />
      )

    case 'date':
      return (
        <CustomInput
          type="datetime-local"
          value={value}
          onChange={onChange}
          disabled={disabled}
          size="sm"
          placeholder="Select date and time"
          showTime={true}
        />
      )

    case 'array':
      return (
        <CustomTag
          tags={extraProps.arrayItems || []}
          onTagsChange={(items) => {
            setExtraProps({ ...extraProps, arrayItems: items })
            onChange(JSON.stringify(items))
          }}
          disabled={disabled}
          placeholder="Add array item..."
          allowDuplicates={false}
          maxTags={50}
        />
      )

    case 'url':
      return (
        <URLInput
          value={value}
          onChange={onChange}
          subType={(extraProps.urlSubType as URLSubType) || 'video_url'}
          onSubTypeChange={(subType) => setExtraProps({ ...extraProps, urlSubType: subType })}
          disabled={disabled}
        />
      )

    case 'localfile':
      return (
        <LocalFileInput
          value={value}
          onChange={onChange}
          subType={(extraProps.localFileSubType as LocalFileSubType) || 'image'}
          onSubTypeChange={(subType) => setExtraProps({ ...extraProps, localFileSubType: subType })}
          disabled={disabled}
          emailAddress={extraProps.emailAddress}
        />
      )

    case 'code':
      return (
        <CustomCodeEditor
          value={value}
          language={extraProps.codeLanguage || 'javascript'}
          onChange={onChange}
          onLanguageChange={(lang) => setExtraProps({ ...extraProps, codeLanguage: lang })}
          disabled={disabled}
        />
      )

    case 'null':
      return (
        <div className="text-sm text-gray-500 italic p-3 bg-gray-50 dark:bg-gray-800 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-600 text-center">
          This field will be set to null value
        </div>
      )

    default:
      return (
        <CustomInput
          value={value}
          onChange={onChange}
          placeholder={type === 'number' ? 'Enter number...' : 'Enter text...'}
          type={type === 'number' ? 'number' : 'text'}
          disabled={disabled}
          size="sm"
        />
      )
  }
}
