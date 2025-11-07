// src/renderer/src/components/common/metadata/Metadata.tsx
import React, { useState } from 'react'
import CustomInput from '../CustomInput'
import CustomCombobox from '../CustomCombobox'
import CustomButton from '../CustomButton'
import { Database, Copy, Trash2, Plus, Edit2 } from 'lucide-react'
import { cn } from '../../../shared/lib/utils'

// Import types and utilities
import {
  MetadataProps,
  EditingField,
  EditingExistingField,
  SIZE_CLASSES,
  FIELD_TYPE_OPTIONS,
  MetadataFieldType
} from './types'
import {
  detectFieldType,
  formatValueForType,
  parseValueByType,
  getFieldTypeDisplay,
  copyToClipboard
} from './utils'
import { renderFieldInput } from './MetadataForm'
import CustomCodeEditor from '../CustomCodeEditor'
import CustomTag from '../CustomTag'

// Enhanced color mapping for different field types
const getFieldTypeColor = (fieldType: string) => {
  const colorMap: Record<string, string> = {
    String: 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300',
    Number: 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-300',
    Boolean: 'bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-300',
    Date: 'bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-300',
    Array: 'bg-pink-100 text-pink-800 dark:bg-pink-900/20 dark:text-pink-300',
    Code: 'bg-slate-100 text-slate-800 dark:bg-slate-900/20 dark:text-slate-300',
    URL: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900/20 dark:text-cyan-300',
    Null: 'bg-gray-100 text-gray-600 dark:bg-gray-900/20 dark:text-gray-400'
  }

  return colorMap[fieldType] || 'bg-gray-100 text-gray-600 dark:bg-gray-900/20 dark:text-gray-400'
}

const Metadata: React.FC<MetadataProps> = ({
  metadata = {},
  onMetadataChange,
  onDelete,
  title = 'Metadata',
  className,
  compact = false,
  collapsible = true,
  defaultExpanded = false,
  readOnly = false,
  hideEmpty = false,
  maxVisibleFields,
  allowCreate = true,
  allowDelete = true,
  allowEdit = true,
  size = 'md',
  protectedFields = [],
  shouldRenderField,
  showDeleteButtons = true
}) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)
  const [showAll] = useState(false)
  const [editingField, setEditingField] = useState<EditingField | null>(null)
  const [editingExistingField, setEditingExistingField] = useState<EditingExistingField | null>(
    null
  )

  const canModify = !readOnly && onMetadataChange

  const entries = Object.entries(metadata).filter(([key, value]) => {
    if (hideEmpty) {
      if (value === null || value === undefined || value === '') {
        return false
      }
    }

    if (shouldRenderField) {
      return shouldRenderField(key, value)
    }

    return true
  })

  if (entries.length === 0 && hideEmpty && !allowCreate) {
    return null
  }

  const visibleEntries = maxVisibleFields && !showAll ? entries.slice(0, maxVisibleFields) : entries
  const currentSize = SIZE_CLASSES[size]

  // Start creating new field
  const startCreateField = () => {
    if (!isExpanded) {
      setIsExpanded(true)
    }

    setEditingField({
      key: '',
      value: '',
      type: 'string',
      isNew: true,
      codeLanguage: 'javascript',
      arrayItems: []
    })
  }

  // Start editing existing field
  const startEditField = (originalKey: string, originalValue: any) => {
    if (protectedFields.includes(originalKey)) {
      return
    }

    const currentType = detectFieldType(originalValue)

    setEditingExistingField({
      originalKey,
      newKey: originalKey,
      newValue: formatValueForType(originalValue, currentType),
      newType: currentType,
      codeLanguage: currentType === 'code' ? 'javascript' : undefined,
      arrayItems:
        currentType === 'array' && Array.isArray(originalValue) ? originalValue.map(String) : []
    })
  }

  // Cancel editing
  const cancelEdit = () => {
    setEditingField(null)
    setEditingExistingField(null)
  }

  // Save new field
  const saveNewField = () => {
    if (!editingField || !canModify) return

    const trimmedKey = editingField.key.trim()

    if (!trimmedKey) {
      alert('Field name is required')
      return
    }

    if (metadata.hasOwnProperty(trimmedKey)) {
      alert('Field name already exists')
      return
    }

    // Validation based on type
    if (editingField.type === 'array' && editingField.arrayItems?.length === 0) {
      alert('Array must have at least one item')
      return
    }

    if (
      editingField.type !== 'null' &&
      editingField.type !== 'array' &&
      !editingField.value.trim()
    ) {
      alert('Field value is required')
      return
    }

    try {
      const parsedValue = parseValueByType(editingField.value, editingField.type, {
        arrayItems: editingField.arrayItems,
        codeLanguage: editingField.codeLanguage
      })

      const newMetadata = { ...metadata }
      newMetadata[trimmedKey] = parsedValue

      onMetadataChange(newMetadata)
      setEditingField(null)
    } catch (error) {
      alert(`Invalid value for ${editingField.type}: ${(error as Error).message}`)
    }
  }

  // Save edited field
  const saveEditedField = () => {
    if (!editingExistingField || !canModify) return

    const trimmedKey = editingExistingField.newKey.trim()

    if (!trimmedKey) {
      alert('Field name is required')
      return
    }

    if (trimmedKey !== editingExistingField.originalKey && metadata.hasOwnProperty(trimmedKey)) {
      alert('Field name already exists')
      return
    }

    // Validation based on type
    if (editingExistingField.newType === 'array' && editingExistingField.arrayItems?.length === 0) {
      alert('Array must have at least one item')
      return
    }

    if (
      editingExistingField.newType !== 'null' &&
      editingExistingField.newType !== 'array' &&
      !editingExistingField.newValue.trim()
    ) {
      alert('Field value is required')
      return
    }

    try {
      const parsedValue = parseValueByType(
        editingExistingField.newValue,
        editingExistingField.newType,
        {
          arrayItems: editingExistingField.arrayItems,
          codeLanguage: editingExistingField.codeLanguage
        }
      )

      const newMetadata = { ...metadata }

      if (trimmedKey !== editingExistingField.originalKey) {
        delete newMetadata[editingExistingField.originalKey]
      }

      newMetadata[trimmedKey] = parsedValue

      onMetadataChange(newMetadata)
      setEditingExistingField(null)
    } catch (error) {
      alert(`Invalid value for ${editingExistingField.newType}: ${(error as Error).message}`)
    }
  }

  // Delete field
  const deleteField = (key: string) => {
    if (!canModify) return

    if (protectedFields.includes(key)) {
      console.warn(`Cannot delete protected field: ${key}`)
      return
    }

    if (onDelete) {
      onDelete(key)
      return
    }

    const newMetadata = { ...metadata }
    delete newMetadata[key]
    onMetadataChange?.(newMetadata)
  }

  const renderField = (key: string, value: any) => {
    const displayValue = String(value)
    const isProtected = protectedFields.includes(key)
    const isBeingEdited = editingExistingField?.originalKey === key
    const fieldType = getFieldTypeDisplay(value)

    if (isBeingEdited && editingExistingField) {
      return (
        <div
          key={key}
          className="bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800 p-4 space-y-3"
        >
          <div className="flex items-center gap-2 mb-1">
            <Edit2 className="h-4 w-4 text-blue-600 dark:text-blue-400" />
            <h5 className="font-medium text-blue-900 dark:text-blue-100">Edit Field</h5>
          </div>

          <div className="space-y-3">
            <CustomInput
              label="Field Name"
              value={editingExistingField.newKey}
              onChange={(value) =>
                setEditingExistingField((prev) => (prev ? { ...prev, newKey: value } : null))
              }
              placeholder="Enter field name..."
              variant="filled"
              size="sm"
              required
            />

            <CustomCombobox
              label="Field Type"
              value={editingExistingField.newType}
              options={FIELD_TYPE_OPTIONS}
              onChange={(value) => {
                const newType = value as MetadataFieldType
                setEditingExistingField((prev) => {
                  if (!prev) return null

                  // Luôn reset value khi thay đổi type
                  let resetValue = ''
                  if (newType === 'boolean') {
                    resetValue = 'false'
                  } else if (newType === 'number') {
                    resetValue = '0'
                  } else if (newType === 'date') {
                    resetValue = ''
                  } else if (newType === 'array') {
                    resetValue = ''
                  } else if (newType === 'null') {
                    resetValue = ''
                  }

                  return {
                    ...prev,
                    newType,
                    newValue: resetValue, // Luôn reset value
                    arrayItems: newType === 'array' ? [] : [], // Reset array items
                    codeLanguage: newType === 'code' ? 'javascript' : undefined,
                    urlSubType: newType === 'url' ? prev.urlSubType || 'video_url' : undefined,
                    localFileSubType:
                      newType === 'localfile' ? prev.localFileSubType || 'image' : undefined
                  }
                })
              }}
              placeholder="Select field type..."
              size="sm"
              required
            />

            <div>
              {!['url', 'localfile', 'code'].includes(editingExistingField.newType) && (
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Field Value
                </label>
              )}
              {renderFieldInput(
                editingExistingField.newType,
                editingExistingField.newValue,
                (value) =>
                  setEditingExistingField((prev) => (prev ? { ...prev, newValue: value } : null)),
                {
                  arrayItems: editingExistingField.arrayItems,
                  codeLanguage: editingExistingField.codeLanguage,
                  urlSubType: editingExistingField.urlSubType,
                  localFileSubType: editingExistingField.localFileSubType
                },
                (props) => setEditingExistingField((prev) => (prev ? { ...prev, ...props } : null)),
                false
              )}
            </div>

            <div className="flex items-center gap-2 pt-1">
              <CustomButton variant="primary" size="sm" onClick={saveEditedField}>
                Save Changes
              </CustomButton>
              <CustomButton variant="secondary" size="sm" onClick={cancelEdit}>
                Cancel
              </CustomButton>
            </div>
          </div>
        </div>
      )
    }

    const isCodeField = detectFieldType(value) === 'code'
    const codeLanguage = isCodeField ? 'javascript' : ''

    return (
      <div key={key} className="">
        {/* Field Label */}
        <div className="flex items-center justify-between w-full mb-2">
          <span className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate pr-3">
            {key}
          </span>
          <div className="flex items-center gap-2">
            <span
              className={cn(
                'text-xs px-2.5 py-1 rounded-lg font-medium whitespace-nowrap flex-shrink-0 transition-colors',
                getFieldTypeColor(fieldType)
              )}
            >
              {fieldType}
            </span>

            {/* Action buttons */}
            <div className="flex items-center gap-1">
              {canModify && allowEdit && !isProtected && (
                <button
                  onClick={() => startEditField(key, value)}
                  className="p-1 h-5 w-5 text-gray-500 hover:text-blue-600 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded transition-colors"
                  title="Edit field"
                >
                  <Edit2 className="h-3 w-3" />
                </button>
              )}

              {!isProtected && (
                <button
                  onClick={() => copyToClipboard(displayValue)}
                  className="p-1 h-5 w-5 text-gray-500 hover:text-blue-600 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded transition-colors"
                  title="Copy value"
                >
                  <Copy className="h-3 w-3" />
                </button>
              )}

              {canModify && allowDelete && showDeleteButtons && !isProtected && (
                <button
                  onClick={() => deleteField(key)}
                  className="p-1 h-5 w-5 text-gray-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors"
                  title="Delete field"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Field Value - Different inputs based on field type */}
        {isCodeField ? (
          <CustomCodeEditor
            value={displayValue}
            language={codeLanguage}
            onChange={
              canModify && allowEdit && !isProtected
                ? (newValue) => {
                    const newMetadata = { ...metadata }
                    newMetadata[key] = newValue
                    onMetadataChange?.(newMetadata)
                  }
                : () => {} // Read-only if cannot modify
            }
            onLanguageChange={() => {}} // Language change disabled in display mode
            disabled={!canModify || !allowEdit || isProtected}
          />
        ) : detectFieldType(value) === 'array' && Array.isArray(value) ? (
          <CustomTag
            tags={value.map(String)}
            onTagsChange={
              canModify && allowEdit && !isProtected
                ? (newTags: string[]) => {
                    const newMetadata = { ...metadata }
                    // Try to preserve original data types where possible
                    const convertedItems = newTags.map((item: string) => {
                      // Try to parse as number if original array contained numbers
                      if (value.some((v) => typeof v === 'number') && !isNaN(Number(item))) {
                        return Number(item)
                      }
                      // Try to parse as boolean if original array contained booleans
                      if (value.some((v) => typeof v === 'boolean')) {
                        if (item.toLowerCase() === 'true') return true
                        if (item.toLowerCase() === 'false') return false
                      }
                      // Default to string
                      return item
                    })
                    newMetadata[key] = convertedItems
                    onMetadataChange?.(newMetadata)
                  }
                : () => {} // Read-only if cannot modify
            }
            disabled={!canModify || !allowEdit || isProtected}
            placeholder="Add item..."
            allowDuplicates={true}
            maxTags={50}
            size="sm"
          />
        ) : (
          <CustomInput value={displayValue} readOnly variant="filled" size="sm" />
        )}
      </div>
    )
  }

  const renderNewFieldForm = () => {
    if (!editingField) return null

    return (
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800 p-4 space-y-3">
        <div className="flex items-center gap-2 mb-1">
          <h5 className="font-medium text-blue-900 dark:text-blue-100">Add New Field</h5>
        </div>

        <div className="space-y-3">
          <CustomInput
            label="Field Name"
            value={editingField.key}
            onChange={(value) => setEditingField((prev) => (prev ? { ...prev, key: value } : null))}
            placeholder="Enter field name..."
            variant="filled"
            size="sm"
            required
          />

          <CustomCombobox
            label="Field Type"
            value={editingField.type}
            options={FIELD_TYPE_OPTIONS}
            onChange={(value) => {
              const newType = value as MetadataFieldType
              setEditingField((prev) => {
                if (!prev) return null
                return {
                  ...prev,
                  type: newType,
                  value: newType === 'null' ? '' : prev.value,
                  arrayItems: newType === 'array' ? prev.arrayItems || [] : [],
                  codeLanguage: newType === 'code' ? prev.codeLanguage || 'javascript' : undefined,
                  urlSubType: newType === 'url' ? prev.urlSubType || 'video_url' : undefined,
                  localFileSubType:
                    newType === 'localfile' ? prev.localFileSubType || 'image' : undefined
                }
              })
            }}
            placeholder="Select field type..."
            size="sm"
            required
          />

          <div>
            {!['url', 'localfile', 'code'].includes(editingField.type) && (
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Field Value
              </label>
            )}
            {renderFieldInput(
              editingField.type,
              editingField.value,
              (value) => setEditingField((prev) => (prev ? { ...prev, value } : null)),
              {
                arrayItems: editingField.arrayItems,
                codeLanguage: editingField.codeLanguage,
                urlSubType: editingField.urlSubType,
                localFileSubType: editingField.localFileSubType
              },
              (props) => setEditingField((prev) => (prev ? { ...prev, ...props } : null)),
              false
            )}
          </div>

          <div className="flex items-center gap-2 pt-1">
            <CustomButton variant="primary" size="sm" onClick={saveNewField}>
              Add Field
            </CustomButton>
            <CustomButton variant="secondary" size="sm" onClick={cancelEdit}>
              Cancel
            </CustomButton>
          </div>
        </div>
      </div>
    )
  }

  const headerContent = (
    <div className="flex items-center justify-between">
      <button
        onClick={collapsible ? () => setIsExpanded(!isExpanded) : undefined}
        className={cn(
          'flex items-center gap-2 flex-1',
          collapsible && 'hover:opacity-70 transition-opacity cursor-pointer'
        )}
        disabled={!collapsible}
      >
        <div
          className={cn(
            'bg-indigo-50 dark:bg-indigo-900/20 rounded-lg flex items-center justify-center',
            currentSize.header
          )}
        >
          <Database className={cn('text-indigo-600 dark:text-indigo-400', currentSize.icon)} />
        </div>
        <h4 className={cn('font-semibold text-text-primary', currentSize.title)}>{title}</h4>
        {entries.length > 0 && (
          <span className={cn('text-gray-500 dark:text-gray-400', compact ? 'text-xs' : 'text-sm')}>
            ({entries.length} field{entries.length !== 1 ? 's' : ''})
          </span>
        )}
      </button>

      <div className="flex items-center gap-2">
        {canModify && allowCreate && !editingField && !editingExistingField && (
          <CustomButton
            variant="ghost"
            size="sm"
            icon={Plus}
            onClick={startCreateField}
            title="Add new field"
            className="!w-auto"
            children={undefined}
          ></CustomButton>
        )}
      </div>
    </div>
  )

  const contentSection = (
    <>
      {editingField && renderNewFieldForm()}

      <div className="space-y-3">
        {visibleEntries.map(([key, value]) => renderField(key, value))}
      </div>
    </>
  )

  if (collapsible) {
    return (
      <div className={cn(currentSize.spacing, className)}>
        {headerContent}
        {isExpanded && contentSection}
      </div>
    )
  }

  return (
    <div className={cn(currentSize.spacing, className)}>
      {headerContent}
      {contentSection}
    </div>
  )
}

export default Metadata
