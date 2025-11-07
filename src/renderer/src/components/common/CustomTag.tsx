// src/renderer/src/components/common/CustomTag.tsx
import React, { useState } from 'react'
import { Button } from '../ui/button'
import { Plus, X, Check } from 'lucide-react'
import { cn } from '../../shared/lib/utils'

interface CustomTagProps {
  tags: string[]
  onTagsChange: (tags: string[]) => void
  className?: string
  disabled?: boolean
  placeholder?: string
  maxTags?: number
  allowDuplicates?: boolean
  onSave?: (tags: string[]) => void
  showSaveButton?: boolean
  size?: 'xs' | 'sm' | 'md' | 'lg'
}

// Predefined Tailwind color classes for tags
const TAG_COLORS = [
  'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/20 dark:text-red-300 dark:border-red-800',
  'bg-teal-100 text-teal-800 border-teal-200 dark:bg-teal-900/20 dark:text-teal-300 dark:border-teal-800',
  'bg-sky-100 text-sky-800 border-sky-200 dark:bg-sky-900/20 dark:text-sky-300 dark:border-sky-800',
  'bg-emerald-100 text-emerald-800 border-emerald-200 dark:bg-emerald-900/20 dark:text-emerald-300 dark:border-emerald-800',
  'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-300 dark:border-yellow-800',
  'bg-purple-100 text-purple-800 border-purple-200 dark:bg-purple-900/20 dark:text-purple-300 dark:border-purple-800',
  'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/20 dark:text-green-300 dark:border-green-800',
  'bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-900/20 dark:text-amber-300 dark:border-amber-800',
  'bg-violet-100 text-violet-800 border-violet-200 dark:bg-violet-900/20 dark:text-violet-300 dark:border-violet-800',
  'bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900/20 dark:text-blue-300 dark:border-blue-800',
  'bg-orange-100 text-orange-800 border-orange-200 dark:bg-orange-900/20 dark:text-orange-300 dark:border-orange-800',
  'bg-lime-100 text-lime-800 border-lime-200 dark:bg-lime-900/20 dark:text-lime-300 dark:border-lime-800',
  'bg-pink-100 text-pink-800 border-pink-200 dark:bg-pink-900/20 dark:text-pink-300 dark:border-pink-800',
  'bg-slate-100 text-slate-800 border-slate-200 dark:bg-slate-900/20 dark:text-slate-300 dark:border-slate-800',
  'bg-indigo-100 text-indigo-800 border-indigo-200 dark:bg-indigo-900/20 dark:text-indigo-300 dark:border-indigo-800',
  'bg-cyan-100 text-cyan-800 border-cyan-200 dark:bg-cyan-900/20 dark:text-cyan-300 dark:border-cyan-800',
  'bg-rose-100 text-rose-800 border-rose-200 dark:bg-rose-900/20 dark:text-rose-300 dark:border-rose-800',
  'bg-fuchsia-100 text-fuchsia-800 border-fuchsia-200 dark:bg-fuchsia-900/20 dark:text-fuchsia-300 dark:border-fuchsia-800',
  'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-900/20 dark:text-gray-300 dark:border-gray-800',
  'bg-stone-100 text-stone-800 border-stone-200 dark:bg-stone-900/20 dark:text-stone-300 dark:border-stone-800'
]

// Function to get consistent color for a tag
const getTagColorClass = (tag: string): string => {
  const index = tag.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % TAG_COLORS.length
  return TAG_COLORS[index]
}

const CustomTag: React.FC<CustomTagProps> = ({
  tags,
  onTagsChange,
  className,
  disabled = false,
  placeholder = 'Enter tag name...',
  maxTags,
  allowDuplicates = false,
  onSave,
  showSaveButton = false,
  size = 'md'
}) => {
  const [hoveredTag, setHoveredTag] = useState<string | null>(null)
  const [showAddTag, setShowAddTag] = useState(false)
  const [newTagValue, setNewTagValue] = useState('')

  const sizeStyles = {
    xs: {
      tag: 'px-2 h-[32px] text-xs',
      input: 'px-2 py-1 text-xs h-[32px]',
      button: 'h-[32px]',
      icon: 'h-2 w-2',
      iconButton: 'h-3 w-3'
    },
    sm: {
      tag: 'px-3 h-[38px] text-sm',
      input: 'px-3 py-2 text-sm h-[38px]',
      button: 'h-[38px]',
      icon: 'h-2.5 w-2.5',
      iconButton: 'h-3 w-3'
    },
    md: {
      tag: 'px-3 h-[48px] text-base',
      input: 'px-3 py-3 text-base h-[48px]',
      button: 'h-[48px]',
      icon: 'h-3 w-3',
      iconButton: 'h-4 w-4'
    },
    lg: {
      tag: 'px-4 h-[56px] text-lg',
      input: 'px-4 py-4 text-lg h-[56px]',
      button: 'h-[56px]',
      icon: 'h-4 w-4',
      iconButton: 'h-5 w-5'
    }
  }

  const canAddMore = !maxTags || (tags?.length || 0) < maxTags

  const handleAddTag = () => {
    const trimmedValue = newTagValue.trim()
    if (trimmedValue && (allowDuplicates || !(tags || []).includes(trimmedValue))) {
      const updatedTags = [...(tags || []), trimmedValue]
      onTagsChange(updatedTags)
      setNewTagValue('')
      setShowAddTag(false)
    }
  }

  const handleRemoveTag = (tagToRemove: string) => {
    const updatedTags = tags.filter((tag) => tag !== tagToRemove)
    onTagsChange(updatedTags)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleAddTag()
    } else if (e.key === 'Escape') {
      setShowAddTag(false)
      setNewTagValue('')
    }
  }

  const handleSave = () => {
    if (onSave) {
      onSave(tags)
    }
  }

  return (
    <div className={cn('space-y-3', className)}>
      {/* Tags Container */}
      <div className="flex flex-wrap gap-2 items-center">
        {/* Existing Tags */}
        {(tags?.length || 0) > 0 &&
          tags?.map((tag, index) => {
            const colorClasses = getTagColorClass(tag)

            return (
              <div
                key={index}
                className="relative group"
                onMouseEnter={() => !disabled && setHoveredTag(tag)}
                onMouseLeave={() => setHoveredTag(null)}
              >
                <div
                  className={cn(
                    'rounded-lg font-medium border transition-all duration-200 cursor-default inline-flex items-center',
                    sizeStyles[size].tag,
                    colorClasses
                  )}
                >
                  <span className="inline-flex items-center gap-2">
                    {tag}

                    {/* Delete Button - appears on hover */}
                    {!disabled && hoveredTag === tag && (
                      <button
                        onClick={() => handleRemoveTag(tag)}
                        className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-black/20 dark:bg-white/20 hover:bg-black/30 dark:hover:bg-white/30 transition-colors"
                      >
                        <X className={sizeStyles[size].icon} />
                      </button>
                    )}
                  </span>
                </div>
              </div>
            )
          })}

        {/* Add Tag Button/Input */}
        {!disabled && canAddMore && (
          <>
            {!showAddTag ? (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowAddTag(true)}
                className={cn(
                  'border-dashed border-gray-300 hover:border-gray-400 hover:bg-gray-50 dark:border-gray-600 dark:hover:border-gray-500 dark:hover:bg-gray-800 text-gray-600 dark:text-gray-400',
                  sizeStyles[size].button
                )}
              >
                <Plus className={`${sizeStyles[size].iconButton} mr-1`} />
                Add Tag
              </Button>
            ) : (
              <div className="flex items-center gap-2">
                <div className="relative">
                  <input
                    type="text"
                    value={newTagValue}
                    onChange={(e) => setNewTagValue(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={placeholder}
                    autoFocus
                    className={cn(
                      'bg-input-background border border-border-default rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 min-w-[120px] text-text-primary placeholder:text-gray-500',
                      sizeStyles[size].input
                    )}
                  />
                </div>

                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleAddTag}
                    disabled={
                      !newTagValue.trim() ||
                      (!allowDuplicates && (tags || []).includes(newTagValue.trim()))
                    }
                    className={cn(
                      'w-7 p-0 text-green-600 hover:text-green-700 hover:bg-green-50 dark:text-green-400 dark:hover:text-green-300 dark:hover:bg-green-900/20',
                      sizeStyles[size].button
                    )}
                  >
                    <Check className={sizeStyles[size].iconButton} />
                  </Button>

                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setShowAddTag(false)
                      setNewTagValue('')
                    }}
                    className={cn(
                      'w-7 p-0 text-gray-400 hover:text-gray-600 hover:bg-gray-50 dark:text-gray-500 dark:hover:text-gray-300 dark:hover:bg-gray-800',
                      sizeStyles[size].button
                    )}
                  >
                    <X className={sizeStyles[size].iconButton} />
                  </Button>
                </div>
              </div>
            )}
          </>
        )}

        {/* No tags message */}
        {(tags?.length || 0) === 0 && !showAddTag && (
          <span className="text-sm text-text-secondary italic">No tags assigned</span>
        )}

        {/* Save Button */}
        {showSaveButton && onSave && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleSave}
            className={cn(
              'text-green-600 hover:text-green-700 hover:bg-green-50 dark:text-green-400 dark:hover:text-green-300 dark:hover:bg-green-900/20 ml-2',
              sizeStyles[size].button,
              sizeStyles[size].tag.includes('text-xs')
                ? 'text-xs'
                : sizeStyles[size].tag.includes('text-sm')
                  ? 'text-sm'
                  : sizeStyles[size].tag.includes('text-lg')
                    ? 'text-lg'
                    : 'text-base'
            )}
          >
            <Check className={`${sizeStyles[size].iconButton} mr-1`} />
            Save Tags
          </Button>
        )}
      </div>

      {/* Helper Text */}
      {showAddTag && (
        <p className="text-xs text-text-secondary">Press Enter to add tag, Escape to cancel</p>
      )}

      {/* Tag Limits */}
      {maxTags && (
        <p className="text-xs text-text-secondary">
          {tags.length} / {maxTags} tags
          {!canAddMore && ' (maximum reached)'}
        </p>
      )}
    </div>
  )
}

export default CustomTag
