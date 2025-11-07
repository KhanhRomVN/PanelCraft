// src/renderer/src/components/common/metadata/utils.ts
import { MetadataFieldType } from './types'

// Utility functions for field types
export const detectFieldType = (value: any): MetadataFieldType => {
  if (value === null) return 'null'
  if (typeof value === 'boolean') return 'boolean'
  if (typeof value === 'number') return 'number'
  if (Array.isArray(value)) return 'array'
  if (typeof value === 'string') {
    // Check if it's an ISO date
    if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/.test(value)) return 'date'
    // Check if it's a URL
    if (/^https?:\/\/.+/.test(value)) return 'url'
    // Check if it looks like code (contains common programming patterns)
    if (
      /^(function|class|const|let|var|import|export|def|public|private|<!DOCTYPE|<html|<?php|\#include)/m.test(
        value.trim()
      )
    ) {
      return 'code'
    }
  }
  return 'string'
}

export const formatValueForType = (value: any, type: MetadataFieldType): string => {
  switch (type) {
    case 'array':
      return Array.isArray(value) ? JSON.stringify(value) : '[]'
    case 'boolean':
      return String(value)
    case 'number':
      return String(value)
    case 'date':
      return typeof value === 'string' ? value : new Date().toISOString()
    case 'url':
    case 'code':
      return String(value)
    case 'null':
      return ''
    default:
      return String(value || '')
  }
}

export const parseValueByType = (value: string, type: MetadataFieldType, extra?: any): any => {
  switch (type) {
    case 'string':
      return value
    case 'number':
      const num = parseFloat(value)
      return isNaN(num) ? 0 : num
    case 'boolean':
      return value.toLowerCase() === 'true'
    case 'date':
      // Validate ISO date format
      try {
        const date = new Date(value)
        if (isNaN(date.getTime())) {
          throw new Error('Invalid date')
        }
        return date.toISOString()
      } catch {
        throw new Error('Date must be a valid date format')
      }
    case 'array':
      if (extra?.arrayItems) {
        return extra.arrayItems
      }
      try {
        const parsed = JSON.parse(value)
        if (!Array.isArray(parsed)) {
          throw new Error('Value must be a valid JSON array')
        }
        return parsed
      } catch {
        throw new Error('Invalid JSON array format')
      }
    case 'url':
      if (!value.trim()) return ''
      try {
        new URL(value)
        return value
      } catch {
        throw new Error('Must be a valid URL')
      }
    case 'code':
      return value
    case 'null':
      return null
    default:
      return value
  }
}

export const getFieldTypeDisplay = (value: any): string => {
  const type = detectFieldType(value)
  const typeLabels = {
    string: 'Text',
    number: 'Number',
    boolean: 'Bool',
    date: 'Date',
    array: 'Array',
    url: 'URL',
    code: 'Code',
    null: 'Null'
  }
  return typeLabels[type] || 'Text'
}

export const copyToClipboard = async (text: string): Promise<void> => {
  try {
    await navigator.clipboard.writeText(text)
  } catch (err) {
    console.error('Failed to copy:', err)
  }
}
