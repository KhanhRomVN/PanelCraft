// Field type definitions
export type MetadataFieldType =
  | 'string'
  | 'number'
  | 'boolean'
  | 'date'
  | 'array'
  | 'url'
  | 'localfile'
  | 'code'
  | 'null'

export type URLSubType = 'video_url' | 'image_url' | 'website_url' | 'api_url' | 'custom_url'
export type LocalFileSubType = 'image' | 'video' | 'audio' | 'document' | 'archive' | 'custom_file'

export interface MetadataFieldInfo {
  value: any
  type: MetadataFieldType
}

export interface MetadataProps {
  metadata: Record<string, any>
  onMetadataChange?: (metadata: Record<string, any>) => void
  onDelete?: (key: string) => void
  title?: string
  className?: string
  compact?: boolean
  collapsible?: boolean
  defaultExpanded?: boolean
  readOnly?: boolean
  hideEmpty?: boolean
  maxVisibleFields?: number
  allowCreate?: boolean
  allowEdit?: boolean
  allowDelete?: boolean
  size?: 'sm' | 'md' | 'lg'
  protectedFields?: string[]
  shouldRenderField?: (key: string, value: any) => boolean
  editable?: boolean
  showDeleteButtons?: boolean
  emailAddress?: string // Added for LocalFile component
}

export interface EditingField {
  key: string
  value: string
  type: MetadataFieldType
  isNew: boolean
  codeLanguage?: string
  arrayItems?: string[]
  urlSubType?: URLSubType
  localFileSubType?: LocalFileSubType
}

export interface EditingExistingField {
  originalKey: string
  newKey: string
  newValue: string
  newType: MetadataFieldType
  codeLanguage?: string
  arrayItems?: string[]
  urlSubType?: URLSubType
  localFileSubType?: LocalFileSubType
}

// Field type options for combobox
export const FIELD_TYPE_OPTIONS = [
  { value: 'string', label: 'String (Text)' },
  { value: 'number', label: 'Number' },
  { value: 'boolean', label: 'Boolean (True/False)' },
  { value: 'date', label: 'Date & Time' },
  { value: 'array', label: 'Array (List)' },
  { value: 'url', label: 'URL (Web Link)' },
  { value: 'localfile', label: 'Local File' },
  { value: 'code', label: 'Code (Programming)' },
  { value: 'null', label: 'Null Value' }
]

export const URL_SUBTYPE_OPTIONS = [
  { value: 'video_url', label: 'Video URL' },
  { value: 'image_url', label: 'Image URL' },
  { value: 'website_url', label: 'Website URL' },
  { value: 'custom_url', label: 'Custom URL' }
]

export const LOCALFILE_SUBTYPE_OPTIONS = [
  { value: 'image', label: 'Image File' },
  { value: 'video', label: 'Video File' },
  { value: 'audio', label: 'Audio File' },
  { value: 'document', label: 'Document File' },
  { value: 'archive', label: 'Archive File' },
  { value: 'custom_file', label: 'Custom File' }
]

// Programming languages for code fields
export const CODE_LANGUAGES = [
  'javascript',
  'typescript',
  'python',
  'java',
  'csharp',
  'cpp',
  'c',
  'php',
  'ruby',
  'go',
  'rust',
  'swift',
  'kotlin',
  'dart',
  'html',
  'css',
  'scss',
  'json',
  'xml',
  'yaml',
  'markdown',
  'sql',
  'bash',
  'powershell',
  'plaintext'
]

// Size styles configuration
export const SIZE_CLASSES = {
  sm: {
    header: 'w-5 h-5',
    icon: 'h-3 w-3',
    title: 'text-sm',
    spacing: 'space-y-2',
    gap: 'gap-2',
    button: 'h-8'
  },
  md: {
    header: 'w-8 h-8',
    icon: 'h-4 w-4',
    title: 'text-base',
    spacing: 'space-y-4',
    gap: 'gap-3',
    button: 'h-10'
  },
  lg: {
    header: 'w-10 h-10',
    icon: 'h-5 w-5',
    title: 'text-lg',
    spacing: 'space-y-6',
    gap: 'gap-4',
    button: 'h-12'
  }
}

// File extensions mapping
export const FILE_EXTENSIONS = {
  image: ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp'],
  video: ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv'],
  audio: ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma'],
  document: ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'],
  archive: ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2']
}

// Helper function to get file extension color
export const getFileExtensionColor = (extension: string): string => {
  const colorMap: Record<string, string> = {
    // Images
    '.jpg': 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300',
    '.jpeg': 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300',
    '.png': 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300',
    '.gif': 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300',
    '.webp': 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300',
    '.svg': 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300',

    // Videos
    '.mp4': 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300',
    '.avi': 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300',
    '.mov': 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300',
    '.webm': 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300',

    // Audio
    '.mp3': 'bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-300',
    '.wav': 'bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-300',
    '.flac': 'bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-300',

    // Documents
    '.pdf': 'bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-300',
    '.doc': 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-300',
    '.docx': 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-300',
    '.txt': 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-300'
  }

  return (
    colorMap[extension.toLowerCase()] ||
    'bg-gray-100 text-gray-600 dark:bg-gray-900/20 dark:text-gray-400'
  )
}
