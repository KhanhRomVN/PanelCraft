const SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png']

export const getFileExtension = (filename: string): string => {
  const ext = filename.split('.').pop()?.toLowerCase() || ''
  return ext
}

export const isImageFile = (filename: string): boolean => {
  const ext = getFileExtension(filename)
  return SUPPORTED_FORMATS.includes(ext)
}

export const validateFolder = (
  files: Array<{ name: string; path: string }>
): {
  validImages: Array<{ name: string; path: string; format: string }>
  invalidFiles: string[]
  formats: string[]
} => {
  const validImages: Array<{ name: string; path: string; format: string }> = []
  const invalidFiles: string[] = []
  const formatsSet = new Set<string>()

  files.forEach((file) => {
    if (isImageFile(file.name)) {
      const format = getFileExtension(file.name)
      validImages.push({ ...file, format })
      formatsSet.add(format)
    } else {
      invalidFiles.push(file.name)
    }
  })

  return {
    validImages,
    invalidFiles,
    formats: Array.from(formatsSet)
  }
}
