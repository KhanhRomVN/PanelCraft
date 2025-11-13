import { FC, useState, useEffect } from 'react'
import CustomButton from '../../../../../components/common/CustomButton'
import { Play, Settings } from 'lucide-react'
import { useProcessing } from '../../../../../contexts/ProcessingContext'
import { useModels } from '../../../../../contexts/ModelsContext'
import { BackendService } from '../../../../../services/backend.service'
import CustomTable from '../../../../../components/common/CustomTable'
import CustomDropdown from '../../../../../components/common/CustomDropdown'
import SettingsDrawer from './components/SettingsDrawer'

interface Character {
  id: string
  name: string
  personality: string
  notes: string
}

interface Manga {
  id: string
  name: string
  characters: Character[]
}

interface TableRow {
  id: string
  stt: number
  character: string
  original: string
  translate: string
  font: string
  size: number
}

const STORAGE_KEY = 'app-manga-characters'

const ControlPanel: FC = () => {
  const [error, setError] = useState<string>('')
  const [isSettingsDrawerOpen, setIsSettingsDrawerOpen] = useState(false)
  const [mangas, setMangas] = useState<Manga[]>([])
  const [selectedManga, setSelectedManga] = useState<string>('')
  const [tableData, setTableData] = useState<TableRow[]>([])

  const {
    imagePaths,
    isProcessing,
    setIsProcessing,
    setProcessedResults,
    currentImageIndex,
    processedResults
  } = useProcessing()

  const { modelsPath } = useModels()

  const currentOCRResults = processedResults[currentImageIndex]?.ocr_results || []

  useEffect(() => {
    loadMangas()
  }, [])

  useEffect(() => {
    if (currentOCRResults.length > 0) {
      setTableData(
        currentOCRResults.map((ocr, index) => ({
          id: `${ocr.segment_id}-${index}`,
          stt: index + 1,
          character: '',
          original: ocr.original_text,
          translate: '',
          font: 'Arial',
          size: 12
        }))
      )
    }
  }, [currentOCRResults])

  const loadMangas = async () => {
    try {
      if (window.api && window.api.storage) {
        const stored = await window.api.storage.get(STORAGE_KEY)
        if (stored) {
          setMangas(stored)
        }
      }
    } catch (error) {
      console.error('[ControlPanel] Failed to load mangas:', error)
    }
  }

  const handleCharacterChange = (rowId: string, characterId: string) => {
    setTableData((prev) =>
      prev.map((row) => (row.id === rowId ? { ...row, character: characterId } : row))
    )
  }

  const handleTranslateChange = (rowId: string, value: string) => {
    setTableData((prev) =>
      prev.map((row) => (row.id === rowId ? { ...row, translate: value } : row))
    )
  }

  const handleFontChange = (rowId: string, font: string) => {
    setTableData((prev) => prev.map((row) => (row.id === rowId ? { ...row, font } : row)))
  }

  const handleSizeChange = (rowId: string, size: number) => {
    setTableData((prev) => prev.map((row) => (row.id === rowId ? { ...row, size } : row)))
  }

  const selectedMangaData = mangas.find((m) => m.id === selectedManga)
  const characterOptions =
    selectedMangaData?.characters.map((c) => ({
      value: c.id,
      label: c.name
    })) || []

  const handleStartProcessing = async () => {
    if (!modelsPath) {
      setError('Models path not configured')
      return
    }

    if (imagePaths.length === 0) {
      setError('No images to process')
      return
    }

    setIsProcessing(true)
    setError('')

    try {
      const response = await BackendService.processImages(imagePaths, modelsPath)

      if (response.success && response.data?.results) {
        setProcessedResults(response.data.results)
      } else {
        setError(response.error || 'Processing failed')
      }
    } catch (error) {
      console.error('Processing error:', error)
      setError(error instanceof Error ? error.message : 'Unknown error occurred')
    } finally {
      setIsProcessing(false)
    }
  }

  const hasImages = imagePaths.length > 0

  return (
    <div className="h-full bg-card-background flex flex-col items-center justify-center p-8">
      <div className="text-center space-y-6 max-w-md w-full">
        <div className="flex items-center justify-between">
          <h3 className="text-xl font-semibold text-text-primary">Control Panel</h3>
          <CustomButton
            variant="ghost"
            size="sm"
            icon={Settings}
            onClick={() => setIsSettingsDrawerOpen(true)}
            className="w-fit"
            children={undefined}
          />
        </div>

        {hasImages && (
          <div className="space-y-4">
            <p className="text-sm text-text-secondary">
              {imagePaths.length} image{imagePaths.length > 1 ? 's' : ''} loaded
            </p>

            <CustomButton
              variant="primary"
              size="lg"
              icon={Play}
              onClick={handleStartProcessing}
              disabled={isProcessing || !modelsPath}
              className="w-full"
            >
              {isProcessing ? 'Processing...' : 'Start Processing'}
            </CustomButton>

            {error && (
              <div className="p-3 bg-red-100 dark:bg-red-900/20 border border-red-300 dark:border-red-800 rounded-lg">
                <p className="text-sm text-red-700 dark:text-red-400">{error}</p>
              </div>
            )}
          </div>
        )}

        {processedResults.length > 0 && (
          <div className="space-y-4 w-full">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium text-text-primary">OCR Results</h4>
              <span className="text-xs text-text-secondary">
                Image {currentImageIndex + 1} / {processedResults.length}
              </span>
            </div>

            <div className="w-full max-w-full">
              <CustomDropdown
                label="Select Manga"
                value={selectedManga}
                onChange={setSelectedManga}
                options={[
                  { value: '', label: 'No manga selected' },
                  ...mangas.map((m) => ({ value: m.id, label: m.name }))
                ]}
                placeholder="Choose a manga..."
                size="sm"
              />
            </div>

            {currentOCRResults.length > 0 ? (
              <div className="h-[400px] w-full">
                <CustomTable<TableRow>
                  data={tableData}
                  columns={[
                    {
                      accessorKey: 'stt',
                      header: 'STT',
                      cell: (info) => (
                        <span className="text-text-secondary font-medium">{info.getValue()}</span>
                      )
                    },
                    {
                      accessorKey: 'character',
                      header: 'Character',
                      cell: (info) => {
                        const row = info.row.original
                        return (
                          <select
                            value={row.character}
                            onChange={(e) => handleCharacterChange(row.id, e.target.value)}
                            className="w-full px-2 py-1 text-sm bg-input-background border border-border-default rounded focus:outline-none text-text-primary"
                            disabled={!selectedManga}
                          >
                            <option value="">Select...</option>
                            {characterOptions.map((opt) => (
                              <option key={opt.value} value={opt.value}>
                                {opt.label}
                              </option>
                            ))}
                          </select>
                        )
                      }
                    },
                    {
                      accessorKey: 'original',
                      header: 'Original',
                      cell: (info) => (
                        <span className="text-text-primary text-sm">{info.getValue()}</span>
                      )
                    },
                    {
                      accessorKey: 'translate',
                      header: 'Translate',
                      cell: (info) => {
                        const row = info.row.original
                        return (
                          <input
                            type="text"
                            value={row.translate}
                            onChange={(e) => handleTranslateChange(row.id, e.target.value)}
                            className="w-full px-2 py-1 text-sm bg-transparent border-none focus:outline-none text-text-primary"
                            placeholder="Enter translation..."
                          />
                        )
                      }
                    },
                    {
                      accessorKey: 'font',
                      header: 'Font',
                      cell: (info) => {
                        const row = info.row.original
                        return (
                          <select
                            value={row.font}
                            onChange={(e) => handleFontChange(row.id, e.target.value)}
                            className="px-2 py-1 text-sm bg-input-background border border-border-default rounded focus:outline-none text-text-primary"
                          >
                            <option value="Arial">Arial</option>
                            <option value="Times">Times</option>
                            <option value="Courier">Courier</option>
                          </select>
                        )
                      }
                    },
                    {
                      accessorKey: 'size',
                      header: 'Size',
                      cell: (info) => {
                        const row = info.row.original
                        return (
                          <input
                            type="number"
                            value={row.size}
                            onChange={(e) => handleSizeChange(row.id, parseInt(e.target.value))}
                            className="w-16 px-2 py-1 text-sm bg-input-background border border-border-default rounded focus:outline-none text-text-primary"
                          />
                        )
                      }
                    }
                  ]}
                  showHeaderWhenEmpty={true}
                  showFooterWhenEmpty={false}
                  emptyStateHeight="h-48"
                />
              </div>
            ) : (
              <div className="h-48 flex items-center justify-center bg-gray-100 dark:bg-gray-800 rounded-lg">
                <p className="text-sm text-text-secondary text-center">
                  {processedResults[currentImageIndex]?.cleaned_text_result
                    ? 'No text detected'
                    : 'Process images to see OCR results'}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
      <SettingsDrawer
        isOpen={isSettingsDrawerOpen}
        onClose={() => setIsSettingsDrawerOpen(false)}
      />
    </div>
  )
}

export default ControlPanel
