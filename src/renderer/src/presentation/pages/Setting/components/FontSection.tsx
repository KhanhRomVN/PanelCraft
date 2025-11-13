import { useState, useEffect } from 'react'
import { Plus, Trash2, Star } from 'lucide-react'
import CustomButton from '../../../../components/common/CustomButton'
import CustomCombobox from '../../../../components/common/CustomCombobox'

interface FontConfig {
  name: string
  isDefault: boolean
}

const STORAGE_KEY = 'app-fonts-config'

const FontSection = () => {
  const [fonts, setFonts] = useState<FontConfig[]>([])
  const [systemFonts, setSystemFonts] = useState<string[]>([])
  const [isLoadingFonts, setIsLoadingFonts] = useState(true)
  const [filterText, setFilterText] = useState('')
  const [selectedFont, setSelectedFont] = useState('')

  useEffect(() => {
    const loadSystemFonts = async () => {
      try {
        const fonts = await window.electronAPI.getSystemFonts()
        setSystemFonts(fonts)
      } catch (error) {
        console.error('Failed to load system fonts:', error)
        setSystemFonts(['Arial', 'Times New Roman', 'Courier New', 'Georgia', 'Verdana'])
      } finally {
        setIsLoadingFonts(false)
      }
    }

    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      setFonts(JSON.parse(stored))
    } else {
      const defaultFonts: FontConfig[] = [
        { name: 'Arial', isDefault: true },
        { name: 'Times New Roman', isDefault: false }
      ]
      setFonts(defaultFonts)
      localStorage.setItem(STORAGE_KEY, JSON.stringify(defaultFonts))
    }

    loadSystemFonts()
  }, [])

  const saveFonts = (newFonts: FontConfig[]) => {
    setFonts(newFonts)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newFonts))
  }

  const handleAddFont = () => {
    if (!selectedFont || fonts.some((f) => f.name === selectedFont)) {
      return
    }
    const newFonts = [...fonts, { name: selectedFont, isDefault: false }]
    saveFonts(newFonts)
    setSelectedFont('')
  }

  const handleRemoveFont = (fontName: string) => {
    const newFonts = fonts.filter((f) => f.name !== fontName)
    if (newFonts.length === 0) {
      return
    }
    saveFonts(newFonts)
  }

  const handleSetDefault = (fontName: string) => {
    const newFonts = fonts.map((f) => ({ ...f, isDefault: f.name === fontName }))
    saveFonts(newFonts)
  }

  const filteredSystemFonts = systemFonts
    .filter((font) => font.toLowerCase().includes(filterText.toLowerCase()))
    .filter((font) => !fonts.some((f) => f.name === font))

  const fontOptions = filteredSystemFonts.map((font) => ({ value: font, label: font }))

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <div className="flex gap-2">
          <div className="flex-1">
            <CustomCombobox
              label=""
              value={selectedFont}
              disabled={isLoadingFonts}
              options={fontOptions}
              onChange={(val) => setSelectedFont(val as string)}
              placeholder="Select a font to add..."
              searchable={true}
              size="sm"
            />
          </div>
          <CustomButton
            variant="primary"
            size="sm"
            icon={Plus}
            onClick={handleAddFont}
            disabled={!selectedFont || fonts.some((f) => f.name === selectedFont)}
          >
            Add Font
          </CustomButton>
        </div>

        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-text-primary">Fonts in Use</h3>
          <div className="space-y-2 max-h-80 overflow-y-auto">
            {fonts.map((font) => (
              <div
                key={font.name}
                className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg border border-border-default"
              >
                <div className="flex items-center gap-3">
                  <span className="text-text-primary font-medium" style={{ fontFamily: font.name }}>
                    {font.name}
                  </span>
                  {font.isDefault && (
                    <span className="px-2 py-1 text-xs bg-blue-100 text-blue-700 dark:bg-blue-900/20 dark:text-blue-300 rounded">
                      Default
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-1">
                  {!font.isDefault && (
                    <CustomButton
                      variant="ghost"
                      size="sm"
                      icon={Star}
                      onClick={() => handleSetDefault(font.name)}
                      className="w-fit"
                      children={undefined}
                    />
                  )}
                  <CustomButton
                    variant="ghost"
                    size="sm"
                    icon={Trash2}
                    onClick={() => handleRemoveFont(font.name)}
                    disabled={fonts.length === 1}
                    className="w-fit text-red-600 hover:text-red-700 hover:bg-red-50 dark:text-red-400 dark:hover:text-red-300 dark:hover:bg-red-900/20"
                    children={undefined}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default FontSection
