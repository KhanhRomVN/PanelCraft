import { useState } from 'react'
import { Palette, Sun, Moon } from 'lucide-react'
import { useTheme } from '../../../../presentation/providers/theme-provider'
import { PRESET_THEMES } from '../../../../components/drawer/themePallate'
import ThemeDrawer from '../../../../components/drawer/ThemeDrawer'

const AppearanceSection = () => {
  const { theme, setTheme, colorSettings, setColorSettings } = useTheme()
  const [isThemeDrawerOpen, setIsThemeDrawerOpen] = useState(false)
  const [showAllPresets, setShowAllPresets] = useState(false)

  const getEffectiveTheme = (): 'light' | 'dark' => {
    if (theme === 'system') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    }
    return theme
  }

  const applyPresetTheme = (preset: any) => {
    const newColorSettings = {
      primary: preset.primary,
      background: preset.background,
      textPrimary: preset.textPrimary || '#0f172a',
      textSecondary: preset.textSecondary || '#475569',
      border: preset.border || '#e2e8f0',
      borderHover: preset.borderHover || '#cbd5e1',
      borderFocus: preset.borderFocus || '#cbd5e1',
      cardBackground: preset.cardBackground,
      inputBackground: preset.inputBackground || preset.cardBackground,
      dialogBackground: preset.dialogBackground || preset.cardBackground,
      dropdownBackground: preset.dropdownBackground || preset.cardBackground,
      dropdownItemHover: preset.dropdownItemHover || '#f8fafc',
      sidebarBackground: preset.sidebarBackground || preset.cardBackground,
      sidebarItemHover: preset.sidebarItemHover || '#f3f4f6',
      sidebarItemFocus: preset.sidebarItemFocus || '#e5e7eb',
      buttonBg: preset.buttonBg || preset.primary,
      buttonBgHover: preset.buttonBgHover || preset.primary,
      buttonText: preset.buttonText || '#ffffff',
      buttonBorder: preset.buttonBorder || preset.primary,
      buttonBorderHover: preset.buttonBorderHover || preset.primary,
      buttonSecondBg: preset.buttonSecondBg || '#d4d4d4',
      buttonSecondBgHover: preset.buttonSecondBgHover || '#b6b6b6',
      bookmarkItemBg: preset.bookmarkItemBg || preset.cardBackground,
      bookmarkItemText: preset.bookmarkItemText || preset.textPrimary || '#0f172a',
      drawerBackground: preset.drawerBackground || preset.cardBackground,
      clockGradientFrom: preset.clockGradientFrom || preset.primary,
      clockGradientTo: preset.clockGradientTo || preset.primary,
      cardShadow: preset.cardShadow,
      dialogShadow: preset.dialogShadow,
      dropdownShadow: preset.dropdownShadow
    }
    setColorSettings(newColorSettings)
  }

  const effectiveTheme = getEffectiveTheme()
  const allPresets = PRESET_THEMES[effectiveTheme]
  const displayedPresets = showAllPresets ? allPresets : allPresets.slice(0, 4)
  const hasMore = allPresets.length > 4

  return (
    <>
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold text-text-primary flex items-center gap-2">
            <Palette className="h-6 w-6 text-pink-600 dark:text-pink-400" />
            Giao diện
          </h2>
          <p className="text-sm text-text-secondary mt-1">
            Tùy chỉnh màu sắc và theme của ứng dụng
          </p>
        </div>

        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-3 text-text-primary">Theme Mode</h3>
            <div className="grid grid-cols-2 gap-3">
              {[
                { mode: 'light', Icon: Sun, label: 'Light' },
                { mode: 'dark', Icon: Moon, label: 'Dark' }
              ].map(({ mode, Icon, label }) => (
                <button
                  key={mode}
                  onClick={() => setTheme(mode as any)}
                  className={`flex flex-col items-center justify-center p-4 rounded-xl border-2 transition-all
                    ${
                      theme === mode
                        ? 'border-primary ring-2 ring-primary/20 bg-primary/5'
                        : 'border-gray-200 dark:border-gray-700 hover:border-primary/50'
                    }`}
                >
                  <div
                    className={`mb-2 p-2 rounded-full ${
                      mode === 'light'
                        ? 'bg-yellow-100 text-yellow-600 dark:bg-yellow-900/30 dark:text-yellow-400'
                        : mode === 'dark'
                          ? 'bg-indigo-100 text-indigo-600 dark:bg-indigo-900/30 dark:text-indigo-400'
                          : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'
                    }`}
                  >
                    <Icon className="h-5 w-5" />
                  </div>
                  <span className="font-medium text-sm text-text-primary">{label}</span>
                </button>
              ))}
            </div>
          </div>

          <div>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {displayedPresets.map((preset, idx) => {
                const isSelected =
                  colorSettings.primary === preset.primary &&
                  colorSettings.background === preset.background &&
                  colorSettings.cardBackground === preset.cardBackground

                return (
                  <button
                    key={idx}
                    onClick={() => applyPresetTheme(preset)}
                    className={`relative flex flex-col p-3 rounded-xl border transition-all
                      ${
                        isSelected
                          ? 'border-primary ring-2 ring-primary/20 shadow-md'
                          : 'border-gray-200 dark:border-gray-700 hover:border-primary/50'
                      }`}
                  >
                    {isSelected && (
                      <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent rounded-xl" />
                    )}
                    <div className="w-full h-16 rounded-lg overflow-hidden mb-2 border border-border relative">
                      <div className="h-3 w-full" style={{ backgroundColor: preset.primary }} />
                      <div className="flex h-13">
                        <div
                          className="w-1/4 h-full"
                          style={{
                            backgroundColor: preset.sidebarBackground || preset.cardBackground
                          }}
                        />
                        <div
                          className="w-3/4 h-full p-1.5"
                          style={{ backgroundColor: preset.background }}
                        >
                          <div
                            className="w-full h-2 rounded mb-1"
                            style={{ backgroundColor: preset.cardBackground }}
                          />
                          <div
                            className="w-3/4 h-2 rounded"
                            style={{ backgroundColor: preset.cardBackground }}
                          />
                        </div>
                      </div>
                    </div>
                    <div className="flex justify-between items-center w-full">
                      <span className="font-medium text-xs text-text-primary">{preset.name}</span>
                      {isSelected && (
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-4 w-4 text-green-500"
                          viewBox="0 0 20 20"
                          fill="currentColor"
                        >
                          <path
                            fillRule="evenodd"
                            d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                            clipRule="evenodd"
                          />
                        </svg>
                      )}
                    </div>
                  </button>
                )
              })}
            </div>
            {hasMore && !showAllPresets && (
              <button
                onClick={() => setShowAllPresets(true)}
                className="w-full mt-3 px-4 py-2 rounded-lg border border-border hover:bg-button-second-bg-hover transition-colors text-sm font-medium text-text-primary"
              >
                Show {allPresets.length - 4} More Themes
              </button>
            )}
            {showAllPresets && (
              <button
                onClick={() => setShowAllPresets(false)}
                className="w-full mt-3 px-4 py-2 rounded-lg border border-border hover:bg-button-second-bg-hover transition-colors text-sm font-medium text-text-primary"
              >
                Show Less
              </button>
            )}
          </div>
        </div>
      </div>

      <ThemeDrawer isOpen={isThemeDrawerOpen} onClose={() => setIsThemeDrawerOpen(false)} />
    </>
  )
}

export default AppearanceSection
