import React from 'react'
import Drawer from 'react-modern-drawer'
import 'react-modern-drawer/dist/index.css'
import { useTheme } from '../../presentation/providers/theme-provider'
import { PRESET_THEMES } from './themePallate'

interface ThemeDrawerProps {
  isOpen: boolean
  onClose: () => void
}

const ThemeDrawer: React.FC<ThemeDrawerProps> = ({ isOpen, onClose }) => {
  const { theme, setTheme, colorSettings, setColorSettings } = useTheme()

  // Apply preset color theme to CSS variables and state
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

  // Icons for theme modes
  const LightIcon = () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      className="h-6 w-6"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707
           M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707
           M16 12a4 4 0 11-8 0 4 4 0 018 0z"
      />
    </svg>
  )

  const DarkIcon = () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      className="h-6 w-6"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M20.354 15.354A9 9 0 018.646 3.646
           9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
      />
    </svg>
  )

  const SystemIcon = () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      className="h-6 w-6"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
      />
    </svg>
  )

  // Theme mode buttons
  const renderThemeSelector = () => (
    <div className="mb-6">
      <h3 className="text-lg font-semibold mb-4">Theme Mode</h3>
      <div className="grid grid-cols-3 gap-4">
        {[
          { mode: 'light', Icon: LightIcon, label: 'Light' },
          { mode: 'dark', Icon: DarkIcon, label: 'Dark' },
          { mode: 'system', Icon: SystemIcon, label: 'System' }
        ].map(({ mode, Icon, label }) => (
          <button
            key={mode}
            onClick={() => setTheme(mode as any)}
            className={`flex flex-col items-center justify-center p-4 rounded-xl border-2 transition-all
              ${
                theme === mode
                  ? 'border-primary ring-2 ring-primary/20'
                  : 'border-gray-200 dark:border-gray-700'
              }
              hover:bg-gray-50 dark:hover:bg-gray-800`}
          >
            <div
              className={`mb-2 p-2 rounded-full ${
                mode === 'light'
                  ? 'bg-yellow-100 text-yellow-600'
                  : mode === 'dark'
                    ? 'bg-indigo-100 text-indigo-600'
                    : 'bg-gray-100 text-gray-600'
              }`}
            >
              <Icon />
            </div>
            <span className="font-medium capitalize">{label}</span>
          </button>
        ))}
      </div>
    </div>
  )

  // Get effective theme for preset selection
  const getEffectiveTheme = (): 'light' | 'dark' => {
    if (theme === 'system') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    }
    return theme
  }

  // Preset color swatches
  const renderPresetThemes = () => {
    const effectiveTheme = getEffectiveTheme()
    const presets = PRESET_THEMES[effectiveTheme]

    return (
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-4">Preset Themes</h3>
        <div className="grid grid-cols-2 gap-4">
          {presets.map((preset, idx) => {
            const isSelected =
              colorSettings.primary === preset.primary &&
              colorSettings.background === preset.background &&
              colorSettings.cardBackground === preset.cardBackground

            return (
              <button
                key={idx}
                onClick={() => applyPresetTheme(preset)}
                className={`relative flex flex-col p-4 rounded-2xl border transition-all overflow-hidden
                  ${
                    isSelected
                      ? 'border-primary ring-4 ring-primary/30 shadow-lg'
                      : 'border-gray-200 dark:border-gray-700'
                  }
                  hover:shadow-md hover:scale-[1.02] duration-200`}
              >
                {isSelected && (
                  <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-transparent" />
                )}
                <div className="w-full h-24 rounded-lg overflow-hidden mb-3 border border-border relative">
                  <div className="h-4 w-full" style={{ backgroundColor: preset.primary }} />
                  <div className="flex h-20">
                    <div
                      className="w-1/4 h-full"
                      style={{
                        backgroundColor: preset.sidebarBackground || preset.cardBackground
                      }}
                    />
                    <div
                      className="w-3/4 h-full p-2"
                      style={{ backgroundColor: preset.background }}
                    >
                      <div
                        className="w-full h-4 rounded mb-1"
                        style={{ backgroundColor: preset.cardBackground }}
                      />
                      <div
                        className="w-3/4 h-4 rounded"
                        style={{ backgroundColor: preset.cardBackground }}
                      />
                    </div>
                  </div>
                  <div className="absolute top-2 right-2 bg-white/80 dark:bg-black/80 p-1.5 rounded-full">
                    {preset.icon ? (
                      <div className="text-lg">{preset.icon}</div>
                    ) : (
                      <div
                        className="w-5 h-5 rounded-full border-2 border-white dark:border-gray-800"
                        style={{ backgroundColor: preset.primary }}
                      />
                    )}
                  </div>
                </div>
                <div className="flex justify-between items-center w-full">
                  <div>
                    <span className="font-medium text-sm block">{preset.name}</span>
                    <span className="text-xs text-text-secondary">
                      {preset.description || 'Modern theme'}
                    </span>
                  </div>
                  {isSelected && (
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-5 w-5 text-green-500 flex-shrink-0"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        fillRule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293
                               a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293
                               a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                        clipRule="evenodd"
                      />
                    </svg>
                  )}
                </div>
                <div className="flex mt-3 gap-1 w-full">
                  {['primary', 'background', 'cardBackground', 'textPrimary'].map((k) => (
                    <div
                      key={k}
                      className="h-2 flex-1 rounded-full"
                      style={{ backgroundColor: (preset as any)[k] || '#000' }}
                    />
                  ))}
                </div>
              </button>
            )
          })}
        </div>
      </div>
    )
  }

  return (
    <Drawer
      open={isOpen}
      onClose={onClose}
      direction="right"
      size="28rem"
      overlayClassName="z-[1500]"
      overlayOpacity={0.3}
      className="!bg-drawer-background"
    >
      <div className="h-full flex flex-col">
        <div className="p-5 border-b border-border bg-card-background">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2
                       h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12
                       a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343
                       M11 7.343l1.657-1.657a2 2 0 012.828 0
                       l2.829 2.829a2 2 0 010 2.828l-8.486 8.485
                       M7 17h.01"
              />
            </svg>
            Theme Settings
          </h2>
          <p className="text-sm text-text-secondary mt-1">
            Customize the look and feel of your app
          </p>
        </div>

        <div className="flex-1 overflow-y-auto p-5">
          {renderThemeSelector()}
          {renderPresetThemes()}
        </div>

        <div className="p-4 border-t border-border bg-card-background flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg border border-border hover:bg-button-second-bg-hover transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-button-bg text-button-text hover:bg-button-bg-hover transition-colors shadow-sm"
          >
            Apply Changes
          </button>
        </div>
      </div>
    </Drawer>
  )
}

export default ThemeDrawer
