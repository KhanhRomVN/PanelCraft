import { createContext, useContext, useEffect, useState, useCallback } from 'react'
import { PRESET_THEMES } from '../../components/drawer/themePallate'

type Theme = 'dark' | 'light' | 'system'

type ThemeProviderProps = {
  children: React.ReactNode
  defaultTheme?: Theme
  storageKey?: string
}

type ColorSettings = {
  primary: string
  background: string
  textPrimary: string
  textSecondary: string
  border: string
  borderHover: string
  borderFocus: string
  cardBackground: string
  inputBackground: string
  dialogBackground: string
  dropdownBackground: string
  dropdownItemHover: string
  sidebarBackground: string
  sidebarItemHover: string
  sidebarItemFocus: string
  buttonBg: string
  buttonBgHover: string
  buttonText: string
  buttonBorder: string
  buttonBorderHover: string
  buttonSecondBg: string
  buttonSecondBgHover: string
  bookmarkItemBg: string
  bookmarkItemText: string
  drawerBackground: string
  clockGradientFrom: string
  clockGradientTo: string
  cardShadow?: string
  dialogShadow?: string
  dropdownShadow?: string
}

type ThemeProviderState = {
  theme: Theme
  setTheme: (theme: Theme) => void
  colorSettings: ColorSettings
  setColorSettings: (settings: ColorSettings) => void
}

const getDefaultColorSettings = (themeType: 'light' | 'dark'): ColorSettings => {
  const defaultPreset = PRESET_THEMES[themeType][0]
  return {
    primary: defaultPreset.primary,
    background: defaultPreset.background,
    textPrimary: defaultPreset.textPrimary || '#0f172a',
    textSecondary: defaultPreset.textSecondary || '#475569',
    border: defaultPreset.border || '#e2e8f0',
    borderHover: defaultPreset.borderHover || '#cbd5e1',
    borderFocus: defaultPreset.borderFocus || '#cbd5e1',
    cardBackground: defaultPreset.cardBackground,
    inputBackground: defaultPreset.inputBackground || defaultPreset.cardBackground,
    dialogBackground: defaultPreset.dialogBackground || defaultPreset.cardBackground,
    dropdownBackground: defaultPreset.dropdownBackground || defaultPreset.cardBackground,
    dropdownItemHover: defaultPreset.dropdownItemHover || '#f8fafc',
    sidebarBackground: defaultPreset.sidebarBackground || defaultPreset.cardBackground,
    sidebarItemHover: defaultPreset.sidebarItemHover || '#f3f4f6',
    sidebarItemFocus: defaultPreset.sidebarItemFocus || '#e5e7eb',
    buttonBg: defaultPreset.buttonBg || defaultPreset.primary,
    buttonBgHover: defaultPreset.buttonBgHover || defaultPreset.primary,
    buttonText: defaultPreset.buttonText || '#ffffff',
    buttonBorder: defaultPreset.buttonBorder || defaultPreset.primary,
    buttonBorderHover: defaultPreset.buttonBorderHover || defaultPreset.primary,
    buttonSecondBg: defaultPreset.buttonSecondBg || '#d4d4d4',
    buttonSecondBgHover: defaultPreset.buttonSecondBgHover || '#b6b6b6',
    bookmarkItemBg: defaultPreset.bookmarkItemBg || defaultPreset.cardBackground,
    bookmarkItemText: defaultPreset.bookmarkItemText || defaultPreset.textPrimary || '#0f172a',
    drawerBackground: defaultPreset.drawerBackground || defaultPreset.cardBackground,
    clockGradientFrom: defaultPreset.clockGradientFrom || defaultPreset.primary,
    clockGradientTo: defaultPreset.clockGradientTo || defaultPreset.primary,
    cardShadow: defaultPreset.cardShadow,
    dialogShadow: defaultPreset.dialogShadow,
    dropdownShadow: defaultPreset.dropdownShadow
  }
}

const initialState: ThemeProviderState = {
  theme: 'system',
  setTheme: () => null,
  colorSettings: getDefaultColorSettings('light'),
  setColorSettings: () => {}
}

const ThemeProviderContext = createContext<ThemeProviderState>(initialState)

export function ThemeProvider({
  children,
  defaultTheme = 'light',
  storageKey = 'vite-ui-theme',
  ...props
}: ThemeProviderProps) {
  const [theme, setTheme] = useState<Theme>(defaultTheme)

  // Get the effective theme (resolve 'system' to actual theme)
  const getEffectiveTheme = useCallback((): 'light' | 'dark' => {
    if (theme === 'system') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    }
    return theme
  }, [theme])

  // Initialize color settings based on theme
  const [colorSettings, setColorSettings] = useState<ColorSettings>(() => {
    return getDefaultColorSettings('light')
  })
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const loadTheme = async () => {
      try {
        if (!window.api) {
          console.warn('[ThemeProvider] window.api is not available')
          setIsLoading(false)
          return
        }

        console.log('[ThemeProvider] Loading theme from storage...')
        const [savedTheme, savedColors] = await Promise.all([
          window.api.storage.get(storageKey),
          window.api.storage.get(`${storageKey}-colors`)
        ])

        console.log('[ThemeProvider] Loaded from storage:', { savedTheme, savedColors })

        if (savedTheme) {
          setTheme(savedTheme as Theme)
        }

        if (savedColors) {
          console.log('[ThemeProvider] Applying saved colors:', savedColors)
          setColorSettings(savedColors)
        } else {
          const effectiveTheme =
            savedTheme === 'system'
              ? window.matchMedia('(prefers-color-scheme: dark)').matches
                ? 'dark'
                : 'light'
              : savedTheme || defaultTheme
          const defaultColors = getDefaultColorSettings(effectiveTheme as 'light' | 'dark')
          console.log('[ThemeProvider] No saved colors, creating defaults:', defaultColors)
          setColorSettings(defaultColors)
          await window.api.storage.set(`${storageKey}-colors`, defaultColors)
          console.log('[ThemeProvider] Default colors saved to storage')
        }

        setIsLoading(false)
      } catch (error) {
        console.warn('Failed to load theme settings:', error)
        setIsLoading(false)
      }
    }
    loadTheme()
  }, [storageKey])

  const updateColorSettings = useCallback(
    async (settings: ColorSettings) => {
      console.log('[ThemeProvider] Updating color settings:', settings)
      setColorSettings(settings)
      try {
        if (window.api) {
          console.log('[ThemeProvider] Saving to storage with key:', `${storageKey}-colors`)
          await window.api.storage.set(`${storageKey}-colors`, settings)
          console.log('[ThemeProvider] ✓ Color settings saved successfully')

          const verify = await window.api.storage.get(`${storageKey}-colors`)
          console.log('[ThemeProvider] Verification - Read back from storage:', verify)
        }
      } catch (e) {
        console.error('[ThemeProvider] ✗ Failed to save color settings:', e)
      }
    },
    [storageKey]
  )

  const applyTheme = useCallback(() => {
    const root = window.document.documentElement

    // Clear existing theme classes
    root.classList.remove('light', 'dark')

    const effectiveTheme = getEffectiveTheme()

    // Add the effective theme class
    root.classList.add(effectiveTheme)

    // Apply color settings as CSS custom properties
    const cssVarMap: Record<keyof ColorSettings, string> = {
      primary: '--primary',
      background: '--background',
      textPrimary: '--text-primary',
      textSecondary: '--text-secondary',
      border: '--border',
      borderHover: '--border-hover',
      borderFocus: '--border-focus',
      cardBackground: '--card-background',
      inputBackground: '--input-background',
      dialogBackground: '--dialog-background',
      dropdownBackground: '--dropdown-background',
      dropdownItemHover: '--dropdown-item-hover',
      sidebarBackground: '--sidebar-background',
      sidebarItemHover: '--sidebar-item-hover',
      sidebarItemFocus: '--sidebar-item-focus',
      buttonBg: '--button-bg',
      buttonBgHover: '--button-bg-hover',
      buttonText: '--button-text',
      buttonBorder: '--button-border',
      buttonBorderHover: '--button-border-hover',
      buttonSecondBg: '--button-second-bg',
      buttonSecondBgHover: '--button-second-bg-hover',
      bookmarkItemBg: '--bookmark-item-bg',
      bookmarkItemText: '--bookmark-item-text',
      drawerBackground: '--drawer-background',
      clockGradientFrom: '--clock-gradient-from',
      clockGradientTo: '--clock-gradient-to',
      cardShadow: '--card-shadow',
      dialogShadow: '--dialog-shadow',
      dropdownShadow: '--dropdown-shadow'
    }

    Object.entries(colorSettings).forEach(([key, value]) => {
      const cssVar = cssVarMap[key as keyof ColorSettings]
      if (cssVar && value) {
        root.style.setProperty(cssVar, value)
      }
    })
  }, [colorSettings, getEffectiveTheme])

  // Apply theme when theme or color settings change
  useEffect(() => {
    if (!isLoading) {
      applyTheme()
    }
  }, [applyTheme, isLoading])

  // Listen for system theme changes only when theme is 'system'
  useEffect(() => {
    if (theme !== 'system') return

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleChange = () => {
      // Force a re-render to update the effective theme
      applyTheme()
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [theme, applyTheme])

  const handleSetTheme = useCallback(
    async (newTheme: Theme) => {
      console.log('[ThemeProvider] Changing theme to:', newTheme)
      try {
        if (window.api) {
          await window.api.storage.set(storageKey, newTheme)
          console.log('[ThemeProvider] ✓ Theme saved successfully')
        }
      } catch (e) {
        console.error('[ThemeProvider] ✗ Failed to save theme:', e)
      }
      setTheme(newTheme)
    },
    [storageKey]
  )

  const value = {
    theme,
    setTheme: handleSetTheme,
    colorSettings,
    setColorSettings: updateColorSettings
  }

  return (
    <ThemeProviderContext.Provider {...props} value={value}>
      {children}
    </ThemeProviderContext.Provider>
  )
}

export const useTheme = () => {
  const context = useContext(ThemeProviderContext)

  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }

  return context
}
