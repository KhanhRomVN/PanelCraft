import { useState } from 'react'
import { Palette, Settings as SettingsIcon, Users } from 'lucide-react'
import AppearanceSection from './components/AppearanceSection'
import FontSection from './components/FontSection'
import CharacterSection from './components/CharacterSection'

type SettingTab = 'font' | 'appearance' | 'character'

interface SettingOption {
  id: SettingTab
  label: string
  icon: React.ElementType
}

const settingOptions: SettingOption[] = [
  {
    id: 'font',
    label: 'Font',
    icon: SettingsIcon
  },
  {
    id: 'appearance',
    label: 'Appearance',
    icon: Palette
  },
  {
    id: 'character',
    label: 'Character',
    icon: Users
  }
]

const SettingPage = () => {
  const [activeTab, setActiveTab] = useState<SettingTab>('font')

  const renderContent = () => {
    switch (activeTab) {
      case 'font':
        return <FontSection />
      case 'appearance':
        return <AppearanceSection />
      case 'character':
        return <CharacterSection />
      default:
        return null
    }
  }

  return (
    <div className="h-full flex flex-col">
      <div className="max-w-7xl mx-auto w-full flex flex-col h-full px-6">
        {/* Page Header - Fixed */}
        <div className="flex-shrink-0 pt-6 pb-4">
          <h1 className="text-3xl font-bold text-text-primary mb-2">Settings</h1>
          <p className="text-text-secondary">Manage your application preferences and settings</p>
        </div>

        {/* Main Content Area with Sidebar - Scrollable */}
        <div className="flex-1 flex gap-6 overflow-hidden pb-6">
          {/* Sidebar - Fixed */}
          <div className="w-64 flex-shrink-0">
            <nav className="bg-card-background rounded-xl border border-gray-200 dark:border-gray-700 p-2 sticky top-0">
              {settingOptions.map((option) => {
                const Icon = option.icon
                const isActive = activeTab === option.id

                return (
                  <button
                    key={option.id}
                    onClick={() => setActiveTab(option.id)}
                    className={`
                      w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-all duration-200
                      ${
                        isActive
                          ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 font-medium'
                          : 'text-text-secondary hover:bg-gray-50 dark:hover:bg-gray-700/50'
                      }
                    `}
                  >
                    <Icon
                      className={`h-5 w-5 ${isActive ? 'text-blue-600 dark:text-blue-400' : ''}`}
                    />
                    <span className="text-sm">{option.label}</span>
                  </button>
                )
              })}
            </nav>
          </div>

          {/* Content - Scrollable */}
          <div className="flex-1 overflow-y-auto">
            <div className="bg-card-background rounded-xl border border-gray-200 dark:border-gray-700 p-6">
              {renderContent()}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SettingPage
