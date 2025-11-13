import { FC, useState } from 'react'
import CustomDrawer from '../../../../../../components/common/CustomDrawer'
import { ChevronDown } from 'lucide-react'
import AppearanceSection from '../../../../Setting/components/AppearanceSection'
import FontSection from '../../../../Setting/components/FontSection'
import CharacterSection from '../../../../Setting/components/CharacterSection'

interface SettingsDrawerProps {
  isOpen: boolean
  onClose: () => void
}

type SectionKey = 'font' | 'appearance' | 'character'

const SettingsDrawer: FC<SettingsDrawerProps> = ({ isOpen, onClose }) => {
  const [expandedSections, setExpandedSections] = useState<Set<SectionKey>>(new Set())

  const toggleSection = (section: SectionKey) => {
    setExpandedSections((prev) => {
      const newSet = new Set(prev)
      if (newSet.has(section)) {
        newSet.delete(section)
      } else {
        newSet.add(section)
      }
      return newSet
    })
  }

  return (
    <CustomDrawer
      isOpen={isOpen}
      onClose={onClose}
      title="Settings"
      subtitle="Manage your application preferences and settings"
      size="lg"
      direction="right"
      animationType="slide"
      enableBlur={true}
    >
      <div className="h-full overflow-y-auto">
        <div className="w-full px-4 py-4 space-y-3">
          {/* General Section */}
          <div className="bg-card-background rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
            <button
              onClick={() => toggleSection('font')}
              className="w-full flex items-center justify-between p-3 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
            >
              <div className="flex items-center gap-2">
                <div className="p-1.5 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                  <svg
                    className="h-4 w-4 text-blue-600 dark:text-blue-400"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                    />
                  </svg>
                </div>
                <div className="text-left">
                  <h3 className="text-lg font-semibold text-text-primary">General Settings</h3>
                  <p className="text-sm text-text-secondary">Font management and preferences</p>
                </div>
              </div>
              <ChevronDown
                className={`h-4 w-4 text-text-secondary transition-transform duration-200 ${
                  expandedSections.has('font') ? 'rotate-180' : ''
                }`}
              />
            </button>
            {expandedSections.has('font') && (
              <div className="p-4 border-t border-gray-200 dark:border-gray-700">
                <FontSection />
              </div>
            )}
          </div>

          {/* Appearance Section */}
          <div className="bg-card-background rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
            <button
              onClick={() => toggleSection('appearance')}
              className="w-full flex items-center justify-between p-4 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
            >
              <div className="flex items-center gap-3">
                <div className="p-2 bg-pink-100 dark:bg-pink-900/30 rounded-lg">
                  <svg
                    className="h-5 w-5 text-pink-600 dark:text-pink-400"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01"
                    />
                  </svg>
                </div>
                <div className="text-left">
                  <h3 className="text-lg font-semibold text-text-primary">Appearance</h3>
                  <p className="text-sm text-text-secondary">Theme and color customization</p>
                </div>
              </div>
              <ChevronDown
                className={`h-5 w-5 text-text-secondary transition-transform duration-200 ${
                  expandedSections.has('appearance') ? 'rotate-180' : ''
                }`}
              />
            </button>
            {expandedSections.has('appearance') && (
              <div className="p-6 border-t border-gray-200 dark:border-gray-700">
                <AppearanceSection />
              </div>
            )}
          </div>

          {/* Character Section */}
          <div className="bg-card-background rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
            <button
              onClick={() => toggleSection('character')}
              className="w-full flex items-center justify-between p-4 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
            >
              <div className="flex items-center gap-3">
                <div className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
                  <svg
                    className="h-5 w-5 text-purple-600 dark:text-purple-400"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z"
                    />
                  </svg>
                </div>
                <div className="text-left">
                  <h3 className="text-lg font-semibold text-text-primary">Manga & Characters</h3>
                  <p className="text-sm text-text-secondary">Manage manga and character database</p>
                </div>
              </div>
              <ChevronDown
                className={`h-5 w-5 text-text-secondary transition-transform duration-200 ${
                  expandedSections.has('character') ? 'rotate-180' : ''
                }`}
              />
            </button>
            {expandedSections.has('character') && (
              <div className="p-6 border-t border-gray-200 dark:border-gray-700">
                <CharacterSection />
              </div>
            )}
          </div>
        </div>
      </div>
    </CustomDrawer>
  )
}

export default SettingsDrawer
