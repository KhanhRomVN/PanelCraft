// src/renderer/src/presentation/pages/Home/components/Canvas/index.tsx
import { FC, useState } from 'react'
import CustomButton from '../../../../../components/common/CustomButton'
import { FolderOpen } from 'lucide-react'
import OriginalPanel from './components/OriginalPanel'
import ProcessPanel from './components/ProcessPanel'
import InvalidFilesWarningModal from './components/InvalidFilesWarningModal'

interface CanvasPanelProps {
  selectedFolder: string | null
  onOpenFolder: () => void
}

const CanvasPanel: FC<CanvasPanelProps> = ({ selectedFolder, onOpenFolder }) => {
  const [showWarningModal, setShowWarningModal] = useState(false)
  const [invalidFiles, setInvalidFiles] = useState<string[]>([])
  const [validFiles, setValidFiles] = useState<string[]>([])
  const [pendingFolderPath, setPendingFolderPath] = useState<string | null>(null)
  const [originalPanelWidth, setOriginalPanelWidth] = useState(50)
  const [originalScrollTop, setOriginalScrollTop] = useState<number>()
  const [processScrollTop, setProcessScrollTop] = useState<number>()

  const handleOpenFolder = () => {
    onOpenFolder()
  }

  const handleContinue = () => {
    setShowWarningModal(false)
    onOpenFolder()
  }

  const handleCancelWarning = () => {
    setShowWarningModal(false)
    setInvalidFiles([])
    setValidFiles([])
    setPendingFolderPath(null)
  }

  return (
    <div className="h-full flex bg-card-background">
      <div className="flex-1 flex overflow-hidden">
        {!selectedFolder ? (
          <div className="flex-1 flex items-center justify-center">
            <div className="flex flex-col items-center justify-center gap-6 p-8">
              <div className="w-24 h-24 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
                <FolderOpen className="w-12 h-12 text-gray-400 dark:text-gray-600" />
              </div>

              <div className="text-center space-y-2">
                <h3 className="text-xl font-semibold text-text-primary">No Folder Selected</h3>
                <p className="text-sm text-text-secondary max-w-md">
                  Select a folder to start working with your images. You can use the button below or
                  press{' '}
                  <kbd className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-800 border border-border-default rounded">
                    Ctrl+O
                  </kbd>
                </p>
              </div>

              <CustomButton
                variant="primary"
                size="md"
                icon={FolderOpen}
                onClick={handleOpenFolder}
                className="w-fit"
              >
                Open Folder
              </CustomButton>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex overflow-hidden">
            <div className="h-full" style={{ width: `${originalPanelWidth}%` }}>
              <OriginalPanel
                folderPath={selectedFolder}
                onScrollSync={setProcessScrollTop}
                syncScrollTop={originalScrollTop}
              />
            </div>

            <div className="h-full" style={{ width: `${100 - originalPanelWidth}%` }}>
              <ProcessPanel
                folderPath={selectedFolder}
                onScrollSync={setOriginalScrollTop}
                syncScrollTop={processScrollTop}
              />
            </div>
          </div>
        )}
      </div>

      <InvalidFilesWarningModal
        isOpen={showWarningModal}
        onClose={handleCancelWarning}
        onContinue={handleContinue}
        invalidFiles={invalidFiles}
        validCount={validFiles.length}
      />
    </div>
  )
}

export default CanvasPanel
