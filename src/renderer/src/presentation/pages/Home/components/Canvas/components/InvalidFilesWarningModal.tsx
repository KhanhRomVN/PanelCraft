import { FC } from 'react'
import CustomModal from '../../../../../../components/common/CustomModal'
import { AlertCircle } from 'lucide-react'

interface InvalidFilesWarningModalProps {
  isOpen: boolean
  onClose: () => void
  onContinue: () => void
  invalidFiles: string[]
  validCount: number
}

const InvalidFilesWarningModal: FC<InvalidFilesWarningModalProps> = ({
  isOpen,
  onClose,
  onContinue,
  invalidFiles,
  validCount
}) => {
  return (
    <CustomModal
      isOpen={isOpen}
      onClose={onClose}
      title="Invalid Files Detected"
      size="md"
      cancelText="Cancel"
      actionText="Continue with Valid Files"
      onAction={onContinue}
      actionVariant="primary"
    >
      <div className="p-6 space-y-4">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm text-text-primary">
              The selected folder contains <strong>{invalidFiles.length}</strong> invalid file(s).
              Only <strong>{validCount}</strong> valid image(s) will be loaded.
            </p>
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 max-h-48 overflow-y-auto">
          <p className="text-xs font-semibold text-text-secondary mb-2">Invalid files:</p>
          <ul className="space-y-1">
            {invalidFiles.map((file, index) => (
              <li key={index} className="text-xs text-red-600 dark:text-red-400">
                • {file}
              </li>
            ))}
          </ul>
        </div>

        <p className="text-xs text-text-secondary">
          <strong>Note:</strong> Only .jpg, .jpeg, and .png files in the root folder are accepted.
          Subfolders and other file types are ignored.
        </p>
      </div>
    </CustomModal>
  )
}

export default InvalidFilesWarningModal
