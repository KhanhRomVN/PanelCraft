import { FC } from 'react'
import CustomTable from '../../../../../../components/common/CustomTable'

export interface TableRow {
  id: string
  stt: number
  character: string
  original: string
  translate: string
  font: string
  size: number
}

interface OCRResultsTableProps {
  data: TableRow[]
  characterOptions: { value: string; label: string }[]
  selectedManga: string
  onCharacterChange: (rowId: string, characterId: string) => void
  onTranslateChange: (rowId: string, value: string) => void
  onFontChange: (rowId: string, font: string) => void
  onSizeChange: (rowId: string, size: number) => void
}

const OCRResultsTable: FC<OCRResultsTableProps> = ({
  data,
  characterOptions,
  selectedManga,
  onCharacterChange,
  onTranslateChange,
  onFontChange,
  onSizeChange
}) => {
  return (
    <div className="h-full w-full">
      <CustomTable<TableRow>
        data={data}
        columns={[
          {
            accessorKey: 'stt',
            header: 'STT',
            size: 60,
            cell: (info) => (
              <span className="text-text-secondary font-medium">{info.getValue() as number}</span>
            )
          },
          {
            accessorKey: 'character',
            header: 'Character',
            size: 140,
            cell: (info) => {
              const row = info.row.original
              return (
                <select
                  value={row.character}
                  onChange={(e) => onCharacterChange(row.id, e.target.value)}
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
            size: 200,
            cell: (info) => (
              <div className="max-w-[200px] truncate" title={info.getValue() as string}>
                <span className="text-text-primary text-sm">{info.getValue() as string}</span>
              </div>
            )
          },
          {
            accessorKey: 'translate',
            header: 'Translate',
            size: 220,
            cell: (info) => {
              const row = info.row.original
              return (
                <input
                  type="text"
                  value={row.translate}
                  onChange={(e) => onTranslateChange(row.id, e.target.value)}
                  className="w-full px-2 py-1 text-sm bg-transparent border-none focus:outline-none text-text-primary"
                  placeholder="Enter translation..."
                />
              )
            }
          },
          {
            accessorKey: 'font',
            header: 'Font',
            size: 120,
            cell: (info) => {
              const row = info.row.original
              return (
                <select
                  value={row.font}
                  onChange={(e) => onFontChange(row.id, e.target.value)}
                  className="w-full px-2 py-1 text-sm bg-input-background border border-border-default rounded focus:outline-none text-text-primary"
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
            size: 80,
            cell: (info) => {
              const row = info.row.original
              return (
                <input
                  type="number"
                  value={row.size}
                  onChange={(e) => onSizeChange(row.id, parseInt(e.target.value))}
                  className="w-16 px-2 py-1 text-sm bg-input-background border border-border-default rounded focus:outline-none text-text-primary"
                  min="8"
                  max="72"
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
  )
}

export default OCRResultsTable
