import { useState, useEffect } from 'react'
import { Users, Plus, Trash2, Edit2, BookOpen, ChevronRight, ArrowLeft, Info } from 'lucide-react'
import CustomButton from '../../../../components/common/CustomButton'
import CustomInput from '../../../../components/common/CustomInput'
import CustomModal from '../../../../components/common/CustomModal'

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

const STORAGE_KEY = 'app-manga-characters'

type ViewMode = 'manga-list' | 'character-list' | 'character-detail'

const CharacterSection = () => {
  const [mangas, setMangas] = useState<Manga[]>([])
  const [selectedManga, setSelectedManga] = useState<Manga | null>(null)
  const [selectedCharacter, setSelectedCharacter] = useState<Character | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>('manga-list')

  const [showMangaModal, setShowMangaModal] = useState(false)
  const [showCharacterModal, setShowCharacterModal] = useState(false)

  const [editingManga, setEditingManga] = useState<Manga | null>(null)
  const [editingCharacter, setEditingCharacter] = useState<Character | null>(null)

  const [mangaName, setMangaName] = useState('')
  const [characterName, setCharacterName] = useState('')
  const [characterPersonality, setCharacterPersonality] = useState('')
  const [characterNotes, setCharacterNotes] = useState('')

  useEffect(() => {
    loadFromStorage()
  }, [])

  const loadFromStorage = async () => {
    try {
      if (window.api && window.api.storage) {
        const stored = await window.api.storage.get(STORAGE_KEY)
        if (stored) {
          setMangas(stored)
        }
      }
    } catch (error) {
      console.error('[CharacterSection] Failed to load from storage:', error)
    }
  }

  const saveMangas = async (newMangas: Manga[]) => {
    setMangas(newMangas)
    try {
      if (window.api && window.api.storage) {
        await window.api.storage.set(STORAGE_KEY, newMangas)
        console.log('[CharacterSection] Saved to storage successfully')
      }
    } catch (error) {
      console.error('[CharacterSection] Failed to save to storage:', error)
    }
  }

  const handleAddManga = () => {
    if (!mangaName.trim()) return
    const newManga: Manga = {
      id: Date.now().toString(),
      name: mangaName,
      characters: []
    }
    saveMangas([...mangas, newManga])
    setMangaName('')
    setShowMangaModal(false)
  }

  const handleEditManga = () => {
    if (!editingManga || !mangaName.trim()) return
    const updated = mangas.map((m) => (m.id === editingManga.id ? { ...m, name: mangaName } : m))
    saveMangas(updated)
    if (selectedManga?.id === editingManga.id) {
      setSelectedManga({ ...selectedManga, name: mangaName })
    }
    setMangaName('')
    setEditingManga(null)
    setShowMangaModal(false)
  }

  const handleDeleteManga = (id: string) => {
    saveMangas(mangas.filter((m) => m.id !== id))
    if (selectedManga?.id === id) {
      setSelectedManga(null)
      setViewMode('manga-list')
    }
  }

  const handleAddCharacter = () => {
    if (!selectedManga || !characterName.trim()) return
    const newChar: Character = {
      id: Date.now().toString(),
      name: characterName,
      personality: characterPersonality,
      notes: characterNotes
    }
    const updated = mangas.map((m) =>
      m.id === selectedManga.id ? { ...m, characters: [...m.characters, newChar] } : m
    )
    saveMangas(updated)
    setSelectedManga({ ...selectedManga, characters: [...selectedManga.characters, newChar] })
    setCharacterName('')
    setCharacterPersonality('')
    setCharacterNotes('')
    setShowCharacterModal(false)
  }

  const handleEditCharacter = () => {
    if (!selectedManga || !editingCharacter || !characterName.trim()) return
    const updatedChars = selectedManga.characters.map((c) =>
      c.id === editingCharacter.id
        ? { ...c, name: characterName, personality: characterPersonality, notes: characterNotes }
        : c
    )
    const updated = mangas.map((m) =>
      m.id === selectedManga.id ? { ...m, characters: updatedChars } : m
    )
    saveMangas(updated)
    setSelectedManga({ ...selectedManga, characters: updatedChars })

    if (selectedCharacter?.id === editingCharacter.id) {
      setSelectedCharacter({
        id: editingCharacter.id,
        name: characterName,
        personality: characterPersonality,
        notes: characterNotes
      })
    }

    setCharacterName('')
    setCharacterPersonality('')
    setCharacterNotes('')
    setEditingCharacter(null)
    setShowCharacterModal(false)
  }

  const handleDeleteCharacter = (charId: string) => {
    if (!selectedManga) return
    const updatedChars = selectedManga.characters.filter((c) => c.id !== charId)
    const updated = mangas.map((m) =>
      m.id === selectedManga.id ? { ...m, characters: updatedChars } : m
    )
    saveMangas(updated)
    setSelectedManga({ ...selectedManga, characters: updatedChars })

    if (selectedCharacter?.id === charId) {
      setSelectedCharacter(null)
      setViewMode('character-list')
    }
  }

  const handleUpdateCharacterDetail = () => {
    if (!selectedManga || !selectedCharacter || !characterName.trim()) return
    const updatedChars = selectedManga.characters.map((c) =>
      c.id === selectedCharacter.id
        ? { ...c, name: characterName, personality: characterPersonality, notes: characterNotes }
        : c
    )
    const updated = mangas.map((m) =>
      m.id === selectedManga.id ? { ...m, characters: updatedChars } : m
    )
    saveMangas(updated)
    setSelectedManga({ ...selectedManga, characters: updatedChars })
    setSelectedCharacter({
      id: selectedCharacter.id,
      name: characterName,
      personality: characterPersonality,
      notes: characterNotes
    })
  }

  const openAddMangaModal = () => {
    setEditingManga(null)
    setMangaName('')
    setShowMangaModal(true)
  }

  const openEditMangaModal = (manga: Manga) => {
    setEditingManga(manga)
    setMangaName(manga.name)
    setShowMangaModal(true)
  }

  const openAddCharacterModal = () => {
    setEditingCharacter(null)
    setCharacterName('')
    setCharacterPersonality('')
    setCharacterNotes('')
    setShowCharacterModal(true)
  }

  const openEditCharacterModal = (char: Character) => {
    setEditingCharacter(char)
    setCharacterName(char.name)
    setCharacterPersonality(char.personality)
    setCharacterNotes(char.notes)
    setShowCharacterModal(true)
  }

  const openMangaCharacters = (manga: Manga) => {
    setSelectedManga(manga)
    setViewMode('character-list')
  }

  const openCharacterDetail = (char: Character) => {
    setSelectedCharacter(char)
    setCharacterName(char.name)
    setCharacterPersonality(char.personality)
    setCharacterNotes(char.notes)
    setViewMode('character-detail')
  }

  const backToMangaList = () => {
    setSelectedManga(null)
    setSelectedCharacter(null)
    setViewMode('manga-list')
  }

  const backToCharacterList = () => {
    setSelectedCharacter(null)
    setViewMode('character-list')
  }

  return (
    <div className="space-y-6">
      {/* Manga List View */}
      {viewMode === 'manga-list' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-text-primary flex items-center gap-2">
              <BookOpen className="h-5 w-5" />
              Manga List
            </h3>
            <CustomButton variant="primary" size="sm" icon={Plus} onClick={openAddMangaModal}>
              Add Manga
            </CustomButton>
          </div>

          <div className="space-y-2">
            {mangas.map((manga) => (
              <div
                key={manga.id}
                className="flex items-center justify-between p-4 bg-card-background rounded-lg border border-border hover:border-primary/50 transition-all group"
              >
                <div className="flex items-center gap-3 flex-1">
                  <BookOpen className="h-5 w-5 text-text-secondary flex-shrink-0" />
                  <div className="flex-1">
                    <h4 className="text-text-primary font-semibold">{manga.name}</h4>
                    <p className="text-xs text-text-secondary">
                      {manga.characters.length} character{manga.characters.length !== 1 ? 's' : ''}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-1">
                  <CustomButton
                    variant="ghost"
                    size="sm"
                    icon={Edit2}
                    onClick={(e) => {
                      e.stopPropagation()
                      openEditMangaModal(manga)
                    }}
                    className="w-fit"
                    children={undefined}
                  />
                  <CustomButton
                    variant="ghost"
                    size="sm"
                    icon={Trash2}
                    onClick={(e) => {
                      e.stopPropagation()
                      handleDeleteManga(manga.id)
                    }}
                    className="w-fit text-red-600 hover:text-red-700 hover:bg-red-50 dark:text-red-400 dark:hover:text-red-300 dark:hover:bg-red-900/20"
                    children={undefined}
                  />
                  <CustomButton
                    variant="ghost"
                    size="sm"
                    icon={ChevronRight}
                    onClick={() => openMangaCharacters(manga)}
                    className="w-fit"
                    children={undefined}
                  />
                </div>
              </div>
            ))}

            {mangas.length === 0 && (
              <div className="p-8 bg-card-background rounded-lg border border-border text-center">
                <BookOpen className="h-12 w-12 text-text-secondary mx-auto mb-3 opacity-50" />
                <p className="text-text-secondary text-sm">
                  No manga added yet. Click "Add Manga" to start.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Character List View */}
      {viewMode === 'character-list' && selectedManga && (
        <div className="space-y-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3 flex-1 min-w-0">
              <CustomButton
                variant="ghost"
                size="sm"
                icon={ArrowLeft}
                onClick={backToMangaList}
                className="w-fit flex-shrink-0"
                children={undefined}
              />
              <div className="flex items-center gap-2 flex-1 min-w-0">
                <Users className="h-5 w-5 text-text-secondary flex-shrink-0" />
                <h3 className="text-lg font-semibold text-text-primary truncate">
                  Characters - {selectedManga.name}
                </h3>
              </div>
            </div>
            <CustomButton
              variant="primary"
              size="sm"
              icon={Plus}
              onClick={openAddCharacterModal}
              className="flex-shrink-0"
            >
              Add Character
            </CustomButton>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {selectedManga.characters.map((char) => (
              <div
                key={char.id}
                className="p-4 bg-card-background rounded-lg border border-border hover:border-primary/50 transition-all group"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <h4 className="text-text-primary font-semibold mb-2">{char.name}</h4>
                    {char.personality && (
                      <p className="text-xs text-text-secondary line-clamp-2 mb-1">
                        <span className="font-medium">Personality:</span> {char.personality}
                      </p>
                    )}
                  </div>
                  <CustomButton
                    variant="ghost"
                    size="sm"
                    icon={Info}
                    onClick={() => openCharacterDetail(char)}
                    className="w-fit opacity-0 group-hover:opacity-100 transition-opacity"
                    children={undefined}
                  />
                </div>

                <div className="flex items-center gap-1 pt-3 border-t border-border">
                  <CustomButton
                    variant="ghost"
                    size="sm"
                    icon={Edit2}
                    onClick={() => openEditCharacterModal(char)}
                    className="w-fit text-xs"
                  >
                    Quick Edit
                  </CustomButton>
                  <CustomButton
                    variant="ghost"
                    size="sm"
                    icon={Trash2}
                    onClick={() => handleDeleteCharacter(char.id)}
                    className="w-fit text-xs text-red-600 hover:text-red-700 hover:bg-red-50 dark:text-red-400 dark:hover:text-red-300 dark:hover:bg-red-900/20"
                  >
                    Delete
                  </CustomButton>
                </div>
              </div>
            ))}

            {selectedManga.characters.length === 0 && (
              <div className="col-span-full p-8 bg-card-background rounded-lg border border-border text-center">
                <Users className="h-12 w-12 text-text-secondary mx-auto mb-3 opacity-50" />
                <p className="text-text-secondary text-sm">
                  No characters yet. Click "Add Character" to start.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Character Detail View */}
      {viewMode === 'character-detail' && selectedCharacter && selectedManga && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <CustomButton
              variant="ghost"
              size="sm"
              icon={ArrowLeft}
              onClick={backToCharacterList}
              className="flex-shrink-0"
              children={undefined}
            />
            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-semibold text-text-primary truncate">
                Edit Character - {selectedCharacter.name}
              </h3>
              <p className="text-xs text-text-secondary mt-1 truncate">from {selectedManga.name}</p>
            </div>
            <CustomButton
              variant="primary"
              size="sm"
              onClick={handleUpdateCharacterDetail}
              disabled={!characterName.trim()}
              className="flex-shrink-0"
            >
              Save Changes
            </CustomButton>
          </div>

          <div className="max-w-2xl bg-card-background rounded-lg border border-border p-6 space-y-4">
            <CustomInput
              label="Character Name"
              value={characterName}
              onChange={setCharacterName}
              placeholder="Enter character name..."
              required
              size="sm"
            />

            <CustomInput
              label="Personality"
              value={characterPersonality}
              onChange={setCharacterPersonality}
              placeholder="Describe personality traits, speaking style, mannerisms..."
              size="sm"
              multiline
              rows={4}
            />

            <CustomInput
              label="Translation Notes"
              value={characterNotes}
              onChange={setCharacterNotes}
              placeholder="Additional context for translation: catchphrases, speech patterns, background info, relationships..."
              size="sm"
              multiline
              rows={6}
            />
          </div>
        </div>
      )}

      {/* Manga Modal */}
      <CustomModal
        isOpen={showMangaModal}
        onClose={() => {
          setShowMangaModal(false)
          setEditingManga(null)
          setMangaName('')
        }}
        title={editingManga ? 'Edit Manga' : 'Add New Manga'}
        actionText={editingManga ? 'Save Changes' : 'Add Manga'}
        onAction={editingManga ? handleEditManga : handleAddManga}
        actionDisabled={!mangaName.trim()}
        size="sm"
      >
        <div className="p-6">
          <CustomInput
            label="Manga Name"
            value={mangaName}
            onChange={setMangaName}
            placeholder="Enter manga name..."
            required
            size="sm"
          />
        </div>
      </CustomModal>

      {/* Character Quick Edit Modal */}
      <CustomModal
        isOpen={showCharacterModal}
        onClose={() => {
          setShowCharacterModal(false)
          setEditingCharacter(null)
          setCharacterName('')
          setCharacterPersonality('')
          setCharacterNotes('')
        }}
        title={editingCharacter ? 'Quick Edit Character' : 'Add New Character'}
        actionText={editingCharacter ? 'Save Changes' : 'Add Character'}
        onAction={editingCharacter ? handleEditCharacter : handleAddCharacter}
        actionDisabled={!characterName.trim()}
        size="lg"
      >
        <div className="p-6 space-y-4">
          <CustomInput
            label="Character Name"
            value={characterName}
            onChange={setCharacterName}
            placeholder="Enter character name..."
            required
            size="sm"
          />

          <CustomInput
            label="Personality"
            value={characterPersonality}
            onChange={setCharacterPersonality}
            placeholder="Describe personality traits..."
            size="sm"
            multiline
            rows={3}
          />

          <CustomInput
            label="Notes"
            value={characterNotes}
            onChange={setCharacterNotes}
            placeholder="Additional notes for translation context..."
            size="sm"
            multiline
            rows={4}
          />
        </div>
      </CustomModal>
    </div>
  )
}

export default CharacterSection
