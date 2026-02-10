import { useState, useEffect, useRef } from 'react'
import '../styles/SearchBar.css'
import { apiUrl } from '../utils/api';

interface Suggestion {
  ticker: string
  title: string
}

interface SearchBarProps {
  onSelect: (ticker: string) => void;
  placeholder?: string;
}

export default function SearchBar({ onSelect, placeholder = "Search..." }: SearchBarProps) {
  const [input, setInput] = useState<string>('')
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const [_, setLoading] = useState<boolean>(false)
  const [isFocus, setFocus] = useState<boolean>(false)
  const [selectedIndex, setSelectedIndex] = useState<number>(-1)
  const selectedRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!input.trim()) {
      setSuggestions([])
	  setSelectedIndex(-1)
      return
    }

    const debounce = setTimeout(async () => {
      setLoading(true)
      try {
        const response = await fetch(apiUrl(`/api/search?q=${input.toUpperCase()}&limit=10`))
        const data: Suggestion[] = await response.json()
        setSuggestions(data)
		setSelectedIndex(-1)
      } catch (error) {
        console.error('Search failed:', error)
      } finally {
        setLoading(false)
      }
    }, 300)

    return () => clearTimeout(debounce)
  }, [input])

	useEffect(() => {
		if (selectedRef.current) {
		  selectedRef.current.scrollIntoView({ block: 'nearest' })
		}
	}, [selectedIndex])

	const handleSelect = (ticker: string) => {
		onSelect(ticker)
		setSuggestions([])
		setInput('')
		setSelectedIndex(-1)
		setFocus(false)
	}

	const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
		if (e.key === 'ArrowDown') {
		  e.preventDefault()
		  setSelectedIndex(prev => 
			prev < suggestions.length - 1 ? prev + 1 : prev
		  )
		} else if (e.key === 'ArrowUp') {
		  e.preventDefault()
		  setSelectedIndex(prev => prev > 0 ? prev - 1 : -1)
		} else if (e.key === 'Enter') {
		  if (selectedIndex >= 0) {
			handleSelect(suggestions[selectedIndex].ticker)
		  } else if (input.length > 0) {
			handleSelect(input.toUpperCase())
		  }
		} else if (e.key === 'Escape') {
		  setSuggestions([])
		  setSelectedIndex(-1)
		  setFocus(false)
		}
	}

  return (
    <div className="search-box-wrapper">
      <input
        type="text"
        placeholder={placeholder}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
		onFocus={() => setFocus(true)}
		onBlur={() => {
			setTimeout(() => setFocus(false), 200)
		}}
        className="search-input"
        autoComplete="off"
      />
      
      {suggestions.length > 0 && isFocus && (
        <div className="suggestions-dropdown">
          {suggestions.map((item,index) => (
            <div
              key={item.ticker}
			  ref={index === selectedIndex ? selectedRef : null}
              className={`suggestion-item ${index === selectedIndex ? 'selected' : ''}`}
              onClick={() => handleSelect(item.ticker)}
			  onMouseEnter={() => setSelectedIndex(index)}
            >
              <div className="suggestion-ticker">{item.ticker}</div>
              <div className="suggestion-title">{item.title}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
