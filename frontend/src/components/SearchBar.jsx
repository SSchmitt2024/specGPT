import { useState } from 'react'

export default function SearchBar({ onSearch, loading }) {
  const [query, setQuery] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (query.trim()) {
      onSearch(query.trim())
    }
  }

  return (
    <form className="search-bar" onSubmit={handleSubmit}>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask about the NVMe specification..."
        disabled={loading}
      />
      <button type="submit" disabled={loading || !query.trim()}>
        {loading ? 'Searching...' : 'Ask'}
      </button>
    </form>
  )
}
